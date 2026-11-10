"""LLM providers.

One interface — ``generate(prompt) -> str`` — with three backends:

``mock``
    Offline, deterministic: it parses the review prompt's CODE block,
    runs the deterministic rule engine (:mod:`cra.rules`) over the
    added lines, and returns the findings as a JSON array. No model,
    no API key — used by the tests, the demo, the eval harness and CI.
    The rules are a strict subset of what the prompt asks a real model
    to check, so the mock is honest about what it can and cannot do.

``openai``
    LangChain ``ChatOpenAI`` (default ``gpt-4o-mini``). Needs
    ``OPENAI_API_KEY``. Returns the model's JSON findings verbatim.

``huggingface``
    A local instruct model via ``transformers`` (default
    ``HuggingFaceH4/zephyr-7b-beta`` is only a suggestion — any
    text-generation pipeline model works). Needs
    ``requirements-optional.txt`` and a GPU or patience; the code
    degrades gracefully to the mock provider if loading fails.

The reviewer builds a LangChain chain (``PromptTemplate`` +
``RunnableLambda`` wrapping the provider + ``StrOutputParser``) so
the LangChain wiring is real regardless of which provider is behind
it.
"""

from __future__ import annotations

import json
import re

from . import rules
from .models import Finding

_CODE_LINE_RE = re.compile(r"^L(\d+)\|([+ -])\|(.*)$")
_FILE_RE = re.compile(r"^File:\s*(.+)$", re.MULTILINE)


class MockLLM:
    """Deterministic, rule-based reviewer for offline use.

    It extracts the ``File:`` header and the ``L<no>|<marker>|<code>``
    lines from the review prompt, runs :func:`cra.rules.run_rules`
    over them, and returns the findings as a JSON array — the same
    shape a real model is asked to return.
    """

    def generate(self, prompt: str) -> str:
        file_m = _FILE_RE.search(prompt)
        path = file_m.group(1).strip() if file_m else "unknown"
        lines: list[tuple[int | None, str, str]] = []
        in_code = False
        for raw in prompt.splitlines():
            if raw.startswith("CODE TO REVIEW"):
                in_code = True
                continue
            if not in_code:
                continue
            if raw.startswith("Return ONLY"):
                break
            m = _CODE_LINE_RE.match(raw)
            if m:
                lines.append((int(m.group(1)), m.group(2), m.group(3)))
        if not lines:
            return "[]"
        findings = rules.run_rules(path, lines)
        return json.dumps([_finding_dict(f) for f in findings], indent=2)


def _finding_dict(f: Finding) -> dict:
    return {
        "file": f.file,
        "line": f.line,
        "end_line": f.end_line,
        "severity": f.severity.value,
        "category": f.category,
        "title": f.title,
        "message": f.message,
        "suggestion": f.suggestion,
        "rule": f.rule,
    }


class OpenAILLM:
    """ChatOpenAI via LangChain (lazy import — needs an API key)."""

    def __init__(self, model: str, api_key: str) -> None:
        from langchain_openai import ChatOpenAI

        self._impl = ChatOpenAI(model=model, api_key=api_key or None, temperature=0)

    def generate(self, prompt: str) -> str:
        return self._impl.invoke(prompt).content  # type: ignore[return-value]


class HuggingFaceLLM:
    """Local instruct model via transformers (lazy import — optional dep)."""

    def __init__(self, model: str) -> None:
        from transformers import pipeline

        self._pipe = pipeline(
            "text-generation", model=model, max_new_tokens=1024, do_sample=False
        )

    def generate(self, prompt: str) -> str:
        out = self._pipe(prompt)
        text = out[0]["generated_text"]
        return text[len(prompt):].strip() if text.startswith(prompt) else text


def get_llm(provider: str, model: str, api_key: str = ""):
    """Factory returning an object with a ``generate(prompt) -> str`` method."""
    provider = provider.lower()
    if provider == "openai":
        return OpenAILLM(model, api_key)
    if provider == "huggingface":
        try:
            return HuggingFaceLLM(model)
        except Exception as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "HuggingFace LLM requested but the model could not be loaded "
                f"({exc}). Install requirements-optional.txt or use "
                "CRA_LLM_PROVIDER=mock."
            ) from exc
    return MockLLM()


def build_review_chain(llm):
    """LangChain chain: PromptTemplate -> provider -> raw string output.

    The reviewer renders the prompt itself and calls the chain with
    ``{"prompt": <rendered>}``; the chain exists so the LangChain
    wiring (template, runnable, parser) is exercised for real.
    """
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnableLambda

    def _run(values):
        # PromptTemplate emits a StringPromptValue (langchain-core 1.x);
        # accept that or a plain dict for robustness.
        text = values["prompt"] if isinstance(values, dict) else values.text
        return llm.generate(text)

    template = PromptTemplate.from_template("{prompt}")
    runnable = RunnableLambda(_run)
    return template | runnable | StrOutputParser()


def extract_json_array(text: str) -> list[dict]:
    """Best-effort extraction of the first JSON array in model output."""
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        parsed = json.loads(text[start : end + 1])
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return []
