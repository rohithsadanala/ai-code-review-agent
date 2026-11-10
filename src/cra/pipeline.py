"""Review pipeline as a LangGraph state machine.

    parse_diff → gather_context → review_hunks → dedupe_prioritize → render_report

Each step is a node with a single job, so the pipeline is explicit,
inspectable and extensible (add a node for, say, a security-only
second pass without touching the others). The graph runs in-process
with no checkpointer — state is a plain dict of objects.

Context gathering: if a repo path is configured, a hybrid BM25+FAISS
index is built over it and, per hunk, the reviewer gets the chunks
most relevant to the changed code plus the usages of any function or
class the hunk defines. Without a repo path the review still runs —
context is simply empty.
"""

from __future__ import annotations

import uuid
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from . import dedup as dedup_mod
from . import report as report_mod
from .config import Settings
from .context import RepoIndex
from .diff_parser import parse_diff
from .llm import build_review_chain, get_llm
from .models import Finding, ReviewReport
from .reviewer import review_hunk, symbols_in_hunk


class ReviewState(TypedDict, total=False):
    raw_diff: str
    repo_path: str
    pr: str | None
    settings: Settings
    parsed_model: Any
    parsed_files: Any
    index: RepoIndex | None
    findings: list[Finding]
    report: ReviewReport


def _parse_diff(state: ReviewState) -> dict:
    model, files = parse_diff(state["raw_diff"])
    return {"parsed_model": model, "parsed_files": files}


def _gather_context(state: ReviewState) -> dict:
    repo_path = state.get("repo_path") or ""
    if not repo_path:
        return {"index": None}
    settings = state["settings"]
    try:
        index = RepoIndex.build(
            repo_path,
            embedding_provider=settings.embedding_provider,
            embedding_model=settings.embedding_model,
            embedding_dim=settings.embedding_dim,
            api_key=settings.openai_api_key,
        )
    except Exception:
        # A broken index must not kill the review — fall back to no context.
        return {"index": None}
    return {"index": index}


def _review_hunks(state: ReviewState) -> dict:
    settings = state["settings"]
    llm = get_llm(settings.llm_provider, settings.llm_model, settings.openai_api_key)
    chain = build_review_chain(llm)
    index = state.get("index")
    findings: list[Finding] = []
    for file in state["parsed_files"]:
        if file.is_deleted or file.is_binary:
            continue
        for hunk in file.hunks:
            if not hunk.added():
                continue
            chunks = []
            if index is not None:
                queries = symbols_in_hunk(hunk)
                query_text = " ".join(queries) if queries else file.path
                chunks = index.retrieve(query_text, k=settings.context_k)
                for symbol in queries:
                    for chunk in index.find_usages(symbol, k=1):
                        if chunk not in chunks:
                            chunks.append(chunk)
            findings.extend(review_hunk(chain, file, hunk, chunks))
    return {"findings": findings}


def _dedupe_prioritize(state: ReviewState) -> dict:
    settings = state["settings"]
    findings = dedup_mod.dedupe(state.get("findings", []))
    findings = dedup_mod.prioritize(findings, settings.max_findings)
    return {"findings": findings}


def _render_report(state: ReviewState) -> dict:
    settings = state["settings"]
    findings: list[Finding] = state.get("findings", [])
    model = state["parsed_model"]
    files = state["parsed_files"]
    files_reviewed = [f.path for f in files if not f.is_deleted and not f.is_binary]
    hunks = sum(len(f.hunks) for f in files)
    verdict = dedup_mod.verdict_for(findings)
    summary = dedup_mod.summarize(findings, files_reviewed, hunks, model.added_lines)
    markdown = report_mod.render_markdown(
        verdict, summary, findings, pr=state.get("pr")
    )
    report = ReviewReport(
        id=str(uuid.uuid4())[:8],
        verdict=verdict,
        summary=summary,
        findings=findings,
        files_reviewed=files_reviewed,
        hunks_reviewed=hunks,
        added_lines=model.added_lines,
        markdown=markdown,
        pr=state.get("pr"),
    )
    return {"report": report}


def build_graph() -> Any:
    graph = StateGraph(ReviewState)
    graph.add_node("parse_diff", _parse_diff)
    graph.add_node("gather_context", _gather_context)
    graph.add_node("review_hunks", _review_hunks)
    graph.add_node("dedupe_prioritize", _dedupe_prioritize)
    graph.add_node("render_report", _render_report)
    graph.set_entry_point("parse_diff")
    graph.add_edge("parse_diff", "gather_context")
    graph.add_edge("gather_context", "review_hunks")
    graph.add_edge("review_hunks", "dedupe_prioritize")
    graph.add_edge("dedupe_prioritize", "render_report")
    graph.add_edge("render_report", END)
    return graph.compile()


def run_review(
    diff_text: str,
    settings: Settings | None = None,
    repo_path: str = "",
    pr: str | None = None,
) -> ReviewReport:
    """Run the full pipeline on a unified diff and return the report."""
    settings = settings or Settings()
    app = build_graph()
    result = app.invoke(
        {
            "raw_diff": diff_text,
            "repo_path": repo_path or settings.repo_path,
            "pr": pr,
            "settings": settings,
        }
    )
    return result["report"]
