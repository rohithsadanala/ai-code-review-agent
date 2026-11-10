"""Per-hunk reviewer.

Builds the review prompt for one hunk — the changed code, the
repository context retrieved for it, and the review rubric — then
runs it through the LangChain chain and parses the JSON findings.

Prompt format (the mock LLM parses this exact layout, so keep it
stable — see :mod:`cra.llm.MockLLM`):

    File: <repo-relative path>
    Hunk: @@ -a,b +c,d @@ <section>

    Repository context:
    <retrieved chunks, or "(no repository context available)">

    CODE TO REVIEW (lines marked + are new code, ' ' is unchanged
    context, - is removed; each line is L<new-line>|<marker>|<code>):
    L42| |def get_user(user_id):
    L43|+|    query = f"SELECT ..."

    Return ONLY a JSON array of findings: ...

Findings are validated against the :class:`cra.models.Finding` schema
and clamped to lines that actually exist in the hunk, so a sloppy
model cannot invent citations.
"""

from __future__ import annotations

import json
import re

from . import llm as llm_mod
from .context import ContextChunk, format_context
from .diff_parser import ParsedFile, ParsedHunk
from .models import Finding, Severity

REVIEW_INSTRUCTIONS = """You are a senior software engineer reviewing a pull request.
Review ONLY the lines marked with + (added code). Use the unchanged
context lines and the repository context to understand intent, but do
not flag pre-existing code.

Check for, in this order:
1. BUGS — logic errors, wrong conditions, off-by-one, None handling,
   mutable default arguments, swallowed exceptions, resource leaks.
2. SECURITY — SQL/command injection, hardcoded secrets, unsafe
   deserialization, TLS verification disabled, weak crypto, insecure
   randomness for tokens, path traversal.
3. PERFORMANCE — N+1 queries, O(n^2) in a hot path, repeated work in
   a loop, missing pagination, unbounded collections.
4. STYLE — None comparisons (use `is None`), print() in production
   code, TODO/FIXME left in, inconsistent naming.
5. TESTING — new public behaviour with no test (only flag this when
   the change clearly adds behaviour and the repository context shows
   no test covers it).
6. MAINTAINABILITY — very long functions, duplicated logic, unclear
   names.

Return ONLY a JSON array of findings. Each finding must have:
  file (the repo-relative path), line (new-file line number),
  end_line, severity ("critical" | "major" | "minor" | "nit"),
  category ("bug" | "security" | "performance" | "style" |
  "testing" | "maintainability"), title, message, suggestion
  (a concrete fix, with a code snippet where useful).

Do not report a finding unless you are confident it is real.
If the code is clean, return an empty array: [].
"""

_VALID_CATEGORIES = {
    "bug", "security", "performance", "style", "testing", "maintainability",
}


def build_prompt(
    file: ParsedFile,
    hunk: ParsedHunk,
    context_chunks: list[ContextChunk],
) -> str:
    """Render the review prompt for one hunk."""
    code_lines: list[str] = []
    marker = {"add": "+", "del": "-", "context": " "}
    for ln in hunk.lines:
        lineno = ln.new_lineno if ln.new_lineno is not None else ln.old_lineno
        code_lines.append(f"L{lineno}|{marker[ln.kind]}|{ln.text}")

    return (
        f"File: {file.path}\n"
        f"Hunk: @@ -{hunk.old_start},{hunk.old_count} "
        f"+{hunk.new_start},{hunk.new_count} @@ {hunk.section}\n\n"
        "Repository context:\n"
        f"{format_context(context_chunks)}\n\n"
        "CODE TO REVIEW (lines marked + are new code, ' ' is unchanged "
        "context, - is removed):\n"
        + "\n".join(code_lines)
        + "\n\n"
        + REVIEW_INSTRUCTIONS
    )


def review_hunk(
    chain,
    file: ParsedFile,
    hunk: ParsedHunk,
    context_chunks: list[ContextChunk],
) -> list[Finding]:
    """Run one hunk through the chain and return validated findings."""
    prompt = build_prompt(file, hunk, context_chunks)
    raw = chain.invoke({"prompt": prompt})
    parsed = llm_mod.extract_json_array(raw)
    valid_lines = {
        ln.new_lineno
        for ln in hunk.lines
        if ln.new_lineno is not None
    }
    findings: list[Finding] = []
    for item in parsed:
        try:
            f = _coerce_finding(item, file.path, valid_lines)
        except (ValueError, TypeError, KeyError):
            continue  # drop malformed model output, never crash the review
        if f is not None:
            findings.append(f)
    return findings


def _coerce_finding(
    item: dict, default_file: str, valid_lines: set[int]
) -> Finding | None:
    """Validate one raw finding dict; return None if it is unusable."""
    line = int(item.get("line") or 0)
    if line not in valid_lines:
        # A finding must cite a line that exists in the hunk.
        return None
    end_line = int(item.get("end_line") or line)
    if end_line < line:
        end_line = line
    severity_raw = str(item.get("severity", "minor")).lower()
    severity = (
        Severity(severity_raw)
        if severity_raw in {s.value for s in Severity}
        else Severity.MINOR
    )
    category = str(item.get("category", "bug")).lower()
    if category not in _VALID_CATEGORIES:
        category = "bug"
    title = str(item.get("title", "")).strip()
    message = str(item.get("message", "")).strip()
    if not title or not message:
        return None
    return Finding(
        file=str(item.get("file") or default_file),
        line=line,
        end_line=end_line,
        severity=severity,
        category=category,
        title=title,
        message=message,
        suggestion=str(item.get("suggestion", "")),
        rule=str(item.get("rule", "")),
    )


def symbols_in_hunk(hunk: ParsedHunk) -> list[str]:
    """Function/class names defined in the added lines — used as RAG
    queries to find their usages elsewhere in the repo."""
    names: list[str] = []
    for ln in hunk.added():
        m = re.search(r"(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)", ln.text)
        if m:
            names.append(m.group(1))
        m = re.search(r"class\s+([A-Za-z_][A-Za-z0-9_]*)", ln.text)
        if m:
            names.append(m.group(1))
    return names
