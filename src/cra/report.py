"""Render a review as Markdown (human) — the JSON form is the
:class:`cra.models.ReviewReport` itself."""

from __future__ import annotations

from .models import Finding, Severity

_SEVERITY_EMOJI = {
    Severity.CRITICAL: "🔴",
    Severity.MAJOR: "🟠",
    Severity.MINOR: "🟡",
    Severity.NIT: "⚪",
}

_VERDICT_TEXT = {
    "approve": "✅ Approve",
    "request_changes": "🔴 Request changes",
    "comment": "💬 Comment",
}


def _render_suggestion(lines: list[str], suggestion: str) -> None:
    """Render a suggestion: prose first, then a code block if the
    suggestion separates prose and code with a blank line."""
    parts = suggestion.split("\n\n", 1)
    if len(parts) == 2:
        lines.append(parts[0].strip())
        lines.append("")
        lines.append("```python")
        lines.append(parts[1].strip())
        lines.append("```")
    elif suggestion.startswith(("def ", "    ", "cursor", "subprocess", "import ")):
        lines.append("```python")
        lines.append(suggestion)
        lines.append("```")
    else:
        lines.append(suggestion)


def render_markdown(
    verdict: str,
    summary: str,
    findings: list[Finding],
    pr: str | None = None,
) -> str:
    lines: list[str] = ["# Code Review\n"]
    if pr:
        lines.append(f"**Pull request:** {pr}\n")
    lines.append(f"**Verdict:** {_VERDICT_TEXT.get(verdict, verdict)}\n")
    lines.append(summary)
    lines.append("")

    if not findings:
        lines.append("No issues found. 🎉\n")
        return "\n".join(lines)

    for severity in Severity:
        group = [f for f in findings if f.severity == severity]
        if not group:
            continue
        lines.append(f"## {_SEVERITY_EMOJI[severity]} {severity.value.title()} "
                     f"({len(group)})\n")
        for f in group:
            loc = f"{f.file}:{f.line}" + (
                f"-{f.end_line}" if f.end_line != f.line else ""
            )
            lines.append(f"### {f.title} — `{loc}`\n")
            lines.append(f"**Category:** {f.category}\n")
            lines.append(f.message)
            lines.append("")
            if f.suggestion:
                lines.append("**Suggested fix:**")
                lines.append("")
                _render_suggestion(lines, f.suggestion)
                lines.append("")
    return "\n".join(lines)


def render_github_comment_body(finding: Finding) -> str:
    """Body for one inline PR review comment."""
    body = f"**{finding.severity.value.upper()} — {finding.title}**\n\n{finding.message}"
    if finding.suggestion:
        parts = finding.suggestion.split("\n\n", 1)
        if len(parts) == 2:
            body += f"\n\n**Suggested fix:** {parts[0]}\n```python\n{parts[1]}\n```"
        else:
            body += f"\n\n**Suggested fix:**\n```python\n{finding.suggestion}\n```"
    return body
