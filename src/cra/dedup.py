"""Deduplication, prioritisation and verdict for review findings."""

from __future__ import annotations

from .models import SEVERITY_ORDER, Finding, Severity, Verdict


def dedupe(findings: list[Finding]) -> list[Finding]:
    """Drop exact duplicates — same file, same line, same category —
    keeping the first (highest-severity) occurrence."""
    seen: set[tuple[str, int, str]] = set()
    out: list[Finding] = []
    for f in findings:
        key = f.key()
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
    return out


def prioritize(findings: list[Finding], max_findings: int = 25) -> list[Finding]:
    """Sort by severity, then file, then line; cap the total.

    The cap keeps the report readable and the GitHub review
    comment count bounded — the most severe findings always win.
    """
    ordered = sorted(
        findings,
        key=lambda f: (SEVERITY_ORDER[f.severity], f.file, f.line),
    )
    return ordered[:max_findings]


def verdict_for(findings: list[Finding]) -> Verdict:
    """Any critical finding blocks the PR; major/minor findings get a
    comment; only nits (or nothing) means approve."""
    severities = {f.severity for f in findings}
    if Severity.CRITICAL in severities:
        return "request_changes"
    if Severity.MAJOR in severities or Severity.MINOR in severities:
        return "comment"
    return "approve"


def summarize(findings: list[Finding], files: list[str], hunks: int, added: int) -> str:
    """One-paragraph summary for the report header."""
    counts = {
        sev: sum(1 for f in findings if f.severity == sev) for sev in Severity
    }
    parts = [f"{counts[sev]} {sev.value}" for sev in Severity if counts[sev] > 0]
    if not parts:
        return (
            f"Reviewed {len(files)} file(s) ({hunks} hunks, {added} added lines). "
            "No issues found — the change looks clean."
        )
    return (
        f"Reviewed {len(files)} file(s) ({hunks} hunks, {added} added lines). "
        f"Found {len(findings)} issue(s): " + ", ".join(parts) + "."
    )
