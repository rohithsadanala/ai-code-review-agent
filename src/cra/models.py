"""Pydantic models shared by the pipeline, API and CLI.

Every finding cites the exact file and new-file line range it refers to,
so a reviewer can jump straight to the code — and so the eval harness can
check citation accuracy mechanically.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class Severity(str, Enum):
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    NIT = "nit"


SEVERITY_ORDER = {
    Severity.CRITICAL: 0,
    Severity.MAJOR: 1,
    Severity.MINOR: 2,
    Severity.NIT: 3,
}

Verdict = Literal["approve", "request_changes", "comment"]


class Finding(BaseModel):
    """One review finding. ``line`` / ``end_line`` are line numbers in the
    *new* version of the file (the side a GitHub PR comment attaches to)."""

    file: str = Field(..., description="Repo-relative path of the reviewed file.")
    line: int = Field(..., ge=1, description="First new-file line the finding refers to.")
    end_line: int = Field(..., ge=1, description="Last new-file line the finding refers to.")
    severity: Severity
    category: str = Field(
        ...,
        description="One of: bug, security, performance, style, testing, maintainability.",
    )
    title: str = Field(..., description="Short, specific title.")
    message: str = Field(..., description="What is wrong and why it matters.")
    suggestion: str = Field(
        default="",
        description="Concrete fix; a code snippet where one is useful.",
    )
    rule: str = Field(default="", description="Rule id that produced the finding (mock LLM).")

    def key(self) -> tuple[str, int, str]:
        """Dedup key: same file, same line, same category = same finding."""
        return (self.file, self.line, self.category)


class Hunk(BaseModel):
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    section: str = ""
    # Lines are stored as (kind, old_lineno, new_lineno, text) tuples in the
    # parser; the API model keeps them as plain strings with a prefix.
    lines: list[str] = Field(default_factory=list)


class FileDiff(BaseModel):
    old_path: str
    new_path: str
    is_new: bool = False
    is_deleted: bool = False
    is_binary: bool = False
    hunks: list[Hunk] = Field(default_factory=list)

    @property
    def path(self) -> str:
        return self.new_path if not self.is_deleted else self.old_path


class ParsedDiff(BaseModel):
    files: list[FileDiff] = Field(default_factory=list)

    @property
    def added_lines(self) -> int:
        """Total added lines across all files (used for the report summary)."""
        count = 0
        for f in self.files:
            for h in f.hunks:
                count += sum(1 for ln in h.lines if ln.startswith("+") and not ln.startswith("+++"))
        return count


class ReviewRequest(BaseModel):
    """What the API / CLI accepts to start a review."""

    diff: str = Field(..., description="Unified diff text to review.")
    repo_path: Optional[str] = Field(
        default=None,
        description="Optional local repo path used as RAG context. "
        "Overrides CRA_REPO_PATH for this review.",
    )
    pr: Optional[str] = Field(
        default=None,
        description="Optional 'owner/repo#number' — when set, the diff is "
        "fetched from GitHub instead of using `diff`.",
    )


class ReviewReport(BaseModel):
    """The full, structured output of a review."""

    id: str
    verdict: Verdict
    summary: str
    findings: list[Finding] = Field(default_factory=list)
    files_reviewed: list[str] = Field(default_factory=list)
    hunks_reviewed: int = 0
    added_lines: int = 0
    markdown: str = Field(default="", description="Rendered Markdown report.")
    pr: Optional[str] = None

    def by_severity(self, severity: Severity) -> list[Finding]:
        return [f for f in self.findings if f.severity == severity]


class ReviewSummary(BaseModel):
    """Lightweight list item for GET /reviews."""

    id: str
    verdict: Verdict
    summary: str
    findings: int
    critical: int
    major: int
    files_reviewed: list[str] = Field(default_factory=list)
    pr: Optional[str] = None


class GitHubPostRequest(BaseModel):
    owner: str
    repo: str
    number: int = Field(..., description="Pull request number.")
