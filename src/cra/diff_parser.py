"""Unified diff parser.

Parses ``git diff`` / ``git format-patch`` style unified diffs into
:class:`ParsedDiff` — per-file hunks with exact new-file line numbers.
Line numbers matter: every finding cites the new-file line range it
refers to, and GitHub PR review comments attach to the new side.

The parser is deliberately dependency-free and tolerant: it skips
headers it does not understand instead of failing the whole diff.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .models import FileDiff, Hunk, ParsedDiff

_DIFF_GIT_RE = re.compile(r"^diff --git a/(.*) b/(.*)$")
_HUNK_RE = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?: (.*))?$"
)
_BINARY_RE = re.compile(r"^Binary files .* differ$")


@dataclass
class DiffLine:
    """One line inside a hunk, with its line numbers on both sides."""

    kind: str  # "add" | "del" | "context"
    old_lineno: int | None
    new_lineno: int | None
    text: str


@dataclass
class ParsedHunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    section: str = ""
    lines: list[DiffLine] = field(default_factory=list)

    def added(self) -> list[DiffLine]:
        return [ln for ln in self.lines if ln.kind == "add"]

    def to_model(self) -> Hunk:
        """Serialisable form: each line is prefixed with its diff marker."""
        prefix = {"add": "+", "del": "-", "context": " "}
        return Hunk(
            old_start=self.old_start,
            old_count=self.old_count,
            new_start=self.new_start,
            new_count=self.new_count,
            section=self.section,
            lines=[f"{prefix[ln.kind]}{ln.text}" for ln in self.lines],
        )


@dataclass
class ParsedFile:
    old_path: str
    new_path: str
    is_new: bool = False
    is_deleted: bool = False
    is_binary: bool = False
    hunks: list[ParsedHunk] = field(default_factory=list)

    @property
    def path(self) -> str:
        return self.new_path if not self.is_deleted else self.old_path

    def to_model(self) -> FileDiff:
        return FileDiff(
            old_path=self.old_path,
            new_path=self.new_path,
            is_new=self.is_new,
            is_deleted=self.is_deleted,
            is_binary=self.is_binary,
            hunks=[h.to_model() for h in self.hunks],
        )


def parse_diff(diff_text: str) -> tuple[ParsedDiff, list[ParsedFile]]:
    """Parse unified diff text.

    Returns ``(model, parsed_files)`` — the serialisable model for the
    API/report, and the richer parsed objects the reviewer works with.
    """
    files: list[ParsedFile] = []
    current: ParsedFile | None = None
    hunk: ParsedHunk | None = None
    old_lineno = new_lineno = 0
    in_hunk = False  # True between an @@ line and the next diff --git.

    def flush_hunk() -> None:
        nonlocal hunk
        if hunk is not None and current is not None:
            current.hunks.append(hunk)
        hunk = None

    def flush_file() -> None:
        nonlocal current
        flush_hunk()
        if current is not None:
            files.append(current)
        current = None

    for raw in diff_text.splitlines():
        line = raw.rstrip("\n")

        m = _DIFF_GIT_RE.match(line)
        if m:
            flush_file()
            current = ParsedFile(old_path=m.group(1), new_path=m.group(2))
            in_hunk = False
            continue

        if _BINARY_RE.match(line):
            if current is None:
                current = ParsedFile(old_path="", new_path="")
            current.is_binary = True
            continue

        # --- / +++ are file headers ONLY outside a hunk body. Inside a
        # hunk, a removed line starting with "--" (or an added line
        # starting with "++") looks identical, so it must be code.
        if not in_hunk and line.startswith("--- "):
            path = line[4:]
            if path.startswith("a/"):
                path = path[2:]
            if current is None:
                current = ParsedFile(old_path=path, new_path=path)
            else:
                current.old_path = path
            if path == "/dev/null":
                current.is_new = True
            continue

        if not in_hunk and line.startswith("+++ "):
            path = line[4:]
            if path.startswith("b/"):
                path = path[2:]
            if current is None:
                current = ParsedFile(old_path=path, new_path=path)
            else:
                current.new_path = path
            if path == "/dev/null":
                current.is_deleted = True
            continue

        m = _HUNK_RE.match(line)
        if m:
            if current is None:
                # A hunk with no file header — skip it.
                hunk = None
                continue
            flush_hunk()
            old_start = int(m.group(1))
            old_count = int(m.group(2)) if m.group(2) is not None else 1
            new_start = int(m.group(3))
            new_count = int(m.group(4)) if m.group(4) is not None else 1
            hunk = ParsedHunk(
                old_start=old_start,
                old_count=old_count,
                new_start=new_start,
                new_count=new_count,
                section=(m.group(5) or "").strip(),
            )
            old_lineno, new_lineno = old_start, new_start
            in_hunk = True
            continue

        if line.startswith("\\"):
            # "\ No newline at end of file" — carries no code.
            continue

        if hunk is not None and current is not None and not current.is_binary:
            if line.startswith("+"):
                hunk.lines.append(
                    DiffLine("add", None, new_lineno, line[1:])
                )
                new_lineno += 1
            elif line.startswith("-"):
                hunk.lines.append(
                    DiffLine("del", old_lineno, None, line[1:])
                )
                old_lineno += 1
            elif line.startswith(" "):
                hunk.lines.append(
                    DiffLine("context", old_lineno, new_lineno, line[1:])
                )
                old_lineno += 1
                new_lineno += 1
            elif line == "":
                # An empty line inside a hunk is a context line with no
                # leading space (some diffs omit the marker).
                hunk.lines.append(
                    DiffLine("context", old_lineno, new_lineno, "")
                )
                old_lineno += 1
                new_lineno += 1
            # Any other line inside a hunk is ignored (e.g. mode lines).

    flush_file()

    model = ParsedDiff(files=[f.to_model() for f in files])
    return model, files


def added_lines_by_file(parsed: list[ParsedFile]) -> dict[str, list[tuple[int, str]]]:
    """Map file path -> [(new_lineno, code), ...] for added lines only."""
    out: dict[str, list[tuple[int, str]]] = {}
    for f in parsed:
        if f.is_deleted or f.is_binary:
            continue
        lines: list[tuple[int, str]] = []
        for h in f.hunks:
            for ln in h.added():
                if ln.new_lineno is not None:
                    lines.append((ln.new_lineno, ln.text))
        if lines:
            out[f.path] = lines
    return out


def hunk_added_text(hunk: ParsedHunk) -> str:
    """Added lines of a hunk, each prefixed with its new-file line number
    (``L42: <code>``) — the format the reviewer prompt uses so findings
    can cite exact lines."""
    return "\n".join(
        f"L{ln.new_lineno}: {ln.text}" for ln in hunk.added() if ln.new_lineno is not None
    )


def hunk_context_text(hunk: ParsedHunk) -> str:
    """Full hunk (context + added + removed) with new-file line numbers
    where they exist — gives the reviewer the surrounding code."""
    parts: list[str] = []
    for ln in hunk.lines:
        marker = {"add": "+", "del": "-", "context": " "}[ln.kind]
        lineno = ln.new_lineno if ln.new_lineno is not None else ln.old_lineno
        parts.append(f"{marker} L{lineno}: {ln.text}")
    return "\n".join(parts)
