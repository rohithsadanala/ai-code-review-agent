"""Deterministic rule engine.

This is the offline brain of the mock LLM: given the added lines of a
hunk (plus their unchanged context), it finds the issues a reviewer
would flag — bugs, security holes, performance traps and style
problems — each with an exact file + line citation, a severity and a
concrete fix.

The real LLM (OpenAI / Hugging Face) receives the same categories in
its prompt and is free to find anything else; the rules exist so the
system is useful with no API key at all. Rules are intentionally
conservative: a rule only fires on a pattern that is almost always a
real problem, to keep the false-positive rate low.

``run_rules`` takes the hunk as ``(lineno, marker, text)`` triples,
where ``marker`` is ``+`` (added), ``-`` (removed) or `` `` (context),
and ``lineno`` is the new-file line number (``None`` for removed
lines). Only *added* lines are flagged — reviewing unchanged code
would spam every PR with pre-existing issues.
"""

from __future__ import annotations

import re

from .models import Finding, Severity

# A parsed hunk line: (new-file lineno or None, marker, code).
ParsedLine = tuple[int | None, str, str]

_SQL_KEYWORDS = ("SELECT", "INSERT", "UPDATE", "DELETE", "WHERE", "FROM")
_DB_CALLS = (".execute(", ".executemany(", ".query(", "session.get(", "objects.get(",
             ".filter(", ".find_one(", ".find(", "cursor.execute(")


def _finding(
    file: str,
    line: int,
    end_line: int,
    severity: Severity,
    category: str,
    title: str,
    message: str,
    suggestion: str,
    rule: str,
) -> Finding:
    return Finding(
        file=file,
        line=line,
        end_line=end_line,
        severity=severity,
        category=category,
        title=title,
        message=message,
        suggestion=suggestion,
        rule=rule,
    )


def _is_test_file(path: str) -> bool:
    base = path.rsplit("/", 1)[-1]
    return base.startswith("test_") or base.endswith("_test.py") or "/tests/" in path


def _indent(text: str) -> int:
    return len(text) - len(text.lstrip())


def _rule_sql_injection(file: str, lines: list[ParsedLine]) -> list[Finding]:
    """SQL built by interpolating / concatenating values into the query."""
    out: list[Finding] = []
    for lineno, marker, text in lines:
        if marker != "+" or lineno is None:
            continue
        upper = text.upper()
        if not any(kw in upper for kw in _SQL_KEYWORDS):
            continue
        interpolated = bool(re.search(r'f["\']', text)) and "{" in text
        percent_fmt = bool(re.search(r'%\s*\(?\s*[A-Za-z_]', text)) and "%" in text
        concat = bool(re.search(r'["\']\s*\+\s*[A-Za-z_]', text))
        if interpolated or percent_fmt or concat:
            how = "f-string interpolation" if interpolated else (
                "%-formatting" if percent_fmt else "string concatenation"
            )
            out.append(_finding(
                file, lineno, lineno, Severity.CRITICAL, "security",
                "SQL injection via string-built query",
                f"User-controlled values are spliced into a SQL string with {how}. "
                "An attacker can break out of the query and read or modify data.",
                "Use a parameterised query instead, e.g.\n\n"
                'cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))',
                "sql-injection",
            ))
    return out


def _rule_hardcoded_secret(file: str, lines: list[ParsedLine]) -> list[Finding]:
    out: list[Finding] = []
    pat = re.compile(
        r"(?i)\b(password|passwd|secret|api[_-]?key|access[_-]?token|private[_-]?key)"
        r"\s*[:=]\s*[\"'][^\"'\n]{4,}[\"']"
    )
    url_pat = re.compile(r"https?://[^/\s:]+:[^/\s@]+@")
    for lineno, marker, text in lines:
        if marker != "+" or lineno is None:
            continue
        if pat.search(text):
            out.append(_finding(
                file, lineno, lineno, Severity.CRITICAL, "security",
                "Hardcoded credential in source",
                "A secret is written directly in the code. It will be committed "
                "to version control and is hard to rotate.",
                "Load it from the environment, e.g.\n\n"
                'API_KEY = os.environ["API_KEY"]',
                "hardcoded-secret",
            ))
        elif url_pat.search(text):
            out.append(_finding(
                file, lineno, lineno, Severity.CRITICAL, "security",
                "Credentials embedded in a URL",
                "A username and password are embedded in a URL literal.",
                "Use environment variables or a secrets manager for credentials.",
                "hardcoded-secret",
            ))
    return out


def _rule_command_injection(file: str, lines: list[ParsedLine]) -> list[Finding]:
    out: list[Finding] = []
    checks = [
        (re.compile(r"\beval\s*\("), "eval() executes arbitrary code"),
        (re.compile(r"\bexec\s*\("), "exec() executes arbitrary code"),
        (re.compile(r"os\.system\s*\("), "os.system() hands the string to a shell"),
        (re.compile(r"os\.popen\s*\("), "os.popen() hands the string to a shell"),
        (re.compile(r"shell\s*=\s*True"), "shell=True hands arguments to a shell"),
    ]
    for lineno, marker, text in lines:
        if marker != "+" or lineno is None:
            continue
        for pat, why in checks:
            if pat.search(text):
                out.append(_finding(
                    file, lineno, lineno, Severity.CRITICAL, "security",
                    "Command injection risk",
                    f"{why}. If any part of the input comes from a user or an "
                    "external source, it can run arbitrary commands.",
                    "Avoid the shell: pass arguments as a list to subprocess.run, "
                    "e.g.\n\nsubprocess.run([\"ls\", path], check=True)",
                    "command-injection",
                ))
                break
    return out


def _rule_unsafe_deserialization(file: str, lines: list[ParsedLine]) -> list[Finding]:
    out: list[Finding] = []
    for lineno, marker, text in lines:
        if marker != "+" or lineno is None:
            continue
        if re.search(r"pickle\.(loads?|Unpickler)\s*\(", text):
            out.append(_finding(
                file, lineno, lineno, Severity.CRITICAL, "security",
                "Unsafe deserialization with pickle",
                "Unpickling untrusted data can execute arbitrary code during "
                "deserialization.",
                "Use a safe format such as JSON for untrusted data.",
                "unsafe-deserialization",
            ))
        elif re.search(r"yaml\.load\s*\([^)]*\)", text) and "Loader" not in text:
            out.append(_finding(
                file, lineno, lineno, Severity.MAJOR, "security",
                "yaml.load without an explicit Loader",
                "yaml.load() without a Loader defaults to the unsafe loader, "
                "which can construct arbitrary Python objects.",
                "Use yaml.safe_load(...) instead.",
                "unsafe-deserialization",
            ))
    return out


def _rule_insecure_request(file: str, lines: list[ParsedLine]) -> list[Finding]:
    out: list[Finding] = []
    for lineno, marker, text in lines:
        if marker != "+" or lineno is None:
            continue
        if re.search(r"verify\s*=\s*False", text):
            out.append(_finding(
                file, lineno, lineno, Severity.MAJOR, "security",
                "TLS verification disabled",
                "verify=False turns off certificate checks, exposing the "
                "connection to man-in-the-middle attacks.",
                "Remove verify=False, or pin the expected certificate.",
                "insecure-request",
            ))
    return out


def _rule_weak_crypto(file: str, lines: list[ParsedLine]) -> list[Finding]:
    out: list[Finding] = []
    for lineno, marker, text in lines:
        if marker != "+" or lineno is None:
            continue
        if re.search(r"hashlib\.(md5|sha1)\s*\(", text):
            out.append(_finding(
                file, lineno, lineno, Severity.MAJOR, "security",
                "Weak hash function (MD5/SHA-1)",
                "MD5 and SHA-1 are broken for security uses such as password "
                "hashing or signatures.",
                "Use hashlib.sha256, or bcrypt/argon2 for passwords.",
                "weak-crypto",
            ))
        elif re.search(r"\brandom\.(random|choice|randint|randrange)\s*\(", text) and re.search(
            r"(?i)(token|password|secret|otp|key)", text
        ):
            out.append(_finding(
                file, lineno, lineno, Severity.MAJOR, "security",
                "Insecure randomness for a security value",
                "The `random` module is not cryptographically secure; tokens "
                "or secrets generated with it are predictable.",
                "Use the secrets module, e.g. secrets.token_urlsafe(32).",
                "insecure-random",
            ))
    return out


def _rule_n_plus_one(file: str, lines: list[ParsedLine]) -> list[Finding]:
    """A DB call inside a loop body — one query per item (N+1)."""
    out: list[Finding] = []
    for i, (lineno, marker, text) in enumerate(lines):
        if marker != "+" or lineno is None:
            continue
        if not re.match(r"\s*for\s+\w+", text):
            continue
        loop_indent = _indent(text)
        for j in range(i + 1, len(lines)):
            l2, m2, t2 = lines[j]
            if t2.strip() and _indent(t2) <= loop_indent:
                break  # loop body ended (any line dedented to the loop level)
            if m2 == "+" and _indent(t2) > loop_indent:
                if any(call in t2 for call in _DB_CALLS):
                    out.append(_finding(
                        file, l2 or lineno, l2 or lineno, Severity.MAJOR, "performance",
                        "N+1 query: database call inside a loop",
                        "This runs one query per loop iteration. For N items that "
                        "is N round-trips instead of one.",
                        "Fetch the related rows in a single query (JOIN / IN (...)) "
                        "and build a lookup dict before the loop.",
                        "n-plus-one",
                    ))
                    break
    return out


def _rule_mutable_default(file: str, lines: list[ParsedLine]) -> list[Finding]:
    out: list[Finding] = []
    pat = re.compile(r"def\s+\w+\s*\([^)]*=\s*(\[\]|\{\}|set\(\))")
    for lineno, marker, text in lines:
        if marker != "+" or lineno is None:
            continue
        if pat.search(text):
            out.append(_finding(
                file, lineno, lineno, Severity.MAJOR, "bug",
                "Mutable default argument",
                "A list/dict/set used as a default argument is created once "
                "and shared across calls, so items appended in one call leak "
                "into the next.",
                "Use None as the default and create the collection inside:\n\n"
                "def f(items=None):\n    items = [] if items is None else items",
                "mutable-default",
            ))
    return out


def _rule_bare_except(file: str, lines: list[ParsedLine]) -> list[Finding]:
    out: list[Finding] = []
    for lineno, marker, text in lines:
        if marker != "+" or lineno is None:
            continue
        if re.match(r"\s*except\s*:", text):
            out.append(_finding(
                file, lineno, lineno, Severity.MAJOR, "bug",
                "Bare except swallows every error",
                "A bare `except:` also catches KeyboardInterrupt and "
                "SystemExit, and hides the real failure from logs and tests.",
                "Catch the specific exception, e.g. `except ValueError:`, "
                "and log it.",
                "bare-except",
            ))
        elif re.match(r"\s*except\s+Exception\s*:", text):
            out.append(_finding(
                file, lineno, lineno, Severity.MINOR, "bug",
                "Overly broad except Exception",
                "`except Exception:` hides unrelated failures and makes "
                "debugging harder.",
                "Catch the specific exception types you expect, or re-raise "
                "after logging.",
                "broad-except",
            ))
    return out


def _rule_style(file: str, lines: list[ParsedLine]) -> list[Finding]:
    out: list[Finding] = []
    test_file = _is_test_file(file)
    for lineno, marker, text in lines:
        if marker != "+" or lineno is None:
            continue
        if re.search(r"==\s*None|!=\s*None", text):
            out.append(_finding(
                file, lineno, lineno, Severity.NIT, "style",
                "Use `is None` instead of `== None`",
                "`== None` can be fooled by objects that override __eq__; "
                "`is None` is the idiomatic check.",
                "Replace `== None` with `is None` (and `!= None` with "
                "`is not None`).",
                "none-comparison",
            ))
        elif re.search(r"(?i)\b(TODO|FIXME|XXX)\b", text):
            out.append(_finding(
                file, lineno, lineno, Severity.NIT, "style",
                "TODO/FIXME left in new code",
                "A leftover TODO in a new change usually means unfinished "
                "work landing in the PR.",
                "Finish the item, or file a tracked issue and reference it.",
                "todo-left",
            ))
        elif not test_file and re.search(r"(?<!\w)print\s*\(", text):
            out.append(_finding(
                file, lineno, lineno, Severity.NIT, "style",
                "print() in production code",
                "print() output is unstructured and hard to control in "
                "production; use the project's logger.",
                "Replace with logging, e.g. logger.info(...).",
                "print-call",
            ))
        elif not test_file and re.match(r"\s*assert\s+", text):
            out.append(_finding(
                file, lineno, lineno, Severity.MINOR, "bug",
                "assert in production code",
                "assert statements are removed when Python runs with -O, so "
                "they cannot be relied on for validation.",
                "Raise a proper exception for invalid input instead.",
                "assert-non-test",
            ))
    return out


def _rule_long_function(file: str, lines: list[ParsedLine]) -> list[Finding]:
    """An added function whose added body exceeds 50 lines."""
    out: list[Finding] = []
    added = [(ln, text) for ln, marker, text in lines if marker == "+"]
    i = 0
    while i < len(added):
        lineno, text = added[i]
        m = re.match(r"(\s*)(?:async\s+)?def\s+\w+\s*\(", text)
        if not m:
            i += 1
            continue
        indent = len(m.group(1))
        body = 0
        j = i + 1
        while j < len(added):
            l2, t2 = added[j]
            if t2.strip():
                if _indent(t2) <= indent:
                    break
                body += 1
            j += 1
        if body > 50:
            out.append(_finding(
                file, lineno, lineno, Severity.MINOR, "maintainability",
                "Very long function",
                f"This function adds {body} lines in one block, which makes "
                "it hard to read, test and review.",
                "Split it into smaller helpers, one per responsibility.",
                "long-function",
            ))
        i = j
    return out


_RULES = [
    _rule_sql_injection,
    _rule_hardcoded_secret,
    _rule_command_injection,
    _rule_unsafe_deserialization,
    _rule_insecure_request,
    _rule_weak_crypto,
    _rule_n_plus_one,
    _rule_mutable_default,
    _rule_bare_except,
    _rule_style,
    _rule_long_function,
]


def run_rules(file: str, lines: list[ParsedLine]) -> list[Finding]:
    """Run every deterministic rule over one hunk's lines."""
    findings: list[Finding] = []
    for rule in _RULES:
        findings.extend(rule(file, lines))
    return findings
