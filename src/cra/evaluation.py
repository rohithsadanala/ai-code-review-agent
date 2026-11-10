"""Eval harness for the review agent.

Each eval case is a small diff with a list of expected findings
(``file``, ``line``, ``category``, ``severity``). The harness runs
the full pipeline with the configured LLM provider and scores:

* **precision** — fraction of reported findings that match an
  expected finding (same file, line within ±1, same category).
* **recall** — fraction of expected findings that were reported.
* **F1** — harmonic mean of the two.
* **false-positive rate** — reported findings with no expected match,
  as a fraction of all reported findings.
* **citation accuracy** — fraction of reported findings whose cited
  file + line actually exists in the diff (a finding that cites a
  line the diff never touched is a hallucinated citation).

Results are written to ``eval_data/eval_results.json`` and printed
as a table. The shipped eval set is tuned for the mock LLM: the
planted issues are all patterns the deterministic rule engine
catches, so a healthy run scores precision/recall near 1.0.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from .config import Settings
from .diff_parser import parse_diff
from .models import Finding
from .pipeline import run_review


@dataclass
class CaseResult:
    name: str
    expected: int
    reported: int
    matched: int
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    citation_accuracy: float
    missed: list[dict]
    extra: list[dict]


def _matches(reported: Finding, expected: dict) -> bool:
    """Same file, line within ±1, same category."""
    return (
        reported.file == expected["file"]
        and abs(reported.line - expected["line"]) <= 1
        and reported.category == expected["category"]
    )


def score_case(
    reported: list[Finding], expected: list[dict], diff_text: str
) -> tuple[float, float, float, float, float, list[dict], list[dict]]:
    matched_expected: set[int] = set()
    matched_reported: set[int] = set()
    for i, exp in enumerate(expected):
        for j, rep in enumerate(reported):
            if j in matched_reported:
                continue
            if _matches(rep, exp):
                matched_expected.add(i)
                matched_reported.add(j)
                break

    matched = len(matched_expected)
    precision = matched / len(reported) if reported else (1.0 if not expected else 0.0)
    recall = matched / len(expected) if expected else 1.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    fpr = 1.0 - precision if reported else 0.0

    # Citation accuracy: cited file+line must exist in the diff.
    _, parsed = parse_diff(diff_text)
    valid: set[tuple[str, int]] = set()
    for f in parsed:
        for h in f.hunks:
            for ln in h.lines:
                if ln.new_lineno is not None:
                    valid.add((f.path, ln.new_lineno))
    cited_ok = sum(
        1 for rep in reported if (rep.file, rep.line) in valid
    )
    citation_accuracy = cited_ok / len(reported) if reported else 1.0

    missed = [e for i, e in enumerate(expected) if i not in matched_expected]
    extra = [
        {
            "file": r.file,
            "line": r.line,
            "category": r.category,
            "severity": r.severity.value,
            "title": r.title,
        }
        for j, r in enumerate(reported)
        if j not in matched_reported
    ]
    return precision, recall, f1, fpr, citation_accuracy, missed, extra


def run_eval(eval_path: str, settings: Settings | None = None) -> dict:
    """Run every case in an eval-set JSON file; return the results dict."""
    settings = settings or Settings()
    with open(eval_path, encoding="utf-8") as fh:
        data = json.load(fh)
    results: list[CaseResult] = []
    for case in data["cases"]:
        report = run_review(case["diff"], settings=settings)
        precision, recall, f1, fpr, citation_acc, missed, extra = score_case(
            report.findings, case["expected"], case["diff"]
        )
        results.append(
            CaseResult(
                name=case["name"],
                expected=len(case["expected"]),
                reported=len(report.findings),
                matched=len(case["expected"]) - len(missed),
                precision=round(precision, 3),
                recall=round(recall, 3),
                f1=round(f1, 3),
                false_positive_rate=round(fpr, 3),
                citation_accuracy=round(citation_acc, 3),
                missed=missed,
                extra=extra,
            )
        )

    n = len(results)
    aggregate = {
        "cases": n,
        "mean_precision": round(sum(r.precision for r in results) / n, 3) if n else 0.0,
        "mean_recall": round(sum(r.recall for r in results) / n, 3) if n else 0.0,
        "mean_f1": round(sum(r.f1 for r in results) / n, 3) if n else 0.0,
        "mean_false_positive_rate": round(
            sum(r.false_positive_rate for r in results) / n, 3
        ) if n else 0.0,
        "mean_citation_accuracy": round(
            sum(r.citation_accuracy for r in results) / n, 3
        ) if n else 0.0,
        "per_case": [
            {
                "name": r.name,
                "expected": r.expected,
                "reported": r.reported,
                "matched": r.matched,
                "precision": r.precision,
                "recall": r.recall,
                "f1": r.f1,
                "false_positive_rate": r.false_positive_rate,
                "citation_accuracy": r.citation_accuracy,
                "missed": r.missed,
                "extra": r.extra,
            }
            for r in results
        ],
    }
    return aggregate


def save_results(results: dict, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
