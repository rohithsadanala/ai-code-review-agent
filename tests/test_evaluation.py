"""Tests for the eval harness."""

from __future__ import annotations

import json
import os

from cra.evaluation import run_eval, score_case
from cra.models import Finding, Severity

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_SET = os.path.join(HERE, "..", "eval_data", "eval_set.json")


def test_eval_set_scores_high(settings):
    """The shipped eval set is tuned for the mock LLM — it should
    score near-perfect precision/recall with no false positives."""
    results = run_eval(EVAL_SET, settings=settings)
    assert results["cases"] == 4
    assert results["mean_precision"] >= 0.9
    assert results["mean_recall"] >= 0.9
    assert results["mean_f1"] >= 0.9
    assert results["mean_citation_accuracy"] == 1.0
    assert results["mean_false_positive_rate"] <= 0.1
    for case in results["per_case"]:
        assert case["missed"] == [], f"{case['name']} missed: {case['missed']}"


def test_score_case_partial_credit():
    diff = """diff --git a/a.py b/a.py
--- a/a.py
+++ b/a.py
@@ -1,2 +1,4 @@
 def f():
+    x = 1
+    y = 2
"""
    reported = [
        Finding(file="a.py", line=2, end_line=2, severity=Severity.MINOR,
                category="bug", title="t", message="m"),
        Finding(file="a.py", line=3, end_line=3, severity=Severity.NIT,
                category="style", title="t", message="m"),
    ]
    expected = [{"file": "a.py", "line": 2, "category": "bug"}]
    precision, recall, f1, fpr, citation_acc, missed, extra = score_case(
        reported, expected, diff
    )
    assert precision == 0.5
    assert recall == 1.0
    assert missed == []
    assert len(extra) == 1
    assert citation_acc == 1.0


def test_score_case_empty_expected_and_reported():
    precision, recall, f1, fpr, citation_acc, missed, extra = score_case(
        [], [], "diff --git a/a.py b/a.py\n"
    )
    assert precision == 1.0
    assert recall == 1.0
    assert f1 == 1.0
    assert citation_acc == 1.0
