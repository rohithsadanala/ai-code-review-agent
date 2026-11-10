"""Tests for the review pipeline end-to-end (mock LLM, offline)."""

from __future__ import annotations

from cra.config import Settings
from cra.pipeline import run_review

from conftest import SAMPLE_DIFF

CLEAN_DIFF = """diff --git a/app/discounts.py b/app/discounts.py
new file mode 100644
--- /dev/null
+++ b/app/discounts.py
@@ -0,0 +1,6 @@
+def apply_discount(amount, discount):
+    if discount < 0.0 or discount > 1.0:
+        raise ValueError("discount must be between 0 and 1")
+    return round(amount * (1.0 - discount), 2)
+
+
"""

MULTI_FILE_DIFF = """diff --git a/app/a.py b/app/a.py
--- a/app/a.py
+++ b/app/a.py
@@ -1,2 +1,3 @@
 def a():
+    print("debug")
     return 1
diff --git a/app/b.py b/app/b.py
--- a/app/b.py
+++ b/app/b.py
@@ -1,2 +1,3 @@
 def b():
+    x = 1
     return 2
"""


def test_pipeline_flags_sql_injection(settings):
    report = run_review(SAMPLE_DIFF, settings=settings)
    assert report.verdict == "request_changes"
    assert any(
        f.category == "security" and f.severity.value == "critical"
        for f in report.findings
    )
    assert report.files_reviewed == ["app/auth.py"]
    # Every citation must point at a real line in the diff.
    assert all(f.line == 17 for f in report.findings if f.category == "security")
    assert "app/auth.py" in report.markdown


def test_pipeline_clean_diff_approves(settings):
    report = run_review(CLEAN_DIFF, settings=settings)
    assert report.findings == []
    assert report.verdict == "approve"
    assert "No issues found" in report.markdown


def test_pipeline_multi_file(settings):
    report = run_review(MULTI_FILE_DIFF, settings=settings)
    assert set(report.files_reviewed) == {"app/a.py", "app/b.py"}
    # Only app/a.py has an issue (print); app/b.py is clean.
    assert len(report.findings) == 1
    assert report.findings[0].file == "app/a.py"
    assert report.findings[0].category == "style"
    assert report.verdict == "approve"  # only a nit


def test_pipeline_dedupes_identical_findings(settings):
    # The same SQL pattern twice in one hunk, same line category but
    # different lines — both kept; a repeated identical line is deduped
    # by the dedupe step (covered implicitly by the single finding).
    report = run_review(SAMPLE_DIFF, settings=settings)
    keys = {(f.file, f.line, f.category) for f in report.findings}
    assert len(keys) == len(report.findings)


def test_pipeline_with_repo_context(tmp_path, settings):
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "auth.py").write_text(
        "def get_connection():\n    return connect(DATABASE_URL)\n"
    )
    report = run_review(SAMPLE_DIFF, settings=settings, repo_path=str(tmp_path))
    assert report.verdict == "request_changes"
    assert any(f.category == "security" for f in report.findings)


def test_pipeline_empty_diff(settings):
    report = run_review("", settings=settings)
    assert report.findings == []
    assert report.verdict == "approve"
    assert report.files_reviewed == []
