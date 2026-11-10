"""Tests for the unified diff parser."""

from __future__ import annotations

from cra.diff_parser import (
    added_lines_by_file,
    hunk_added_text,
    parse_diff,
)

SIMPLE = """diff --git a/app.py b/app.py
--- a/app.py
+++ b/app.py
@@ -1,3 +1,5 @@
 def hello():
-    print("hi")
+    logger.info("hi")
+    return True
+
 
 def bye():
"""

NEW_FILE = """diff --git a/new.py b/new.py
new file mode 100644
--- /dev/null
+++ b/new.py
@@ -0,0 +1,3 @@
+def f():
+    return 1
+
"""

MULTI = SIMPLE + NEW_FILE.replace("new.py", "other.py").replace(
    "def f():", "def g():"
)


def test_parse_simple_diff():
    model, files = parse_diff(SIMPLE)
    assert len(files) == 1
    f = files[0]
    assert f.path == "app.py"
    assert not f.is_new
    assert len(f.hunks) == 1
    h = f.hunks[0]
    assert (h.old_start, h.old_count) == (1, 3)
    assert (h.new_start, h.new_count) == (1, 5)
    added = h.added()
    assert [ln.text for ln in added] == ['    logger.info("hi")', "    return True", ""]
    assert [ln.new_lineno for ln in added] == [2, 3, 4]


def test_parse_new_file():
    model, files = parse_diff(NEW_FILE)
    assert len(files) == 1
    assert files[0].is_new
    assert files[0].path == "new.py"
    assert len(files[0].hunks[0].added()) == 3


def test_parse_multi_file():
    model, files = parse_diff(MULTI)
    assert [f.path for f in files] == ["app.py", "other.py"]


def test_added_lines_by_file():
    _, files = parse_diff(SIMPLE)
    added = added_lines_by_file(files)
    assert added["app.py"][0] == (2, '    logger.info("hi")')


def test_hunk_added_text_has_line_numbers():
    _, files = parse_diff(SIMPLE)
    text = hunk_added_text(files[0].hunks[0])
    assert text.splitlines()[0].startswith("L2: ")


def test_deleted_file_and_binary():
    diff = """diff --git a/old.py b/old.py
deleted file mode 100644
--- a/old.py
+++ /dev/null
@@ -1,2 +0,0 @@
-def old():
-    pass
diff --git a/img.png b/img.png
Binary files a/img.png and b/img.png differ
"""
    _, files = parse_diff(diff)
    assert files[0].is_deleted
    assert files[0].path == "old.py"
    assert files[1].is_binary


def test_removed_line_starting_with_dashes_is_code_not_header():
    # A removed line whose content starts with "--" must not be
    # mistaken for a --- file header.
    diff = """diff --git a/a.py b/a.py
--- a/a.py
+++ b/a.py
@@ -1,3 +1,3 @@
 def f():
--- old comment
+-- new comment
     return 1
"""
    _, files = parse_diff(diff)
    h = files[0].hunks[0]
    kinds = [ln.kind for ln in h.lines]
    assert kinds == ["context", "del", "add", "context"]
    assert h.lines[1].text == "-- old comment"
    assert h.lines[2].text == "-- new comment"
