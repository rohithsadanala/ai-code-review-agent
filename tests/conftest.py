"""Shared fixtures for the AI Code Review Agent tests."""

from __future__ import annotations

import pytest

from cra.config import Settings


@pytest.fixture()
def settings() -> Settings:
    """Offline settings: mock LLM + fake embeddings, no keys."""
    return Settings(
        llm_provider="mock",
        embedding_provider="fake",
        data_dir="/tmp/cra-test-data",
    )


SAMPLE_DIFF = """diff --git a/app/auth.py b/app/auth.py
--- a/app/auth.py
+++ b/app/auth.py
@@ -12,6 +12,12 @@ def login(username):
     return user
 
 
+def get_user_by_email(email):
+    conn = get_connection()
+    query = f"SELECT * FROM users WHERE email = '{email}'"
+    cursor = conn.cursor()
+    cursor.execute(query)
+    return cursor.fetchone()
+
+
 def logout(session_id):
     sessions.pop(session_id, None)
 """
