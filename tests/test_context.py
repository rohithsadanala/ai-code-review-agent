"""Tests for the repo-context (RAG) index."""

from __future__ import annotations

from pathlib import Path

from cra.context import RepoIndex


def _write_repo(tmp_path: Path) -> str:
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "billing.py").write_text(
        "def compute_discount(plan, coupon_code):\n"
        "    if coupon_code is None:\n"
        "        return 0.0\n"
        "    return min(coupon.percent_off, plan.max_discount)\n"
    )
    (tmp_path / "app" / "users.py").write_text(
        "def get_user(user_id):\n"
        "    return db.session.get(User, user_id)\n"
    )
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_billing.py").write_text(
        "from app.billing import compute_discount\n"
        "\n"
        "\n"
        "def test_discount_none_coupon():\n"
        "    assert compute_discount(plan, None) == 0.0\n"
    )
    # Skipped content: binary, venv, .git
    (tmp_path / "app" / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("[core]\n")
    return str(tmp_path)


def test_index_skips_binary_and_git(tmp_path):
    repo = _write_repo(tmp_path)
    index = RepoIndex.build(repo)
    paths = {c.path for c in index.chunks}
    assert "app/image.png" not in paths
    assert ".git/config" not in paths
    assert "app/billing.py" in paths
    assert "tests/test_billing.py" in paths


def test_retrieve_finds_relevant_chunk(tmp_path):
    repo = _write_repo(tmp_path)
    index = RepoIndex.build(repo)
    results = index.retrieve("compute_discount coupon plan", k=2)
    assert results
    assert any("compute_discount" in c.text for c in results)
    assert all(c.path for c in results)


def test_find_usages(tmp_path):
    repo = _write_repo(tmp_path)
    index = RepoIndex.build(repo)
    results = index.find_usages("compute_discount", k=3)
    assert any("test_billing.py" in c.path for c in results)


def test_empty_repo(tmp_path):
    index = RepoIndex.build(str(tmp_path))
    assert index.retrieve("anything") == []
