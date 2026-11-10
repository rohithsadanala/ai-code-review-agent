"""Tests for the FastAPI backend."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from cra.api import app

from conftest import SAMPLE_DIFF


@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("CRA_DATA_DIR", str(tmp_path / "data"))
    # Re-import so the store points at the temp dir.
    import importlib

    import cra.api as api_mod

    importlib.reload(api_mod)
    return TestClient(api_mod.app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_create_and_get_review(client):
    resp = client.post("/reviews", json={"diff": SAMPLE_DIFF})
    assert resp.status_code == 201
    body = resp.json()
    assert body["verdict"] == "request_changes"
    assert body["findings"]
    assert body["markdown"]

    got = client.get(f"/reviews/{body['id']}")
    assert got.status_code == 200
    assert got.json()["id"] == body["id"]


def test_list_reviews(client):
    client.post("/reviews", json={"diff": SAMPLE_DIFF})
    resp = client.get("/reviews")
    assert resp.status_code == 200
    assert len(resp.json()) >= 1
    assert resp.json()[0]["critical"] >= 1


def test_empty_diff_rejected(client):
    resp = client.post("/reviews", json={"diff": "   "})
    assert resp.status_code == 400


def test_unknown_review_404(client):
    resp = client.get("/reviews/does-not-exist")
    assert resp.status_code == 404


def test_post_to_github_without_token(client):
    resp = client.post("/reviews", json={"diff": SAMPLE_DIFF})
    review_id = resp.json()["id"]
    resp = client.post(
        f"/reviews/{review_id}/post-to-github",
        json={"owner": "octocat", "repo": "hello-world", "number": 1},
    )
    # No GITHUB_TOKEN in the test env, so this must fail cleanly.
    assert resp.status_code == 400
    assert "GITHUB_TOKEN" in resp.json()["detail"]
