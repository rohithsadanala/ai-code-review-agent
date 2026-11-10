"""Tests for the GitHub API client (HTTP mocked — no network)."""

from __future__ import annotations

import json

import httpx
import pytest

from cra.github_client import GitHubError, fetch_pr_diff, parse_pr_ref, post_review
from cra.models import Finding, ReviewReport, Severity


def test_parse_pr_ref():
    assert parse_pr_ref("octocat/hello-world#42") == ("octocat", "hello-world", 42)
    with pytest.raises(ValueError):
        parse_pr_ref("not-a-pr-ref")


def _mock_client(monkeypatch, handler):
    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)

    class _ClientCtx:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return client

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(httpx, "Client", _ClientCtx)


def test_fetch_pr_diff(monkeypatch):
    def handler(request):
        assert request.headers["Accept"] == "application/vnd.github.diff"
        assert request.url.path == "/repos/octocat/hello-world/pulls/42"
        return httpx.Response(200, text="diff --git a/x b/x\n")

    _mock_client(monkeypatch, handler)
    assert fetch_pr_diff("octocat", "hello-world", 42, "tok") == "diff --git a/x b/x\n"


def test_fetch_pr_diff_error(monkeypatch):
    def handler(request):
        return httpx.Response(404, text="Not Found")

    _mock_client(monkeypatch, handler)
    with pytest.raises(GitHubError):
        fetch_pr_diff("octocat", "hello-world", 42, "tok")


def test_post_review_payload(monkeypatch):
    seen: dict = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        assert request.url.path == "/repos/octocat/hello-world/pulls/42/reviews"
        return httpx.Response(201, json={"id": 7, "state": "CHANGES_REQUESTED",
                                        "html_url": "https://example.invalid/r/7"})

    _mock_client(monkeypatch, handler)
    report = ReviewReport(
        id="abc123",
        verdict="request_changes",
        summary="Found 1 issue(s): 1 critical.",
        findings=[
            Finding(file="app/auth.py", line=8, end_line=8,
                    severity=Severity.CRITICAL, category="security",
                    title="SQL injection", message="bad",
                    suggestion="use params"),
        ],
        files_reviewed=["app/auth.py"],
    )
    result = post_review("octocat", "hello-world", 42, "tok", report)
    assert result["id"] == 7
    assert seen["body"]["event"] == "REQUEST_CHANGES"
    assert len(seen["body"]["comments"]) == 1
    comment = seen["body"]["comments"][0]
    assert comment["path"] == "app/auth.py"
    assert comment["line"] == 8
    assert comment["side"] == "RIGHT"
    assert "SQL injection" in comment["body"]


def test_post_review_error(monkeypatch):
    def handler(request):
        return httpx.Response(403, text="Forbidden")

    _mock_client(monkeypatch, handler)
    report = ReviewReport(
        id="abc123", verdict="approve", summary="clean",
        files_reviewed=[],
    )
    with pytest.raises(GitHubError):
        post_review("octocat", "hello-world", 42, "tok", report)
