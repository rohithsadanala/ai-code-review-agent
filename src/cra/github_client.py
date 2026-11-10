"""GitHub API client (httpx, no extra dependency).

Two operations, both token-authenticated with ``GITHUB_TOKEN`` from
the environment — the token is never logged or stored in code:

* ``fetch_pr_diff`` — download a pull request as a unified diff
  (``Accept: application/vnd.github.diff``).
* ``post_review`` — submit a pull-request review: the verdict becomes
  the review event (APPROVE / REQUEST_CHANGES / COMMENT) and every
  finding with a valid new-side line becomes an inline comment
  (``path`` + ``line`` + ``side: RIGHT``). Findings that cannot be
  attached to a line go into the review body instead.
"""

from __future__ import annotations

import httpx

from .models import ReviewReport
from .report import render_github_comment_body

_API = "https://api.github.com"
_DIFF_ACCEPT = "application/vnd.github.diff"


class GitHubError(RuntimeError):
    """Raised when the GitHub API returns an error."""


def _headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "ai-code-review-agent",
    }


def fetch_pr_diff(owner: str, repo: str, number: int, token: str) -> str:
    """Download a PR as a unified diff."""
    url = f"{_API}/repos/{owner}/{repo}/pulls/{number}"
    headers = _headers(token)
    headers["Accept"] = _DIFF_ACCEPT
    with httpx.Client(timeout=30) as client:
        resp = client.get(url, headers=headers)
    if resp.status_code != 200:
        raise GitHubError(
            f"Fetching PR {owner}/{repo}#{number} failed: "
            f"{resp.status_code} {resp.text[:200]}"
        )
    return resp.text


def post_review(
    owner: str, repo: str, number: int, token: str, report: ReviewReport
) -> dict:
    """Submit the review to the PR and return the API response JSON."""
    event = {
        "approve": "APPROVE",
        "request_changes": "REQUEST_CHANGES",
        "comment": "COMMENT",
    }[report.verdict]

    comments: list[dict] = []
    body_extras: list[str] = []
    for f in report.findings:
        comment = {
            "path": f.file,
            "line": f.end_line,
            "side": "RIGHT",
            "body": render_github_comment_body(f),
        }
        comments.append(comment)

    body = f"## AI Code Review — {report.verdict.replace('_', ' ').title()}\n\n{report.summary}"
    if body_extras:
        body += "\n\n" + "\n\n".join(body_extras)

    payload: dict = {"event": event, "body": body}
    if comments:
        payload["comments"] = comments

    url = f"{_API}/repos/{owner}/{repo}/pulls/{number}/reviews"
    with httpx.Client(timeout=30) as client:
        resp = client.post(url, headers=_headers(token), json=payload)
    if resp.status_code not in (200, 201):
        raise GitHubError(
            f"Posting review to {owner}/{repo}#{number} failed: "
            f"{resp.status_code} {resp.text[:300]}"
        )
    return resp.json()


def parse_pr_ref(pr_ref: str) -> tuple[str, str, int]:
    """Parse ``owner/repo#number`` into its parts."""
    if "#" not in pr_ref or "/" not in pr_ref:
        raise ValueError(
            f"Expected 'owner/repo#number', got {pr_ref!r}. "
            "Example: octocat/hello-world#42"
        )
    repo_part, number_part = pr_ref.rsplit("#", 1)
    owner, repo = repo_part.split("/", 1)
    return owner, repo, int(number_part)
