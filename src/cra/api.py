"""FastAPI backend for the AI Code Review Agent.

Endpoints
---------
GET  /                 — the single-page UI.
GET  /health           — liveness check.
POST /reviews          — submit a diff (or a ``owner/repo#number`` PR
                          reference) and run the full review pipeline.
GET  /reviews          — list past review summaries (newest first).
GET  /reviews/{id}     — full report for one review.
POST /reviews/{id}/post-to-github — post the stored report as a PR
                          review (needs ``GITHUB_TOKEN`` in the env).
GET  /docs             — OpenAPI docs (built in).

Reviews are kept in memory and persisted as JSON under ``CRA_DATA_DIR``
so a restart does not lose them.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from .config import Settings
from .github_client import GitHubError, fetch_pr_diff, parse_pr_ref, post_review
from .models import GitHubPostRequest, ReviewReport, ReviewRequest, ReviewSummary
from .pipeline import run_review

load_dotenv()

settings = Settings()
app = FastAPI(
    title="AI Code Review Agent",
    description="Reviews pull-request diffs with an LLM agent: hybrid RAG "
    "context over the repo, LangGraph pipeline, cited findings.",
    version="0.1.0",
)
_DATA_DIR = Path(settings.data_dir)
_DATA_DIR.mkdir(parents=True, exist_ok=True)

# In-memory store, hydrated from disk at startup.
_store: dict[str, ReviewReport] = {}


def _store_path(report_id: str) -> Path:
    return _DATA_DIR / f"{report_id}.json"


def _persist(report: ReviewReport) -> None:
    try:
        _store_path(report.id).write_text(
            report.model_dump_json(indent=2), encoding="utf-8"
        )
    except OSError:
        pass  # persistence is best-effort; the in-memory copy still works


def _load_store() -> None:
    for path in _DATA_DIR.glob("*.json"):
        try:
            _store[path.stem] = ReviewReport.model_validate_json(path.read_text())
        except (OSError, ValueError):
            continue


_load_store()


def _summary(report: ReviewReport) -> ReviewSummary:
    return ReviewSummary(
        id=report.id,
        verdict=report.verdict,
        summary=report.summary,
        findings=len(report.findings),
        critical=len(report.by_severity("critical")),
        major=len(report.by_severity("major")),
        files_reviewed=report.files_reviewed,
        pr=report.pr,
    )


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "llm_provider": settings.llm_provider,
        "embedding_provider": settings.embedding_provider,
        "reviews_stored": len(_store),
        "time": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/", include_in_schema=False)
def index():
    frontend = Path(__file__).resolve().parent.parent.parent / "frontend" / "index.html"
    if frontend.exists():
        return FileResponse(frontend, media_type="text/html")
    return JSONResponse(
        {"message": "AI Code Review Agent API — see /docs for the API."}
    )


@app.post("/reviews", response_model=ReviewReport, status_code=201)
def create_review(request: ReviewRequest) -> ReviewReport:
    diff_text = request.diff
    pr = request.pr
    if pr:
        try:
            owner, repo, number = parse_pr_ref(pr)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not settings.github_token:
            raise HTTPException(
                status_code=400,
                detail="GITHUB_TOKEN is not set — cannot fetch a PR diff.",
            )
        try:
            diff_text = fetch_pr_diff(owner, repo, number, settings.github_token)
        except GitHubError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
    if not diff_text.strip():
        raise HTTPException(status_code=400, detail="Diff text is empty.")
    try:
        report = run_review(
            diff_text,
            settings=settings,
            repo_path=request.repo_path or "",
            pr=pr,
        )
    except Exception as exc:  # noqa: BLE001 — surface pipeline errors as 500s
        raise HTTPException(status_code=500, detail=f"Review failed: {exc}") from exc
    _store[report.id] = report
    _persist(report)
    return report


@app.get("/reviews", response_model=list[ReviewSummary])
def list_reviews() -> list[ReviewSummary]:
    return [_summary(r) for r in sorted(_store.values(), key=lambda r: r.id)]


@app.get("/reviews/{review_id}", response_model=ReviewReport)
def get_review(review_id: str) -> ReviewReport:
    report = _store.get(review_id)
    if report is None:
        raise HTTPException(status_code=404, detail=f"No review {review_id!r}.")
    return report


@app.post("/reviews/{review_id}/post-to-github")
def post_review_to_github(review_id: str, request: GitHubPostRequest) -> dict:
    report = _store.get(review_id)
    if report is None:
        raise HTTPException(status_code=404, detail=f"No review {review_id!r}.")
    if not settings.github_token:
        raise HTTPException(
            status_code=400,
            detail="GITHUB_TOKEN is not set — cannot post to GitHub.",
        )
    try:
        result = post_review(
            request.owner, request.repo, request.number, settings.github_token, report
        )
    except GitHubError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {
        "posted": True,
        "review_id": result.get("id"),
        "state": result.get("state"),
        "html_url": result.get("html_url"),
    }
