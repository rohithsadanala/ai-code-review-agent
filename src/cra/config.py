"""Configuration for the AI Code Review Agent.

All settings come from environment variables (or a ``.env`` file loaded
by :mod:`cra.cli` / :mod:`cra.api`), so no secret ever lives in code.
See ``.env.example`` for the full list.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _getenv(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _getint(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _getbool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    """Runtime configuration. Defaults run fully offline (fake embeddings
    + mock LLM) with no API key and no GitHub token."""

    # -- data ------------------------------------------------------------------
    data_dir: str = field(default_factory=lambda: _getenv("CRA_DATA_DIR", "./data"))
    """Directory for persisted review reports (JSON)."""

    # -- llm -------------------------------------------------------------------
    llm_provider: str = field(default_factory=lambda: _getenv("CRA_LLM_PROVIDER", "mock"))
    """One of ``mock`` (deterministic, offline), ``openai`` or ``huggingface``."""
    llm_model: str = field(
        default_factory=lambda: _getenv("CRA_LLM_MODEL", "gpt-4o-mini")
    )
    """Model name for the OpenAI / Hugging Face provider."""
    openai_api_key: str = field(default_factory=lambda: _getenv("OPENAI_API_KEY", ""))
    """OpenAI API key. Read from the environment only; never hardcoded."""

    # -- embeddings -------------------------------------------------------------
    embedding_provider: str = field(
        default_factory=lambda: _getenv("CRA_EMBEDDING_PROVIDER", "fake")
    )
    """One of ``fake`` (deterministic, offline), ``huggingface`` or ``openai``."""
    embedding_model: str = field(
        default_factory=lambda: _getenv(
            "CRA_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    embedding_dim: int = field(default_factory=lambda: _getint("CRA_EMBEDDING_DIM", 384))
    """Vector size. 384 matches all-MiniLM-L6-v2 and the fake provider."""

    # -- review -----------------------------------------------------------------
    max_findings: int = field(default_factory=lambda: _getint("CRA_MAX_FINDINGS", 25))
    """Cap on findings kept in a report (after severity prioritisation)."""
    context_k: int = field(default_factory=lambda: _getint("CRA_CONTEXT_K", 3))
    """How many repo-context chunks to attach to each hunk review."""
    repo_path: str = field(default_factory=lambda: _getenv("CRA_REPO_PATH", ""))
    """Default local repo path used as RAG context (empty = no context)."""

    # -- github -----------------------------------------------------------------
    github_token: str = field(default_factory=lambda: _getenv("GITHUB_TOKEN", ""))
    """GitHub personal access token. Environment only; never hardcoded."""

    # -- server -----------------------------------------------------------------
    host: str = field(default_factory=lambda: _getenv("CRA_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _getint("CRA_PORT", 8000))
    reload: bool = field(default_factory=lambda: _getbool("CRA_RELOAD", False))
