"""RAG context over the repository's own codebase.

This is what makes the reviewer an *agent* rather than a one-shot
prompt: before reviewing a hunk it looks at the surrounding code to
answer questions a bare diff cannot — is this function used elsewhere,
does the codebase already have a pattern for this, does the change
break an existing convention?

Indexing
--------
Text files under ``repo_path`` are chunked into overlapping line
windows (default 40 lines, 10 overlap) with ``(path, start, end)``
metadata. Binary files, ``.git``, ``__pycache__``, ``.venv``, ``node_modules``
and other generated/vendor directories are skipped.

Retrieval
---------
Hybrid search: BM25 (lexical) + FAISS dense search over the same
chunks, fused with Reciprocal Rank Fusion — the same pattern as a
standard RAG stack. The fake (hash) embeddings keep this fully
offline; swap in Hugging Face / OpenAI embeddings via env vars.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np
from rank_bm25 import BM25Okapi

from .embeddings import get_embeddings

_SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    "dist",
    "build",
    ".eggs",
    "site-packages",
}
_SKIP_EXTS = {
    ".pyc",
    ".pyo",
    ".so",
    ".dll",
    ".dylib",
    ".exe",
    ".bin",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".ico",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".whl",
    ".lock",
}
_MAX_FILE_BYTES = 200_000
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+")


@dataclass
class ContextChunk:
    path: str
    start_line: int
    end_line: int
    text: str
    score: float = 0.0


@dataclass
class RepoIndex:
    """In-memory hybrid index over one repository snapshot."""

    chunks: list[ContextChunk] = field(default_factory=list)
    _bm25: BM25Okapi | None = None
    _faiss_index: object | None = None
    _embedder: object = None
    _dim: int = 384

    # -- building ------------------------------------------------------------
    @classmethod
    def build(
        cls,
        repo_path: str,
        chunk_lines: int = 40,
        overlap: int = 10,
        embedding_provider: str = "fake",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        api_key: str = "",
    ) -> "RepoIndex":
        index = cls(_dim=embedding_dim)
        index._embedder = get_embeddings(
            embedding_provider, embedding_model, embedding_dim, api_key
        )
        texts: list[str] = []
        for path, start, end, text in _iter_chunks(repo_path, chunk_lines, overlap):
            index.chunks.append(
                ContextChunk(path=path, start_line=start, end_line=end, text=text)
            )
            texts.append(text)
        if not index.chunks:
            return index
        # BM25 (lexical)
        tokenized = [_TOKEN_RE.findall(t.lower()) for t in texts]
        index._bm25 = BM25Okapi(tokenized)
        # FAISS (dense, inner product over L2-normalised vectors)
        import faiss

        vectors = np.asarray(
            index._embedder.embed(texts), dtype=np.float32
        )
        faiss_index = faiss.IndexFlatIP(vectors.shape[1])
        faiss_index.add(vectors)
        index._faiss_index = faiss_index
        return index

    # -- retrieval -------------------------------------------------------------
    def retrieve(self, query: str, k: int = 3) -> list[ContextChunk]:
        """Hybrid BM25 + FAISS retrieval, fused with Reciprocal Rank Fusion."""
        if not self.chunks:
            return []
        n = len(self.chunks)
        take = max(k * 3, 5)

        # BM25 ranking (best first).
        bm25_scores = self._bm25.get_scores(_TOKEN_RE.findall(query.lower()))
        bm25_rank = list(np.argsort(-np.asarray(bm25_scores))[:take])

        # FAISS ranking (best first).
        faiss_rank: list[int] = []
        if self._faiss_index is not None:
            q = np.asarray(self._embedder.embed([query]), dtype=np.float32)
            _, idx = self._faiss_index.search(q, min(take, n))
            faiss_rank = [int(i) for i in idx[0] if i != -1]

        # RRF fusion: score = sum(1 / (60 + rank)) over both rankings.
        fused: dict[int, float] = {}
        for rank, idx in enumerate(bm25_rank):
            fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (60 + rank + 1)
        for rank, idx in enumerate(faiss_rank):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (60 + rank + 1)

        top = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        results: list[ContextChunk] = []
        for idx, score in top:
            chunk = self.chunks[idx]
            results.append(
                ContextChunk(
                    path=chunk.path,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    text=chunk.text,
                    score=score,
                )
            )
        return results

    def find_usages(self, symbol: str, k: int = 3) -> list[ContextChunk]:
        """Chunks that mention a symbol — used to check whether a changed
        function is called elsewhere in the codebase."""
        return self.retrieve(symbol, k=k)


def _iter_chunks(repo_path: str, chunk_lines: int, overlap: int):
    """Yield ``(rel_path, start_line, end_line, text)`` windows."""
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS)
        for name in sorted(files):
            if os.path.splitext(name)[1].lower() in _SKIP_EXTS:
                continue
            full = os.path.join(root, name)
            try:
                if os.path.getsize(full) > _MAX_FILE_BYTES:
                    continue
                with open(full, encoding="utf-8", errors="ignore") as fh:
                    content = fh.read()
            except OSError:
                continue
            if "\x00" in content:
                continue  # binary content slipped past the extension check
            rel = os.path.relpath(full, repo_path)
            lines = content.splitlines()
            if not lines:
                continue
            step = max(1, chunk_lines - overlap)
            start = 0
            while start < len(lines):
                end = min(start + chunk_lines, len(lines))
                yield rel, start + 1, end, "\n".join(lines[start:end])
                if end >= len(lines):
                    break
                start += step


def format_context(chunks: list[ContextChunk], max_chars: int = 3000) -> str:
    """Render retrieved chunks for the reviewer prompt."""
    if not chunks:
        return "(no repository context available)"
    parts: list[str] = []
    used = 0
    for c in chunks:
        block = f"--- {c.path}:{c.start_line}-{c.end_line} ---\n{c.text}"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts) if parts else "(no repository context available)"
