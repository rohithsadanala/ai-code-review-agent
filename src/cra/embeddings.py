"""Embedding providers.

All providers share one interface — ``embed(texts) -> list[list[float]]``
— so the context index never cares which model is behind it.

``fake``
    Deterministic, offline, no downloads: each text is hashed into a
    fixed-size vector (seeded per token). Good enough for tests, demos
    and CI, and it preserves the property that identical texts embed
    identically and similar texts (shared tokens) embed closer.

``huggingface``
    Sentence-Transformers (default ``all-MiniLM-L6-v2``, 384 dims).
    Needs ``requirements-optional.txt``.

``openai``
    ``text-embedding-3-small`` via LangChain. Needs ``OPENAI_API_KEY``.
"""

from __future__ import annotations

import hashlib
import math
import re

import numpy as np

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+")


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


class FakeEmbeddings:
    """Deterministic hash-based embeddings (offline, no model download).

    Each token contributes to a fixed number of pseudo-random buckets,
    so texts sharing vocabulary end up closer in cosine space than
    unrelated texts — enough for retrieval tests and the demo.
    """

    def __init__(self, dim: int = 384, seed: int = 42) -> None:
        self.dim = dim
        self.seed = seed

    def _token_vector(self, token: str) -> list[float]:
        # 8 bytes -> 4 float buckets per hash, chained for the full dim.
        vec = [0.0] * self.dim
        for i in range(0, self.dim, 4):
            chunk = hashlib.sha256(
                f"{self.seed}:{token}:{i}".encode()
            ).digest()
            for j in range(4):
                if i + j < self.dim:
                    vec[i + j] = (chunk[j * 2] - 128) / 128.0
        return vec

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            tokens = _TOKEN_RE.findall(text.lower())
            for token in tokens:
                tv = self._token_vector(token)
                for i in range(self.dim):
                    vec[i] += tv[i]
            out.append(_l2_normalize(vec))
        return out


class HuggingFaceEmbeddings:
    """Sentence-Transformers embeddings (lazy import — optional dep)."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [list(map(float, v)) for v in np.asarray(vectors)]


class OpenAIEmbeddings:
    """OpenAI embeddings via LangChain (lazy import — optional dep)."""

    def __init__(self, model: str, api_key: str) -> None:
        from langchain_openai import OpenAIEmbeddings as _OE

        self._impl = _OE(model=model, api_key=api_key or None)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._impl.embed_documents(texts)


def get_embeddings(provider: str, model: str, dim: int, api_key: str = ""):
    """Factory returning an object with an ``embed(texts)`` method."""
    provider = provider.lower()
    if provider == "huggingface":
        try:
            return HuggingFaceEmbeddings(model)
        except Exception as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "HuggingFace embeddings requested but sentence-transformers "
                f"could not be loaded ({exc}). Install requirements-optional.txt "
                "or use CRA_EMBEDDING_PROVIDER=fake."
            ) from exc
    if provider == "openai":
        return OpenAIEmbeddings(model, api_key)
    return FakeEmbeddings(dim=dim)
