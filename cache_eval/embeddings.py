"""OPTIONAL: score the same queries with a real embedding model.

Production semantic caches compare embeddings, not word overlap. This module
lets the experiment run with OpenAI embeddings so the lexical result can be
checked against a real model. It is never used by the tests or the default
experiment, and it needs ``pip install openai`` plus an ``OPENAI_API_KEY``.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Sequence

Embedder = Callable[[list[str]], list[list[float]]]


class EmbeddingCosine:
    """Cosine similarity of embeddings, embedding each distinct text only once."""

    def __init__(self, embed: Embedder) -> None:
        self._embed = embed
        self._cache: dict[str, list[float]] = {}

    def prefetch(self, texts: Sequence[str]) -> None:
        """Embed every not-yet-seen text in one batched call."""
        new = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if not new:
            return
        vectors = self._embed(new)
        if len(vectors) != len(new):
            raise RuntimeError(f"embedder returned {len(vectors)} vectors for {len(new)} texts")
        self._cache.update(zip(new, vectors))

    def __call__(self, a: str, b: str) -> float:
        self.prefetch([a, b])
        return dense_cosine(self._cache[a], self._cache[b])


def dense_cosine(u: Sequence[float], v: Sequence[float]) -> float:
    """Cosine similarity of two dense vectors (0.0 if either is all zeros)."""
    if len(u) != len(v):
        raise ValueError(f"vector lengths differ: {len(u)} vs {len(v)}")
    dot = sum(x * y for x, y in zip(u, v))
    norm = math.sqrt(sum(x * x for x in u)) * math.sqrt(sum(y * y for y in v))
    return 0.0 if norm == 0.0 else dot / norm


def openai_embedder(model: str = "text-embedding-3-small") -> Embedder:
    """Build an embedder backed by the OpenAI embeddings API.

    Raises:
        RuntimeError: if ``OPENAI_API_KEY`` is not set or ``openai`` is not installed.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set; the embedding check is optional and needs a key")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("install the optional dependency first: pip install openai==1.109.1") from exc

    client = OpenAI()

    def embed(texts: list[str]) -> list[list[float]]:
        response = client.embeddings.create(model=model, input=texts)
        return [item.embedding for item in sorted(response.data, key=lambda d: d.index)]

    return embed
