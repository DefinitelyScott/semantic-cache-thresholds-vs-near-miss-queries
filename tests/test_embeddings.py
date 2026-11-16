"""The optional embedding path, exercised with a fake embedder (no network, no key)."""

import pytest

from cache_eval.embeddings import EmbeddingCosine, dense_cosine, openai_embedder


def test_dense_cosine():
    assert dense_cosine([1.0, 0.0], [2.0, 0.0]) == pytest.approx(1.0)
    assert dense_cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert dense_cosine([0.0, 0.0], [1.0, 1.0]) == 0.0
    with pytest.raises(ValueError):
        dense_cosine([1.0], [1.0, 2.0])


def test_each_distinct_text_is_embedded_once():
    calls: list[list[str]] = []

    def fake_embed(texts: list[str]) -> list[list[float]]:
        calls.append(list(texts))
        return [[float(len(t)), 1.0] for t in texts]

    sim = EmbeddingCosine(fake_embed)
    sim.prefetch(["a", "bb", "a"])
    sim("a", "bb")
    sim("bb", "ccc")
    assert calls == [["a", "bb"], ["ccc"]]


def test_mismatched_embedder_output_is_an_error():
    sim = EmbeddingCosine(lambda texts: [[1.0]])
    with pytest.raises(RuntimeError, match="vectors"):
        sim.prefetch(["a", "b"])


def test_openai_embedder_requires_a_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        openai_embedder()
