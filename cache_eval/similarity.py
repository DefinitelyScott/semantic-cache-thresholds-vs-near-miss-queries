"""Lexical similarity measures used as stand-ins for an embedding cache.

Lexical measures, each a score in [0, 1]:

- ``jaccard``: shared distinct words / all distinct words.
- ``TfidfCosine``: cosine similarity of TF-IDF vectors, so rare words
  (``Austria``) count for more than common ones (``the``). With
  ``ngram_max=2`` it also uses adjacent word pairs (``boston_to``), which makes
  it sensitive to word order.

No stopwords are removed on purpose: words such as ``not`` and ``no`` carry the
meaning in the negation near-misses, and dropping them would hide exactly the
signal under test.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_POSSESSIVE = re.compile(r"'s\b")
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase ``text`` and split it into alphanumeric words.

    Possessive ``'s`` is dropped (``Austria's`` -> ``austria``) and other
    apostrophes are removed so contractions stay one word (``don't`` -> ``dont``).
    Everything else that is not a letter or digit separates words, so
    ``$800`` -> ``800`` and ``two-factor`` -> ``two``, ``factor``.
    """
    lowered = _POSSESSIVE.sub("", text.lower()).replace("'", "")
    return _TOKEN.findall(lowered)


def features(text: str, ngram_max: int = 1) -> list[str]:
    """Words of ``text`` plus, if ``ngram_max`` is 2, joined adjacent word pairs."""
    if ngram_max not in (1, 2):
        raise ValueError(f"ngram_max must be 1 or 2, got {ngram_max}")
    words = tokenize(text)
    if ngram_max == 1:
        return words
    return words + [f"{left}_{right}" for left, right in zip(words, words[1:])]


def jaccard(a: str, b: str) -> float:
    """Jaccard similarity of the two queries' word sets (0.0 if both are empty)."""
    set_a, set_b = set(tokenize(a)), set(tokenize(b))
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


class TfidfCosine:
    """TF-IDF cosine similarity with IDF weights fitted on a fixed corpus.

    IDF uses the smoothed formula ``ln((1 + N) / (1 + df)) + 1`` (the same one
    scikit-learn uses by default), so a word seen in every document still gets
    a small positive weight and an unseen word gets the largest weight.
    """

    def __init__(self, corpus: list[str], ngram_max: int = 1) -> None:
        if not corpus:
            raise ValueError("TfidfCosine needs a non-empty corpus to fit IDF weights")
        self.ngram_max = ngram_max
        self._n_docs = len(corpus)
        doc_freq: Counter[str] = Counter()
        for doc in corpus:
            doc_freq.update(set(features(doc, ngram_max)))
        self._doc_freq = doc_freq

    def idf(self, term: str) -> float:
        """Smoothed inverse document frequency of ``term``."""
        return math.log((1 + self._n_docs) / (1 + self._doc_freq.get(term, 0))) + 1.0

    def vector(self, text: str) -> dict[str, float]:
        """Sparse TF-IDF vector of ``text`` (raw term counts times IDF)."""
        counts = Counter(features(text, self.ngram_max))
        return {term: count * self.idf(term) for term, count in counts.items()}

    def __call__(self, a: str, b: str) -> float:
        """Cosine similarity of the TF-IDF vectors of ``a`` and ``b``."""
        return cosine(self.vector(a), self.vector(b))


def cosine(u: dict[str, float], v: dict[str, float]) -> float:
    """Cosine similarity of two sparse vectors (0.0 if either is all zeros)."""
    dot = sum(weight * v[word] for word, weight in u.items() if word in v)
    norm_u = math.sqrt(sum(w * w for w in u.values()))
    norm_v = math.sqrt(sum(w * w for w in v.values()))
    if norm_u == 0.0 or norm_v == 0.0:
        return 0.0
    return dot / (norm_u * norm_v)
