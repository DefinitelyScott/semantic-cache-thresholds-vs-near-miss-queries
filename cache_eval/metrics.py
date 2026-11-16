"""Metrics for a similarity-threshold cache.

The cache reuses a stored answer when ``similarity >= threshold``. For a
paraphrase that is a correct reuse (a *hit*); for a near-miss or an unrelated
query it serves the wrong answer (a *false hit*).
"""

from __future__ import annotations

from math import comb

# Scores are compared with a small tolerance so that two identical word bags,
# whose cosine can come out as 0.9999999999999998, still count as 1.0.
EPS = 1e-9

# The threshold grid used everywhere: 0.00, 0.05, ..., 1.00.
THRESHOLDS: tuple[float, ...] = tuple(round(i * 0.05, 2) for i in range(21))


def served(score: float, threshold: float) -> bool:
    """True if a query with this similarity would be answered from the cache."""
    return score >= threshold - EPS


def served_rate(scores: list[float], threshold: float) -> float:
    """Fraction of ``scores`` that the cache would serve at ``threshold``."""
    if not scores:
        raise ValueError("served_rate needs at least one score")
    return sum(served(s, threshold) for s in scores) / len(scores)


def roc_auc(positives: list[float], negatives: list[float]) -> float:
    """Probability that a random positive outscores a random negative.

    This is the Mann-Whitney form of ROC AUC: ties count as half. 1.0 means
    some threshold separates the two groups perfectly; 0.5 means the score
    carries no information.
    """
    if not positives or not negatives:
        raise ValueError("roc_auc needs at least one positive and one negative")
    wins = 0.0
    for p in positives:
        for n in negatives:
            if abs(p - n) <= EPS:
                wins += 0.5
            elif p > n:
                wins += 1.0
    return wins / (len(positives) * len(negatives))


def lowest_safe_threshold(negatives: list[float], thresholds: tuple[float, ...] = THRESHOLDS) -> float | None:
    """Smallest grid threshold at which none of ``negatives`` would be served.

    This is how a team might tune a cache: "raise the threshold until the bad
    examples stop getting through." Returns None when no grid threshold works,
    which happens when a negative scores 1.0 (an identical bag of words).
    """
    for threshold in sorted(thresholds):
        if not any(served(n, threshold) for n in negatives):
            return threshold
    return None


def sign_test_p(successes: int, trials: int) -> float:
    """Exact two-sided binomial sign test against p = 0.5.

    Used for the per-anchor question "does the near-miss score higher than the
    paraphrase?" with ties dropped beforehand.
    """
    if trials <= 0:
        raise ValueError("sign_test_p needs at least one trial")
    if not 0 <= successes <= trials:
        raise ValueError("successes must be between 0 and trials")
    k = min(successes, trials - successes)
    tail = sum(comb(trials, i) for i in range(k + 1)) / 2**trials
    return min(1.0, 2 * tail)
