"""Run the cache-threshold experiment and format the text report.

For every anchor (a cached query) three incoming queries are scored:

- its paraphrase (should be served from the cache),
- its near-miss (should not), and
- an *unrelated* query: the paraphrase of the anchor 15 places further down
  the list (wrapping around). Shifting by half the list guarantees it comes
  from a different near-miss type and usually a different topic. Unrelated
  pairs are the "easy negatives" a cache is often tuned against.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from cache_eval.dataset import NEAR_MISS_TYPES, Triple, all_queries
from cache_eval.guard import guard_allows
from cache_eval.metrics import (
    THRESHOLDS,
    lowest_safe_threshold,
    roc_auc,
    served,
    served_rate,
    sign_test_p,
)
from cache_eval.similarity import TfidfCosine, jaccard

Similarity = Callable[[str, str], float]

UNRELATED_SHIFT = 15


@dataclass(frozen=True)
class ScoredTriple:
    """Similarity of one anchor to each of its three incoming queries."""

    triple: Triple
    unrelated_text: str
    paraphrase: float
    near_miss: float
    unrelated: float


@dataclass(frozen=True)
class Summary:
    """Headline numbers for one similarity measure."""

    measure: str
    auc_vs_unrelated: float
    auc_vs_near_miss: float
    near_miss_above_paraphrase: int
    near_miss_below_paraphrase: int
    ties: int
    sign_test_p: float
    threshold_tuned_on_unrelated: float | None
    paraphrase_hit_rate: float | None
    near_miss_false_hit_rate: float | None
    threshold_tuned_on_near_miss: float | None
    paraphrase_hit_rate_strict: float | None


def unrelated_partner(triples: list[Triple], index: int) -> str:
    """The unrelated incoming query paired with ``triples[index]``."""
    return triples[(index + UNRELATED_SHIFT) % len(triples)].paraphrase


def score_triples(triples: list[Triple], similarity: Similarity) -> list[ScoredTriple]:
    """Score every anchor against its paraphrase, near-miss and unrelated query."""
    scored = []
    for index, triple in enumerate(triples):
        other = unrelated_partner(triples, index)
        scored.append(
            ScoredTriple(
                triple=triple,
                unrelated_text=other,
                paraphrase=similarity(triple.anchor, triple.paraphrase),
                near_miss=similarity(triple.anchor, triple.near_miss),
                unrelated=similarity(triple.anchor, other),
            )
        )
    return scored


def build_measures(triples: list[Triple]) -> dict[str, Similarity]:
    """The offline similarity measures. TF-IDF weights are fitted on every query."""
    corpus = all_queries(triples)
    return {
        "tfidf_cosine": TfidfCosine(corpus),
        "tfidf_cosine_bigrams": TfidfCosine(corpus, ngram_max=2),
        "jaccard": jaccard,
    }


def summarize(measure: str, scored: list[ScoredTriple]) -> Summary:
    """Compute the headline numbers for one measure."""
    para = [s.paraphrase for s in scored]
    near = [s.near_miss for s in scored]
    unrel = [s.unrelated for s in scored]

    above = sum(s.near_miss > s.paraphrase + 1e-9 for s in scored)
    below = sum(s.near_miss < s.paraphrase - 1e-9 for s in scored)
    ties = len(scored) - above - below

    t_easy = lowest_safe_threshold(unrel)
    t_hard = lowest_safe_threshold(near)
    return Summary(
        measure=measure,
        auc_vs_unrelated=roc_auc(para, unrel),
        auc_vs_near_miss=roc_auc(para, near),
        near_miss_above_paraphrase=above,
        near_miss_below_paraphrase=below,
        ties=ties,
        sign_test_p=sign_test_p(above, above + below) if above + below else 1.0,
        threshold_tuned_on_unrelated=t_easy,
        paraphrase_hit_rate=served_rate(para, t_easy) if t_easy is not None else None,
        near_miss_false_hit_rate=served_rate(near, t_easy) if t_easy is not None else None,
        threshold_tuned_on_near_miss=t_hard,
        paraphrase_hit_rate_strict=served_rate(para, t_hard) if t_hard is not None else None,
    )


def threshold_table(scored: list[ScoredTriple], thresholds: tuple[float, ...] = THRESHOLDS) -> list[dict]:
    """Served rates for each kind of incoming query at each threshold."""
    rows = []
    for t in thresholds:
        rows.append({
            "threshold": t,
            "paraphrase_hit": served_rate([s.paraphrase for s in scored], t),
            "near_miss_false_hit": served_rate([s.near_miss for s in scored], t),
            "unrelated_false_hit": served_rate([s.unrelated for s in scored], t),
        })
    return rows


def by_type(scored: list[ScoredTriple], threshold: float) -> list[dict]:
    """Per near-miss type present in ``scored``: mean similarities and near-misses served."""
    rows = []
    for kind in NEAR_MISS_TYPES:
        group = [s for s in scored if s.triple.type == kind]
        if not group:
            continue
        rows.append({
            "type": kind,
            "n": len(group),
            "mean_paraphrase": sum(s.paraphrase for s in group) / len(group),
            "mean_near_miss": sum(s.near_miss for s in group) / len(group),
            "near_miss_served": sum(served(s.near_miss, threshold) for s in group),
            "near_miss_served_with_guard": sum(
                served(s.near_miss, threshold) and guard_allows(s.triple.anchor, s.triple.near_miss)
                for s in group
            ),
        })
    return rows


def guard_effect(scored: list[ScoredTriple], threshold: float) -> dict[str, tuple[int, int]]:
    """Counts served (without guard, with guard) for each kind of incoming query."""
    def counts(pick: Callable[[ScoredTriple], tuple[float, str]]) -> tuple[int, int]:
        plain = guarded = 0
        for s in scored:
            score, text = pick(s)
            if served(score, threshold):
                plain += 1
                guarded += guard_allows(s.triple.anchor, text)
        return plain, guarded

    return {
        "paraphrase": counts(lambda s: (s.paraphrase, s.triple.paraphrase)),
        "near_miss": counts(lambda s: (s.near_miss, s.triple.near_miss)),
        "unrelated": counts(lambda s: (s.unrelated, s.unrelated_text)),
    }


def guard_on_paraphrases(triples: list[Triple]) -> list[Triple]:
    """Triples whose paraphrase the guard would wrongly block, at any threshold."""
    return [t for t in triples if not guard_allows(t.anchor, t.paraphrase)]


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _thr(x: float | None) -> str:
    return "none" if x is None else f"{x:.2f}"


def format_headline(summaries: list[Summary]) -> list[str]:
    """One row per measure: the numbers quoted in the README's headline table."""
    lines = [
        "measure                AUC_vs_unrelated  AUC_vs_near_miss  t_easy  paraphrases_served  near_misses_served",
    ]
    for s in summaries:
        lines.append(
            f"{s.measure:<22} {s.auc_vs_unrelated:>16.3f}  {s.auc_vs_near_miss:>16.3f}  "
            f"{_thr(s.threshold_tuned_on_unrelated):>6}  {_pct(s.paraphrase_hit_rate):>18}  "
            f"{_pct(s.near_miss_false_hit_rate):>18}"
        )
    return lines


def format_measure(name: str, scored: list[ScoredTriple]) -> list[str]:
    """The detailed report block for one similarity measure."""
    n = len(scored)
    s = summarize(name, scored)
    lines = [
        f"=== {name} ===",
        f"ROC AUC, paraphrase vs. unrelated : {s.auc_vs_unrelated:.3f}",
        f"ROC AUC, paraphrase vs. near-miss : {s.auc_vs_near_miss:.3f}",
        f"near-miss scored above its paraphrase in {s.near_miss_above_paraphrase}/{n} anchors "
        f"(below: {s.near_miss_below_paraphrase}, ties: {s.ties}; sign test p = {s.sign_test_p:.2g})",
        "",
        f"threshold tuned on unrelated pairs (lowest with 0 false hits): {_thr(s.threshold_tuned_on_unrelated)}",
        f"  paraphrases served          : {_pct(s.paraphrase_hit_rate)}",
        f"  near-misses wrongly served  : {_pct(s.near_miss_false_hit_rate)}",
        f"threshold tuned on near-misses (lowest with 0 false hits): {_thr(s.threshold_tuned_on_near_miss)}",
        f"  paraphrases served          : {_pct(s.paraphrase_hit_rate_strict)}",
        "",
        "threshold  paraphrase_hit  near_miss_false_hit  unrelated_false_hit",
    ]
    for row in threshold_table(scored):
        if row["threshold"] < 0.10:
            continue
        lines.append(
            f"   {row['threshold']:.2f}        {_pct(row['paraphrase_hit']):>6}"
            f"               {_pct(row['near_miss_false_hit']):>6}"
            f"               {_pct(row['unrelated_false_hit']):>6}"
        )
    t = s.threshold_tuned_on_unrelated
    if t is not None:
        lines += ["", f"by near-miss type, at threshold {t:.2f}:",
                  "type        n  mean_sim_paraphrase  mean_sim_near_miss  served  served_with_guard"]
        for row in by_type(scored, t):
            lines.append(
                f"{row['type']:<10} {row['n']:>2}  {row['mean_paraphrase']:>19.3f}  "
                f"{row['mean_near_miss']:>18.3f}  {row['near_miss_served']:>3}/{row['n']}"
                f"  {row['near_miss_served_with_guard']:>9}/{row['n']}"
            )
        lines += ["", f"number/negation guard at threshold {t:.2f} (served without -> with guard):"]
        for kind, (plain, guarded) in guard_effect(scored, t).items():
            lines.append(f"  {kind:<11}: {plain:>2}/{n} -> {guarded:>2}/{n}")
    return lines


def format_report(triples: list[Triple]) -> str:
    """Run every offline measure and return the full plain-text report."""
    n = len(triples)
    scored_by_measure = {
        name: score_triples(triples, similarity) for name, similarity in build_measures(triples).items()
    }
    summaries = [summarize(name, scored) for name, scored in scored_by_measure.items()]

    lines = [f"{n} cached queries, each with 1 paraphrase, 1 near-miss and 1 unrelated query", ""]
    lines += format_headline(summaries) + [""]
    for name, scored in scored_by_measure.items():
        lines += format_measure(name, scored) + [""]

    blocked = guard_on_paraphrases(triples)
    lines.append(f"paraphrases the guard would block at any threshold: {len(blocked)}/{n}")
    for t in blocked:
        lines.append(f"  #{t.id}: {t.anchor!r} vs. {t.paraphrase!r}")
    return "\n".join(lines).rstrip() + "\n"
