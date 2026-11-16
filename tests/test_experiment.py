"""Core experiment logic, checked on small hand-made inputs with a fake similarity."""

from cache_eval.dataset import Triple
from cache_eval.experiment import (
    UNRELATED_SHIFT,
    by_type,
    guard_effect,
    score_triples,
    summarize,
    threshold_table,
    unrelated_partner,
)

TRIPLES = [
    Triple(1, "number", "timer for 10 minutes", "start a 10 minute timer", "timer for 30 minutes"),
    Triple(2, "antonym", "zoom in on a map", "closer view on the map", "zoom out on a map"),
]

# A fake similarity with fixed answers, so the expected numbers are obvious.
FIXED = {
    ("timer for 10 minutes", "start a 10 minute timer"): 0.6,
    ("timer for 10 minutes", "timer for 30 minutes"): 0.9,
    ("zoom in on a map", "closer view on the map"): 0.4,
    ("zoom in on a map", "zoom out on a map"): 0.8,
}


def fake_similarity(a: str, b: str) -> float:
    return FIXED.get((a, b), 0.05)


def test_unrelated_partner_shifts_by_half_the_list_and_wraps():
    triples = [Triple(i, "number", f"a{i}", f"p{i}", f"n{i}") for i in range(30)]
    assert UNRELATED_SHIFT == 15
    assert unrelated_partner(triples, 0) == "p15"
    assert unrelated_partner(triples, 20) == "p5"


def test_score_triples_scores_each_kind_against_the_anchor():
    scored = score_triples(TRIPLES, fake_similarity)
    assert [(s.paraphrase, s.near_miss, s.unrelated) for s in scored] == [(0.6, 0.9, 0.05), (0.4, 0.8, 0.05)]
    assert scored[0].unrelated_text == TRIPLES[1].paraphrase


def test_summarize_on_known_scores():
    s = summarize("fake", score_triples(TRIPLES, fake_similarity))
    assert s.auc_vs_unrelated == 1.0
    assert s.auc_vs_near_miss == 0.0
    assert (s.near_miss_above_paraphrase, s.near_miss_below_paraphrase, s.ties) == (2, 0, 0)
    assert s.threshold_tuned_on_unrelated == 0.10
    assert s.paraphrase_hit_rate == 1.0
    assert s.near_miss_false_hit_rate == 1.0
    assert s.threshold_tuned_on_near_miss == 0.95
    assert s.paraphrase_hit_rate_strict == 0.0


def test_threshold_table_rows():
    rows = {r["threshold"]: r for r in threshold_table(score_triples(TRIPLES, fake_similarity))}
    assert rows[0.5] == {"threshold": 0.5, "paraphrase_hit": 0.5,
                         "near_miss_false_hit": 1.0, "unrelated_false_hit": 0.0}


def test_guard_blocks_the_number_near_miss_but_not_the_antonym():
    scored = score_triples(TRIPLES, fake_similarity)
    effect = guard_effect(scored, 0.10)
    assert effect["near_miss"] == (2, 1)
    assert effect["paraphrase"] == (2, 2)
    rows = {r["type"]: r for r in by_type(scored, 0.10)}
    assert set(rows) == {"number", "antonym"}
    assert rows["number"]["near_miss_served_with_guard"] == 0
    assert rows["antonym"]["near_miss_served_with_guard"] == 1
