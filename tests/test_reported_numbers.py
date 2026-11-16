"""Every number quoted in README.md, pinned so the docs cannot drift from the code."""

import pytest

from cache_eval.dataset import load_triples
from cache_eval.experiment import build_measures, by_type, guard_effect, guard_on_paraphrases, score_triples, summarize
from cache_eval.metrics import served_rate

TRIPLES = load_triples()
SCORED = {name: score_triples(TRIPLES, sim) for name, sim in build_measures(TRIPLES).items()}
SUMMARY = {name: summarize(name, scored) for name, scored in SCORED.items()}


@pytest.mark.parametrize(
    "measure, auc_easy, auc_hard, t_easy, para_served, t_hard",
    [
        ("tfidf_cosine", 0.999, 0.020, 0.20, 29 / 30, None),
        ("tfidf_cosine_bigrams", 0.998, 0.004, 0.15, 26 / 30, 0.85),
        ("jaccard", 0.989, 0.002, 0.25, 21 / 30, None),
    ],
)
def test_headline_table(measure, auc_easy, auc_hard, t_easy, para_served, t_hard):
    s = SUMMARY[measure]
    assert round(s.auc_vs_unrelated, 3) == auc_easy
    assert round(s.auc_vs_near_miss, 3) == auc_hard
    assert s.threshold_tuned_on_unrelated == t_easy
    assert s.paraphrase_hit_rate == pytest.approx(para_served)
    assert s.near_miss_false_hit_rate == 1.0
    assert s.threshold_tuned_on_near_miss == t_hard
    # Every measure ranks the near-miss above the paraphrase for all 30 anchors.
    assert (s.near_miss_above_paraphrase, s.ties) == (30, 0)
    assert s.sign_test_p == pytest.approx(2 / 2**30)


def test_bigram_threshold_that_stops_near_misses_serves_no_paraphrases():
    assert SUMMARY["tfidf_cosine_bigrams"].paraphrase_hit_rate_strict == 0.0


def test_role_swaps_score_exactly_one_without_bigrams():
    for measure in ("tfidf_cosine", "jaccard"):
        swaps = [s.near_miss for s in SCORED[measure] if s.triple.type == "role_swap"]
        assert swaps == pytest.approx([1.0] * 6)
    scored = SCORED["tfidf_cosine"]
    assert served_rate([s.near_miss for s in scored], 1.0) == pytest.approx(6 / 30)


def test_threshold_rows_quoted_in_readme():
    scored = SCORED["tfidf_cosine"]
    para = [s.paraphrase for s in scored]
    near = [s.near_miss for s in scored]
    assert served_rate(para, 0.50) == pytest.approx(13 / 30)
    assert served_rate(near, 0.50) == 1.0
    assert served_rate(para, 0.80) == 0.0
    assert served_rate(near, 0.80) == pytest.approx(18 / 30)


def test_closest_unrelated_pair_sits_just_under_the_tuned_threshold():
    worst = max(s.unrelated for s in SCORED["tfidf_cosine"])
    assert round(worst, 4) == 0.1999


def test_mean_similarity_by_type_tfidf():
    rows = {r["type"]: r for r in by_type(SCORED["tfidf_cosine"], 0.20)}
    expected = {
        "number": (0.479, 0.793),
        "negation": (0.355, 0.873),
        "role_swap": (0.583, 1.000),
        "entity": (0.594, 0.756),
        "antonym": (0.411, 0.792),
    }
    for kind, (para, near) in expected.items():
        assert round(rows[kind]["mean_paraphrase"], 3) == para
        assert round(rows[kind]["mean_near_miss"], 3) == near


def test_guard_effect_tfidf_at_tuned_threshold():
    effect = guard_effect(SCORED["tfidf_cosine"], 0.20)
    assert effect == {"paraphrase": (29, 29), "near_miss": (30, 18), "unrelated": (0, 0)}
    rows = {r["type"]: r["near_miss_served_with_guard"] for r in by_type(SCORED["tfidf_cosine"], 0.20)}
    assert rows == {"number": 0, "negation": 0, "role_swap": 6, "entity": 6, "antonym": 6}


def test_guard_never_blocks_a_paraphrase():
    assert guard_on_paraphrases(TRIPLES) == []
