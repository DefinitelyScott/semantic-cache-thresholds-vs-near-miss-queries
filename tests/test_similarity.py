import math

import pytest

from cache_eval.similarity import TfidfCosine, cosine, features, jaccard, tokenize


def test_tokenize_handles_possessives_contractions_and_symbols():
    assert tokenize("Austria's capital") == ["austria", "capital"]
    assert tokenize("Which cereals don't contain gluten?") == ["which", "cereals", "dont", "contain", "gluten"]
    assert tokenize("Laptops under $800") == ["laptops", "under", "800"]
    assert tokenize("two-factor") == ["two", "factor"]


def test_features_adds_ordered_word_pairs():
    assert features("from Boston to Denver", ngram_max=2) == [
        "from", "boston", "to", "denver", "from_boston", "boston_to", "to_denver",
    ]
    with pytest.raises(ValueError):
        features("x", ngram_max=3)


def test_jaccard_basic_values():
    assert jaccard("a b c", "a b c") == 1.0
    assert jaccard("a b", "c d") == 0.0
    assert jaccard("a b c", "a b d") == pytest.approx(2 / 4)
    assert jaccard("", "") == 0.0


def test_unigram_measures_ignore_word_order():
    # This is the root of the role-swap failure: same words, different meaning.
    a, b = "Flights from Boston to Denver", "Flights from Denver to Boston"
    model = TfidfCosine([a, b])
    assert jaccard(a, b) == 1.0
    assert model(a, b) == pytest.approx(1.0)


def test_bigrams_make_the_score_order_sensitive():
    a, b = "Flights from Boston to Denver", "Flights from Denver to Boston"
    model = TfidfCosine([a, b], ngram_max=2)
    assert model(a, b) < 0.9


def test_idf_matches_the_smoothed_formula():
    model = TfidfCosine(["red apple", "green apple", "red car"])
    assert model.idf("apple") == pytest.approx(math.log(4 / 3) + 1)
    assert model.idf("car") == pytest.approx(math.log(4 / 2) + 1)
    assert model.idf("unseen") == pytest.approx(math.log(4 / 1) + 1)


def test_rare_shared_word_counts_more_than_common_shared_word():
    corpus = ["the cat", "the dog", "the bird", "a zebra", "the zebra"]
    model = TfidfCosine(corpus)
    # Each pair shares exactly one word; "zebra" is rarer than "the".
    assert model("a zebra", "the zebra") > model("the cat", "the dog")


def test_cosine_edge_cases():
    assert cosine({}, {"a": 1.0}) == 0.0
    assert cosine({"a": 1.0}, {"b": 1.0}) == 0.0
    assert cosine({"a": 3.0}, {"a": 1.0}) == pytest.approx(1.0)


def test_empty_corpus_is_rejected():
    with pytest.raises(ValueError):
        TfidfCosine([])
