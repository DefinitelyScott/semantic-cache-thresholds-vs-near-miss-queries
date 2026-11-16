import pytest

from cache_eval.metrics import THRESHOLDS, lowest_safe_threshold, roc_auc, served, served_rate, sign_test_p


def test_threshold_grid():
    assert THRESHOLDS[0] == 0.0 and THRESHOLDS[-1] == 1.0 and len(THRESHOLDS) == 21


def test_served_uses_a_tolerance_for_float_noise():
    assert served(0.9999999999999998, 1.0)
    assert served(0.5, 0.5)
    assert not served(0.49, 0.5)


def test_served_rate():
    assert served_rate([0.1, 0.5, 0.9, 1.0], 0.5) == 0.75
    with pytest.raises(ValueError):
        served_rate([], 0.5)


def test_roc_auc_perfect_inverted_and_tied():
    assert roc_auc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert roc_auc([0.1, 0.2], [0.9, 0.8]) == 0.0
    assert roc_auc([0.5], [0.5]) == 0.5
    # 3 of 4 pairs won, none tied.
    assert roc_auc([0.9, 0.3], [0.2, 0.5]) == 0.75
    with pytest.raises(ValueError):
        roc_auc([], [0.1])


def test_lowest_safe_threshold():
    assert lowest_safe_threshold([0.0, 0.12, 0.31]) == 0.35
    assert lowest_safe_threshold([0.30]) == 0.35
    assert lowest_safe_threshold([0.2999]) == 0.30
    assert lowest_safe_threshold([0.4, 1.0]) is None


def test_sign_test_p_known_values():
    assert sign_test_p(5, 10) == 1.0
    assert sign_test_p(10, 10) == pytest.approx(2 / 1024)
    assert sign_test_p(0, 10) == pytest.approx(2 / 1024)
    # P(X <= 1) for Binomial(6, 0.5) is 7/64; two-sided doubles it.
    assert sign_test_p(5, 6) == pytest.approx(14 / 64)
    with pytest.raises(ValueError):
        sign_test_p(3, 2)
    with pytest.raises(ValueError):
        sign_test_p(0, 0)
