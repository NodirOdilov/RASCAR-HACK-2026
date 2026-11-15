import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from rascar import metrics


def test_ap_orientation_high_score_means_error():
    y = np.array([1, 1, 0, 0])
    assert metrics.average_precision(y, [0.9, 0.8, 0.2, 0.1]) == pytest.approx(1.0)
    assert metrics.average_precision(y, [0.1, 0.2, 0.8, 0.9]) < 0.6       # inverted ranking is punished


def test_ap_known_value():
    # ranking: 1, 0, 1, 0 -> precisions at the positives: 1/1 and 2/3
    assert metrics.average_precision([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.6]) == pytest.approx((1 + 2 / 3) / 2)


def test_constant_score_equals_prevalence_and_matches_sklearn_ties():
    y = np.array([1, 0, 0, 1, 0, 0, 0, 1])
    assert metrics.average_precision(y, np.full(8, 0.5)) == pytest.approx(y.mean())
    s = np.array([0.5, 0.5, 0.2, 0.9, 0.9, 0.1, 0.1, 0.5])
    assert metrics.average_precision(y, s) == pytest.approx(average_precision_score(y, s))


def test_baseline_direction_one_minus_confidence():
    conf = np.array([0.95, 0.9, 0.5, 0.45])
    error = np.array([0, 0, 1, 1])
    assert metrics.average_precision(error, 1 - conf) == pytest.approx(1.0)
    assert metrics.average_precision(error, conf) < 1.0


def test_precision_at_k():
    y, s = [1, 0, 1, 1, 0], [0.9, 0.8, 0.7, 0.6, 0.5]
    assert metrics.precision_at_k(y, s, 1) == 1.0
    assert metrics.precision_at_k(y, s, 2) == 0.5
    assert metrics.precision_at_k(y, s, 100) == pytest.approx(3 / 5)      # k larger than n is clipped


def test_no_positive_gives_nan_not_exception():
    assert np.isnan(metrics.average_precision([0, 0, 0], [0.1, 0.2, 0.3]))


def test_bootstrap_is_deterministic_and_brackets_the_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 120)
    s = y * 0.3 + rng.uniform(0, 1, 120)
    groups = np.repeat(np.arange(30), 4)
    a = metrics.bootstrap_ap(y, s, groups, n_boot=300, seed=5)
    b = metrics.bootstrap_ap(y, s, groups, n_boot=300, seed=5)
    assert a == b
    assert a["ci_low"] <= a["ap"] <= a["ci_high"]


def test_paired_bootstrap_detects_a_clearly_better_score():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    good = y + rng.normal(0, 0.3, 200)
    noise = rng.normal(0, 1, 200)
    out = metrics.paired_bootstrap_delta(y, good, noise, np.repeat(np.arange(50), 4), n_boot=300, seed=2)
    assert out["delta"] > 0.2 and out["p_a_better"] > 0.99 and out["ci_low"] > 0
