"""Tests for the P5 statistical protocol.

Written before any confirmatory run exists, which is the point: an analysis
implemented after seeing the numbers is no longer pre-registered, however
honestly it is done.
"""

import numpy as np
import pytest

from flywire_rl.analysis import (
    compare_arms,
    holm_adjust,
    iqm,
    probability_of_improvement,
    stratified_bootstrap_ci,
)


# ------------------------------------------------------------------- IQM


def test_iqm_is_the_mean_of_the_middle_half():
    # 0..99: the interquartile range is 25..74, whose mean is 49.5.
    assert iqm(np.arange(100.0)) == pytest.approx(49.5)


def test_iqm_ignores_outliers_that_drag_the_mean():
    clean = np.concatenate([np.zeros(50), np.ones(50)])
    spoiled = clean.copy()
    spoiled[0] = -1e6
    spoiled[-1] = 1e6

    assert np.mean(spoiled) != pytest.approx(np.mean(clean))
    assert iqm(spoiled) == pytest.approx(iqm(clean))


def test_iqm_handles_a_short_sample():
    assert iqm(np.array([1.0, 2.0])) == pytest.approx(1.5)


# ------------------------------------------------------------- bootstrap


def test_bootstrap_is_deterministic_given_a_seed():
    scores = np.random.default_rng(0).normal(size=(10, 40))

    first = stratified_bootstrap_ci(scores, seed=7, n_resamples=200)
    second = stratified_bootstrap_ci(scores, seed=7, n_resamples=200)

    assert first == second


def test_bootstrap_interval_brackets_the_point_estimate():
    scores = np.random.default_rng(1).normal(loc=0.4, size=(10, 40))

    low, high = stratified_bootstrap_ci(scores, seed=0, n_resamples=500)

    assert low < iqm(scores.ravel()) < high


def test_bootstrap_narrows_as_seeds_are_added():
    rng = np.random.default_rng(2)
    few = rng.normal(loc=0.3, size=(4, 40))
    many = rng.normal(loc=0.3, size=(40, 40))

    narrow = np.diff(stratified_bootstrap_ci(many, seed=0, n_resamples=500))
    wide = np.diff(stratified_bootstrap_ci(few, seed=0, n_resamples=500))

    assert narrow < wide


# ------------------------------------------- probability of improvement


def test_probability_of_improvement_is_one_when_an_arm_dominates():
    better = np.full((5, 10), 1.0)
    worse = np.full((5, 10), -1.0)

    assert probability_of_improvement(better, worse) == pytest.approx(1.0)


def test_probability_of_improvement_counts_ties_as_half():
    same = np.zeros((5, 10))

    assert probability_of_improvement(same, same) == pytest.approx(0.5)


def test_probability_of_improvement_is_antisymmetric():
    rng = np.random.default_rng(3)
    a, b = rng.normal(0.5, size=(6, 20)), rng.normal(size=(6, 20))

    forward = probability_of_improvement(a, b)
    backward = probability_of_improvement(b, a)

    assert forward + backward == pytest.approx(1.0)


# ------------------------------------------------------------------ Holm


def test_holm_matches_a_worked_example():
    # Two hypotheses: the smallest p is multiplied by 2, the largest by 1, and
    # monotonicity is enforced afterwards.
    adjusted = holm_adjust([0.01, 0.04])

    assert adjusted == pytest.approx([0.02, 0.04])


def test_holm_never_reduces_a_p_value():
    raw = [0.2, 0.01, 0.5]
    adjusted = holm_adjust(raw)

    assert all(a >= r for a, r in zip(adjusted, raw))


def test_holm_is_monotonic_in_the_sorted_order():
    raw = [0.001, 0.02, 0.03]
    adjusted = np.array(holm_adjust(raw))
    order = np.argsort(raw)

    assert list(adjusted[order]) == sorted(adjusted[order])


def test_holm_caps_at_one():
    assert max(holm_adjust([0.6, 0.7])) <= 1.0


# ----------------------------------------------------------- comparison


def test_compare_arms_detects_a_real_effect():
    rng = np.random.default_rng(4)
    arms = {
        "real": rng.normal(loc=0.6, scale=0.3, size=(10, 40)),
        "shuffled": rng.normal(loc=0.0, scale=0.3, size=(10, 40)),
        "random": rng.normal(loc=-0.1, scale=0.3, size=(10, 40)),
    }

    result = compare_arms(arms, seed=0, n_resamples=500)

    assert result["comparisons"]["real_vs_shuffled"]["reject"] is True
    assert result["comparisons"]["real_vs_shuffled"]["ci"][0] > 0


def test_compare_arms_does_not_invent_an_effect():
    rng = np.random.default_rng(5)
    arms = {
        "real": rng.normal(size=(10, 40)),
        "shuffled": rng.normal(size=(10, 40)),
        "random": rng.normal(size=(10, 40)),
    }

    result = compare_arms(arms, seed=0, n_resamples=500)
    interval = result["comparisons"]["real_vs_shuffled"]["ci"]

    assert interval[0] < 0 < interval[1]
    assert result["comparisons"]["real_vs_shuffled"]["reject"] is False


def test_compare_arms_reports_everything_p5_requires():
    rng = np.random.default_rng(6)
    arms = {name: rng.normal(size=(6, 20)) for name in ("real", "shuffled", "random")}

    result = compare_arms(arms, seed=0, n_resamples=200)

    assert {"iqm", "ci", "comparisons", "alpha", "n_resamples"} <= set(result)
    for key in ("real_vs_shuffled", "real_vs_random"):
        entry = result["comparisons"][key]
        assert {
            "difference",
            "ci",
            "p_value",
            "p_value_holm",
            "probability_of_improvement",
            "reject",
        } <= set(entry)


def test_compare_arms_applies_holm_across_both_comparisons():
    rng = np.random.default_rng(7)
    arms = {
        "real": rng.normal(loc=0.3, size=(8, 30)),
        "shuffled": rng.normal(size=(8, 30)),
        "random": rng.normal(size=(8, 30)),
    }

    result = compare_arms(arms, seed=0, n_resamples=300)

    for entry in result["comparisons"].values():
        assert entry["p_value_holm"] >= entry["p_value"]


def test_compare_arms_requires_the_named_arms():
    with pytest.raises(KeyError, match="shuffled"):
        compare_arms({"real": np.zeros((3, 5)), "random": np.zeros((3, 5))})
