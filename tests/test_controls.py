"""Tests for the P4 topology controls.

These are the causal core of the experiment. If a control is not matched exactly
where P4 says "tolerance 0", any difference between arms is attributable to the
mismatch rather than to topology, and the result means nothing.
"""

import numpy as np
import pytest

from flywire_rl.controls import (
    Graph,
    branching_ratio,
    calibrate_branching,
    calibrate_input_gain,
    graph_stats,
    normalise_spectral_radius,
    population_rate_hz,
    random_control,
    shuffled_control,
)


def _graph(n=60, m=400, seed=0):
    rng = np.random.default_rng(seed)
    seen: set[tuple[int, int]] = set()
    while len(seen) < m:
        a, b = int(rng.integers(n)), int(rng.integers(n))
        if a != b:
            seen.add((a, b))
    pre, post = map(np.array, zip(*sorted(seen)))
    return Graph(
        n=n,
        pre=pre.astype(np.int64),
        post=post.astype(np.int64),
        weight=rng.gamma(2.0, 3.0, size=m).astype(np.float32),
        sign=rng.choice([1, -1, 0], size=n, p=[0.75, 0.2, 0.05]).astype(np.int8),
    )


def _degrees(graph: Graph):
    return (
        np.bincount(graph.pre, minlength=graph.n),
        np.bincount(graph.post, minlength=graph.n),
    )


# --------------------------------------------------------------- shuffled


def test_shuffle_preserves_every_node_degree_exactly():
    original = _graph()
    shuffled = shuffled_control(original, seed=1, swaps_per_edge=20)

    for before, after in zip(_degrees(original), _degrees(shuffled)):
        assert np.array_equal(before, after)


def test_shuffle_preserves_edge_count_exactly():
    original = _graph()
    assert len(shuffled_control(original, seed=1, swaps_per_edge=20).pre) == len(
        original.pre
    )


def test_shuffle_preserves_the_weight_multiset_exactly():
    original = _graph()
    shuffled = shuffled_control(original, seed=1, swaps_per_edge=20)

    assert np.array_equal(np.sort(original.weight), np.sort(shuffled.weight))


def test_shuffle_preserves_the_presynaptic_neuron_of_every_edge():
    """This is *why* Dale's law survives: an edge only ever changes its target."""
    original = _graph()
    shuffled = shuffled_control(original, seed=1, swaps_per_edge=20)

    assert np.array_equal(np.sort(original.pre), np.sort(shuffled.pre))
    assert np.array_equal(original.sign, shuffled.sign)


def test_shuffle_keeps_each_weight_attached_to_its_own_presynaptic_neuron():
    original = _graph()
    shuffled = shuffled_control(original, seed=1, swaps_per_edge=20)

    def by_source(graph):
        totals = np.zeros(graph.n)
        np.add.at(totals, graph.pre, graph.weight)
        return totals

    np.testing.assert_allclose(by_source(original), by_source(shuffled), rtol=1e-5)


def test_shuffle_produces_no_self_loops_or_duplicate_edges():
    shuffled = shuffled_control(_graph(), seed=1, swaps_per_edge=20)

    assert not np.any(shuffled.pre == shuffled.post)
    pairs = set(zip(shuffled.pre.tolist(), shuffled.post.tolist()))
    assert len(pairs) == len(shuffled.pre)


def test_shuffle_actually_rewires_rather_than_returning_the_input():
    original = _graph()
    shuffled = shuffled_control(original, seed=1, swaps_per_edge=20)

    before = set(zip(original.pre.tolist(), original.post.tolist()))
    after = set(zip(shuffled.pre.tolist(), shuffled.post.tolist()))
    assert len(before & after) < 0.5 * len(before)


def test_shuffle_changes_higher_order_structure():
    """Degree-matched but structurally different is the entire point of this arm."""
    original = _graph()
    shuffled = shuffled_control(original, seed=1, swaps_per_edge=40)

    assert graph_stats(original)["reciprocity"] != pytest.approx(
        graph_stats(shuffled)["reciprocity"], abs=1e-9
    )


def test_shuffle_is_deterministic_given_a_seed():
    original = _graph()
    a = shuffled_control(original, seed=5, swaps_per_edge=10)
    b = shuffled_control(original, seed=5, swaps_per_edge=10)

    assert np.array_equal(a.pre, b.pre)
    assert np.array_equal(a.post, b.post)


# ----------------------------------------------------------------- random


def test_random_control_matches_size_weights_and_signs():
    original = _graph()
    control = random_control(original, seed=2)

    assert control.n == original.n
    assert len(control.pre) == len(original.pre)
    assert np.array_equal(np.sort(original.weight), np.sort(control.weight))
    assert np.array_equal(np.sort(original.sign), np.sort(control.sign))


def test_random_control_does_not_preserve_the_degree_sequence():
    """P4 requires the degree sequence to be destroyed here, unlike the shuffle."""
    original = _graph()
    control = random_control(original, seed=2)

    assert not np.array_equal(_degrees(original)[0], _degrees(control)[0])


def test_random_control_produces_no_self_loops_or_duplicates():
    control = random_control(_graph(), seed=2)

    assert not np.any(control.pre == control.post)
    pairs = set(zip(control.pre.tolist(), control.post.tolist()))
    assert len(pairs) == len(control.pre)


def test_random_control_is_deterministic_given_a_seed():
    original = _graph()
    a = random_control(original, seed=9)
    b = random_control(original, seed=9)

    assert np.array_equal(a.pre, b.pre) and np.array_equal(a.post, b.post)


# ---------------------------------------------------------- normalisation


def test_spectral_normalisation_hits_the_target_radius():
    original = _graph()
    normalised = normalise_spectral_radius(original, target=0.95)

    assert graph_stats(normalised)["spectral_radius"] == pytest.approx(0.95, rel=1e-3)


def test_spectral_normalisation_is_a_pure_rescale():
    original = _graph()
    normalised = normalise_spectral_radius(original, target=0.95)

    ratios = normalised.weight / original.weight
    np.testing.assert_allclose(ratios, ratios[0], rtol=1e-5)
    assert np.array_equal(original.pre, normalised.pre)


def test_all_three_arms_share_a_spectral_radius_after_normalisation():
    """Without this, a 'topology effect' could just be a gain difference."""
    real = _graph()
    arms = [
        real,
        shuffled_control(real, seed=1, swaps_per_edge=20),
        random_control(real, seed=1),
    ]
    radii = [
        graph_stats(normalise_spectral_radius(arm, target=0.95))["spectral_radius"]
        for arm in arms
    ]

    assert all(r == pytest.approx(0.95, rel=1e-3) for r in radii)


# ------------------------------------------------------------------ stats


def test_graph_stats_reports_everything_p4_requires():
    stats = graph_stats(_graph())
    required = {
        "n",
        "edges",
        "density",
        "in_degree_mean",
        "out_degree_mean",
        "reciprocity",
        "clustering",
        "largest_scc",
        "spectral_radius",
        "excitatory",
        "inhibitory",
        "modulatory",
    }

    assert required <= set(stats)


def test_graph_stats_counts_signs_from_the_per_neuron_vector():
    graph = _graph()
    stats = graph_stats(graph)

    assert stats["excitatory"] == int((graph.sign > 0).sum())
    assert stats["inhibitory"] == int((graph.sign < 0).sum())


# ---------------------------------------------------- dynamical calibration


def test_population_rate_increases_with_input_gain():
    """Bisection in calibrate_input_gain depends on this being monotonic."""
    graph = normalise_spectral_radius(_graph(), target=0.95)
    rates = [population_rate_hz(graph, gain, steps=60) for gain in (0.5, 5.0, 50.0)]

    assert rates[0] < rates[1] < rates[2]


def test_calibration_reaches_the_target_rate_within_tolerance():
    graph = normalise_spectral_radius(_graph(), target=0.95)
    gain = calibrate_input_gain(graph, target_hz=5.0, tolerance_hz=0.5, steps=60)

    assert population_rate_hz(graph, gain, steps=60) == pytest.approx(5.0, abs=0.5)


def test_every_arm_can_be_calibrated_to_the_same_rate():
    """P4's dynamical match: a silent arm learns nothing, so rates must agree."""
    real = normalise_spectral_radius(_graph(), target=0.95)
    arms = [
        real,
        normalise_spectral_radius(shuffled_control(real, seed=1, swaps_per_edge=20)),
        normalise_spectral_radius(random_control(real, seed=1)),
    ]

    for arm in arms:
        gain = calibrate_input_gain(arm, target_hz=5.0, tolerance_hz=0.5, steps=60)
        assert population_rate_hz(arm, gain, steps=60) == pytest.approx(5.0, abs=0.5)


def test_calibration_raises_when_the_target_is_unreachable():
    graph = normalise_spectral_radius(_graph(), target=0.95)

    with pytest.raises(ValueError, match="unreachable"):
        calibrate_input_gain(graph, target_hz=5000.0, tolerance_hz=0.5, steps=40)


# -------------------------------------------------- branching calibration


def test_branching_ratio_rises_with_the_weight_scale():
    """Bisection in calibrate_branching depends on this being monotonic."""
    graph = _graph(n=400, m=4000, seed=1)
    ratios = [branching_ratio(graph, scale) for scale in (0.01, 1.0, 50.0)]

    assert ratios[0] <= ratios[1] <= ratios[2]


def test_a_network_that_cannot_transmit_reports_zero_branching():
    graph = _graph(n=400, m=4000, seed=1)

    assert branching_ratio(graph, scale=1e-6) == 0.0


def test_calibration_reaches_the_target_branching():
    graph = _graph(n=400, m=4000, seed=1)
    scale = calibrate_branching(graph, target=1.0, tolerance=0.2)

    assert branching_ratio(graph, scale) == pytest.approx(1.0, abs=0.2)


def test_every_arm_can_be_calibrated_to_the_same_branching():
    """A4-style match: arms must transmit alike, or topology is confounded
    with whether the substrate propagates at all."""
    real = _graph(n=400, m=4000, seed=1)
    arms = [
        real,
        shuffled_control(real, seed=1, swaps_per_edge=10),
        random_control(real, seed=1),
    ]

    for arm in arms:
        scale = calibrate_branching(arm, target=1.0, tolerance=0.2)
        assert branching_ratio(arm, scale) == pytest.approx(1.0, abs=0.2)


def test_calibration_refuses_an_unreachable_target():
    graph = _graph(n=400, m=4000, seed=1)

    with pytest.raises(ValueError, match="unreachable"):
        calibrate_branching(graph, target=500.0, hi=1.0)
