"""Tests for the P4 topology controls.

These are the causal core of the experiment. If a control is not matched exactly
where P4 says "tolerance 0", any difference between arms is attributable to the
mismatch rather than to topology, and the result means nothing.
"""

import numpy as np
import pytest

from flywire_rl.spiking import ShiuParams
from flywire_rl.controls import (
    Graph,
    W_SYN_CANDIDATES,
    activity_profile,
    calibrate_w_syn,
    graph_stats,
    normalise_spectral_radius,
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


# ------------------------------------------------- functional calibration
#
# The pilot calibrated to a dynamical constant -- first rho = 0.95, then a
# branching ratio of 1 -- and never asked whether the readout saw a spike. On
# the v783 subset it did not: 100% of spikes stayed in the directly driven input
# neurons. What is calibrated now is the one free parameter of the Shiu model,
# w_syn, against what the substrate has to do: light the readout, stay below
# runaway, and respond to the input rather than to itself.

INPUT_IDX = np.arange(20)
READOUT_IDX = np.arange(380, 400)


def _big_graph():
    return _graph(n=400, m=4000, seed=1)


def test_a_blank_input_leaves_the_network_completely_silent():
    """No intrinsic noise: an undriven Shiu network has a baseline of exactly 0."""
    profile = activity_profile(
        _big_graph(), 0.0, INPUT_IDX, READOUT_IDX, steps=300
    )

    assert profile.rate_hz == 0.0
    assert profile.active_fraction == 0.0
    assert profile.readout_live == 0


def test_activity_rises_with_the_per_synapse_scale():
    """The sweep walks w_syn upward, so activity must not fall as it grows."""
    graph = _big_graph()
    rates = [
        activity_profile(
            graph, 0.06, INPUT_IDX, READOUT_IDX, params=ShiuParams(w_syn_mv=w), steps=300
        ).rate_hz
        for w in (0.275, 2.2, 8.8)
    ]

    assert rates[0] <= rates[1] <= rates[2]


def test_calibration_picks_a_scale_that_lights_the_readout():
    """The criterion the pilot lacked: the readout must actually carry signal."""
    graph = _big_graph()
    w_syn, sweep = calibrate_w_syn(
        graph, INPUT_IDX, READOUT_IDX, drive_mv=0.06, steps=300
    )

    chosen = activity_profile(
        graph, 0.06, INPUT_IDX, READOUT_IDX, params=ShiuParams(w_syn_mv=w_syn), steps=300
    )
    assert chosen.readout_live >= 1
    assert chosen.active_fraction <= 0.25
    assert sweep, "the sweep must be returned for the run's provenance"


def test_calibration_records_every_candidate_it_tried_and_stops_at_the_winner():
    graph = _big_graph()
    chosen, sweep = calibrate_w_syn(
        graph, INPUT_IDX, READOUT_IDX, drive_mv=0.06,
        candidates=(0.275, 2.2, 3.2, 8.8), steps=300,
    )

    assert chosen == 3.2
    assert [p.w_syn_mv for p in sweep] == pytest.approx([0.275, 2.2, 3.2])


def test_the_default_grid_resolves_the_operating_window():
    """A doubling grid steps over it.

    Between "the readout never fires" and "the network saturates" there is only
    a narrow band of w_syn -- measured at roughly 3.0 to 4.0 mV on a
    matched-size synthetic graph, where 2.2 left the readout dead and 4.4 was
    already past the 25% activity ceiling. Consecutive candidates must be close
    enough that the sweep cannot jump the band.
    """
    ratios = [
        b / a for a, b in zip(W_SYN_CANDIDATES, W_SYN_CANDIDATES[1:])
    ]

    assert W_SYN_CANDIDATES[0] == 0.275, "the grid must start at the published value"
    assert max(ratios) <= 1.6


def test_calibration_refuses_a_scale_that_makes_the_network_run_away():
    """An arm where everything fires is not transmitting, it is saturating."""
    with pytest.raises(ValueError, match="no candidate w_syn"):
        calibrate_w_syn(
            _big_graph(), INPUT_IDX, READOUT_IDX, drive_mv=0.06,
            max_active_fraction=0.0, steps=300,
        )


def test_calibration_raises_instead_of_returning_a_best_effort_value():
    """A substrate that fails the criterion must stop the run, not proceed.

    The pilot's whole failure was a substrate that transmitted nothing and was
    run past anyway, producing three byte-identical arms.
    """
    with pytest.raises(ValueError, match="no candidate w_syn"):
        calibrate_w_syn(
            _big_graph(), INPUT_IDX, READOUT_IDX, drive_mv=0.06,
            candidates=(1e-6, 1e-5), steps=300,
        )


def test_the_error_names_what_each_candidate_actually_did():
    """The sweep table is the diagnostic; without it the failure is unreadable."""
    with pytest.raises(ValueError) as caught:
        calibrate_w_syn(
            _big_graph(), INPUT_IDX, READOUT_IDX, drive_mv=0.06,
            candidates=(1e-6,), steps=300,
        )

    message = str(caught.value)
    assert "w_syn= 0.000" in message
    assert "readout_live=" in message
