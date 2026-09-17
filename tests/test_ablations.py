"""The rows that decide whether a score is evidence of anything.

A policy that scores the same with its observation removed is not using it.
liuzihe02/fly-craftax found exactly that in their own trained readout and
described it as "a clock with a small drive-dependent jitter, not a visually
guided policy" -- and they only found it because they replayed the policy with
vision blacked out. An arm ordering between clocks is not a topology result, so
these controls are a precondition for reporting any comparison, not an extra.

cobanov/flyjump's table is the shape to match: the same trained weights with
the circuit silenced, and the untrained readout on a live circuit, both
collapsing to the floor while the intact policy does not.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from flywire_rl.controls import Graph
from flywire_rl.experiment import (
    evaluate_policy,
    evaluate_random,
    make_policy,
    run_experiment,
)

N = 120
M = 1200
INPUTS = torch.arange(0, 10)
OUTPUTS = torch.arange(N - 6, N)
#: A window in which this fixture's substrate is actually live.
#:
#: Measured while writing these tests: at 5 steps -- a 0.5 ms window -- the
#: 120-neuron fixture returns v_rest for every observation, so blanking the
#: input and silencing the substrate produce identical features and the tests
#: below pass for the wrong reason. At 60 steps the features differ between
#: observations and a blank observation still moves the membrane, which is the
#: regime the controls are meant to distinguish.
LIVE_STEPS = 60

#: Plumbing only: run_experiment's wiring does not need a live substrate, and
#: five steps keeps the end-to-end tests fast.
BUDGET = dict(
    n_seeds=1,
    total_steps=30,
    n_eval_episodes=3,
    rank=2,
    steps=5,
    rollout_steps=15,
    minibatch=5,
    epochs=1,
)
RUNGS = {"untrained", "silent", "blank", "random"}


def _graph(seed: int = 0) -> Graph:
    rng = np.random.default_rng(seed)
    seen: set[tuple[int, int]] = set()
    while len(seen) < M:
        a, b = int(rng.integers(N)), int(rng.integers(N))
        if a != b:
            seen.add((a, b))
    pre, post = map(np.array, zip(*sorted(seen)))
    return Graph(
        n=N,
        pre=pre.astype(np.int64),
        post=post.astype(np.int64),
        weight=rng.gamma(2.0, 3.0, size=M).astype(np.float32),
        sign=rng.choice([1, -1], size=N, p=[0.78, 0.22]).astype(np.int8),
    )


def _policy(steps: int = LIVE_STEPS):
    return make_policy(_graph(), INPUTS, OUTPUTS, seed=0, rank=2, steps=steps)


def _two_observations(policy):
    """Two observations that a live substrate must map to different features."""
    rng = np.random.default_rng(0)
    width = policy.encoder.in_features
    return (
        torch.from_numpy(rng.random((1, width)).astype(np.float32)),
        torch.from_numpy(rng.random((1, width)).astype(np.float32)),
    )


# --- the intact policy must actually depend on its input ------------------


def test_the_substrate_maps_different_observations_to_different_features():
    """The premise of every row below. If this fails, nothing else means much."""
    policy = _policy()
    first, second = _two_observations(policy)

    with torch.no_grad():
        a = policy.raw_features(first)
        b = policy.raw_features(second)

    assert not torch.allclose(a, b)


# --- silencing the substrate ----------------------------------------------


def test_silencing_returns_the_resting_constant():
    policy = _policy()
    policy.ablation = "silent"
    first, second = _two_observations(policy)

    with torch.no_grad():
        a = policy.raw_features(first)
        b = policy.raw_features(second)

    assert a.shape == (1, len(OUTPUTS))
    assert torch.equal(a, b)
    assert torch.all(a == policy.params.v_rest_mv)


def test_silencing_does_not_run_the_substrate():
    """Cheap by construction: the row is meant to be free, not merely quiet."""
    policy = _policy()
    calls = []
    original = policy.substrate.rollout
    policy.substrate.rollout = lambda *a, **k: calls.append(1) or original(*a, **k)

    policy.ablation = "silent"
    with torch.no_grad():
        policy.raw_features(_two_observations(policy)[0])

    assert calls == []


# --- blanking the observation ---------------------------------------------


def test_blanking_makes_the_features_independent_of_the_game():
    """The substrate still runs; it just learns nothing about the board."""
    from flywire_rl.minicard import MiniCard

    policy = _policy()
    policy.ablation = "blank"

    first, second = MiniCard(seed=1), MiniCard(seed=2)
    first.reset()
    second.reset()
    _, a = policy.act_with_record(first)
    _, b = policy.act_with_record(second)

    # Different games, identical observation after blanking, so the value the
    # critic assigns can differ only through the action masks, never through
    # the state.
    assert np.allclose(a["value"], b["value"])


def test_blanking_still_drives_the_substrate():
    """Distinguishes 'needs the substrate' from 'needs to see the board'."""
    policy = _policy()
    policy.ablation = "blank"
    blank = torch.zeros((1, policy.encoder.in_features))

    with torch.no_grad():
        features = policy.raw_features(blank)

    assert not torch.all(features == policy.params.v_rest_mv)


# --- the switch must not leak ---------------------------------------------


def test_evaluate_policy_restores_the_switch():
    policy = _policy()
    evaluate_policy(policy, n_episodes=2, seed=0, ablation="silent")

    assert policy.ablation is None


def test_the_switch_is_restored_even_when_evaluation_raises():
    """A policy left ablated would poison every later evaluation in the run."""
    policy = _policy()

    def explode(_game):
        raise RuntimeError("simulated failure")

    policy.act_with_record = explode
    with pytest.raises(RuntimeError):
        evaluate_policy(policy, n_episodes=2, seed=0, ablation="blank")

    assert policy.ablation is None


def test_the_ablation_switch_does_not_travel_in_the_state_dict():
    policy = _policy()
    policy.ablation = "silent"

    assert not any("ablation" in key for key in policy.state_dict())


# --- the random floor ------------------------------------------------------


def test_random_play_returns_legal_terminal_scores():
    scores = evaluate_random(n_episodes=4, seed=0)

    assert scores.shape == (4,)
    assert set(np.unique(scores)) <= {-1.0, 0.0, 1.0}


def test_random_play_is_seed_deterministic():
    assert np.array_equal(
        evaluate_random(n_episodes=4, seed=3), evaluate_random(n_episodes=4, seed=3)
    )


# --- the table reaches the result -----------------------------------------


def test_the_rungs_reach_the_result_provenance():
    result = run_experiment(
        _graph(), INPUTS, OUTPUTS, label="pilot", ablations=True, **BUDGET
    )

    rungs = result.provenance["ablations"]
    assert set(rungs) == {"real", "shuffled", "random"}
    for arm, per_rung in rungs.items():
        assert set(per_rung) == RUNGS, arm
        for name, per_seed in per_rung.items():
            assert len(per_seed) == BUDGET["n_seeds"], (arm, name)
            assert len(per_seed[0]) == BUDGET["n_eval_episodes"], (arm, name)


def test_the_rungs_are_off_by_default():
    """Four extra evaluations per arm and seed is not a silent default."""
    result = run_experiment(_graph(), INPUTS, OUTPUTS, label="pilot", **BUDGET)

    assert result.provenance["ablations"] == {}
    assert result.config["ablations"] is False


def test_the_result_records_that_the_rungs_were_run():
    result = run_experiment(
        _graph(), INPUTS, OUTPUTS, label="pilot", ablations=True, **BUDGET
    )

    assert result.config["ablations"] is True
