"""Tests for the experiment harness.

Everything here guards the same property: arms must differ in their edges and in
nothing else. Any other difference — a node index, an adapter shape, a budget —
is a second explanation for whatever the experiment measures.
"""

import json

import numpy as np
import pytest
import torch

from flywire_rl.controls import Graph, branching_ratio
from flywire_rl.experiment import (
    ExperimentResult,
    build_arms,
    evaluate_policy,
    load_result,
    make_policy,
    save_result,
)
from flywire_rl.minicard import board_control_opponent, greedy_face_opponent


def _graph(n=400, m=4000, seed=0):
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
        sign=rng.choice([1, -1], size=n, p=[0.78, 0.22]).astype(np.int8),
    )


# ------------------------------------------------------------------- arms


def test_build_arms_returns_the_three_required_arms():
    arms = build_arms(_graph(), seed=0, swaps_per_edge=5)

    assert set(arms) == {"real", "shuffled", "random"}


def test_arms_share_node_and_edge_counts():
    arms = build_arms(_graph(), seed=0, swaps_per_edge=5)

    assert len({a.n for a in arms.values()}) == 1
    assert len({a.n_edges for a in arms.values()}) == 1


def test_arms_share_a_branching_ratio():
    """Amendment A3: arms are matched on transmission, not spectral radius.

    Each arm gets its own weight scale so that all of them carry one spike per
    spike. Matching the scale instead would leave them transmitting at
    different rates, a larger difference than the topology under test.
    """
    arms = build_arms(_graph(), seed=0, swaps_per_edge=5, target_branching=1.0)

    for arm in arms.values():
        assert branching_ratio(arm) == pytest.approx(1.0, abs=0.2)


def test_arms_receive_different_weight_scales():
    """Equal transmission generally requires unequal scaling: that is the point."""
    original = _graph()
    arms = build_arms(original, seed=0, swaps_per_edge=5, target_branching=1.0)
    scales = {
        name: float(arm.weight.sum() / original.weight.sum())
        for name, arm in arms.items()
    }

    assert len(set(round(v, 6) for v in scales.values())) > 1


def test_arms_preserve_the_sign_vector_exactly():
    original = _graph()
    arms = build_arms(original, seed=0, swaps_per_edge=5)

    for arm in arms.values():
        assert np.array_equal(arm.sign, original.sign)


def test_build_arms_is_deterministic_given_a_seed():
    first = build_arms(_graph(), seed=3, swaps_per_edge=5)
    second = build_arms(_graph(), seed=3, swaps_per_edge=5)

    assert np.array_equal(first["shuffled"].post, second["shuffled"].post)
    assert np.array_equal(first["random"].pre, second["random"].pre)


# ----------------------------------------------------------------- policy


def test_policies_have_identical_trainable_capacity_across_arms():
    arms = build_arms(_graph(), seed=0, swaps_per_edge=5)
    inputs, outputs = torch.arange(0, 12), torch.arange(388, 400)

    counts = {
        name: sum(
            p.numel()
            for p in make_policy(
                arm, inputs, outputs, seed=0, rank=4, steps=5
            ).parameters()
            if p.requires_grad
        )
        for name, arm in arms.items()
    }

    assert len(set(counts.values())) == 1


def test_policies_use_the_same_interface_nodes_across_arms():
    arms = build_arms(_graph(), seed=0, swaps_per_edge=5)
    inputs, outputs = torch.arange(0, 12), torch.arange(388, 400)

    for arm in arms.values():
        policy = make_policy(arm, inputs, outputs, seed=0, rank=4, steps=5)
        assert torch.equal(policy.input_indices, inputs)
        assert torch.equal(policy.output_indices, outputs)


def test_policy_freezes_the_connectome():
    arm = build_arms(_graph(), seed=0, swaps_per_edge=5)["real"]
    policy = make_policy(arm, torch.arange(0, 12), torch.arange(388, 400), seed=0, rank=4)

    trainable = {name for name, p in policy.named_parameters() if p.requires_grad}
    assert not any(name.endswith("weights") for name in trainable)


# ------------------------------------------------------------- evaluation


def test_evaluation_returns_one_score_per_episode():
    arm = build_arms(_graph(), seed=0, swaps_per_edge=5)["real"]
    policy = make_policy(
        arm, torch.arange(0, 12), torch.arange(388, 400), seed=0, rank=4, steps=5
    )

    scores = evaluate_policy(
        policy, [board_control_opponent, greedy_face_opponent], n_episodes=6, seed=0
    )

    assert scores.shape == (6,)
    assert set(np.unique(scores).tolist()) <= {-1.0, 0.0, 1.0}


def test_evaluation_is_deterministic_given_a_seed():
    arm = build_arms(_graph(), seed=0, swaps_per_edge=5)["real"]
    policy = make_policy(
        arm, torch.arange(0, 12), torch.arange(388, 400), seed=0, rank=4, steps=5
    )

    first = evaluate_policy(policy, [board_control_opponent], n_episodes=6, seed=4)
    second = evaluate_policy(policy, [board_control_opponent], n_episodes=6, seed=4)

    assert np.array_equal(first, second)


# ------------------------------------------------------------ persistence


def test_result_round_trips_through_disk(tmp_path):
    result = ExperimentResult(
        label="pilot",
        scores={name: np.zeros((2, 3)) for name in ("real", "shuffled", "random")},
        provenance={"device": "cpu", "manifest": {"connections.csv.gz": "abc123"}},
        config={"seeds": 2, "total_steps": 10},
    )
    path = tmp_path / "result.json"
    save_result(result, path)
    restored = load_result(path)

    assert restored.label == "pilot"
    assert restored.config == result.config
    for name, scores in result.scores.items():
        assert np.array_equal(restored.scores[name], scores)


def test_saved_result_records_its_provenance(tmp_path):
    result = ExperimentResult(
        label="pilot",
        scores={"real": np.zeros((1, 2))},
        provenance={"device": "mps", "torch": "2.14.0"},
        config={},
    )
    path = tmp_path / "result.json"
    save_result(result, path)

    payload = json.loads(path.read_text())
    assert payload["provenance"]["device"] == "mps"
    assert payload["provenance"]["torch"] == "2.14.0"


def test_label_must_say_whether_a_run_is_confirmatory():
    with pytest.raises(ValueError, match="pilot|confirmatory"):
        ExperimentResult(label="whatever", scores={}, provenance={}, config={})
