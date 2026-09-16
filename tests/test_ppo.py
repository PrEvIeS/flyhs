"""Tests for the PPO trainer.

The load-bearing test here is ``test_recomputed_log_prob_matches_the_stored_one``.
PPO's ratio is ``exp(new_logp - old_logp)``, so at epoch 0, before any update,
it must be exactly 1. If the stored masks or action codes are misaligned the
ratio starts wrong, the clipped objective optimises something other than the
policy that generated the data, and nothing about the failure is visible except
poor results.
"""

import numpy as np
import pytest
import torch

from flywire_rl.minicard import (
    MiniCard,
    board_control_opponent,
    greedy_face_opponent,
)
from flywire_rl.policy import SpikingPolicy
from flywire_rl.ppo import RolloutBuffer, collect_rollout, compute_gae, ppo_update


def _policy(n=40, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    pre = rng.integers(0, n, size=200)
    post = rng.integers(0, n, size=200)
    keep = pre != post
    return SpikingPolicy(
        n_neurons=n,
        pre_idx=torch.from_numpy(pre[keep]),
        post_idx=torch.from_numpy(post[keep]),
        weights=torch.ones(int(keep.sum())) * 200.0,
        input_indices=torch.arange(0, 12),
        output_indices=torch.arange(n - 12, n),
        rank=4,
        steps=60,
        input_drive_mv=0.06,
    )


def _make_game(seed):
    game = MiniCard(seed=seed, max_turns=8)
    game.reset()
    return game


# -------------------------------------------------------------------- GAE


def test_gae_matches_a_hand_computed_example():
    # One episode, three steps, gamma = 0.5, lambda = 1 so GAE is the plain
    # discounted-return advantage.
    rewards = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    values = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    dones = np.array([False, False, True])

    advantages, returns = compute_gae(rewards, values, dones, gamma=0.5, lam=1.0)

    np.testing.assert_allclose(returns, [0.25, 0.5, 1.0], rtol=1e-6)
    np.testing.assert_allclose(advantages, [0.25, 0.5, 1.0], rtol=1e-6)


def test_gae_subtracts_the_value_baseline():
    rewards = np.array([0.0, 1.0], dtype=np.float32)
    values = np.array([0.3, 0.7], dtype=np.float32)
    dones = np.array([False, True])

    advantages, returns = compute_gae(rewards, values, dones, gamma=1.0, lam=1.0)

    np.testing.assert_allclose(returns - values, advantages, rtol=1e-5)


def test_gae_does_not_bootstrap_across_an_episode_boundary():
    rewards = np.array([1.0, 0.0], dtype=np.float32)
    values = np.array([0.0, 0.0], dtype=np.float32)
    dones = np.array([True, True])

    _, returns = compute_gae(rewards, values, dones, gamma=0.9, lam=1.0)

    # The first return must not see the second episode's reward.
    np.testing.assert_allclose(returns, [1.0, 0.0], rtol=1e-6)


# ---------------------------------------------------------------- rollout


def test_rollout_has_consistent_lengths():
    policy = _policy()
    buffer = collect_rollout(policy, _make_game, board_control_opponent, n_steps=25)

    length = len(buffer.rewards)
    assert length == 25
    for field in (
        buffer.observations,
        buffer.kind_codes,
        buffer.source_codes,
        buffer.target_codes,
        buffer.kind_masks,
        buffer.source_masks,
        buffer.target_masks,
        buffer.log_probs,
        buffer.values,
        buffer.dones,
    ):
        assert len(field) == length


def test_reward_is_terminal_only_by_default():
    policy = _policy()
    buffer = collect_rollout(policy, _make_game, board_control_opponent, n_steps=60)

    non_zero = np.flatnonzero(buffer.rewards)
    assert set(non_zero.tolist()) <= set(np.flatnonzero(buffer.dones).tolist())


def test_terminal_rewards_are_plus_minus_one_or_zero():
    policy = _policy()
    buffer = collect_rollout(policy, _make_game, board_control_opponent, n_steps=80)

    assert set(np.unique(buffer.rewards[buffer.dones]).tolist()) <= {-1.0, 0.0, 1.0}


def test_seats_alternate_across_episodes():
    policy = _policy()
    buffer = collect_rollout(policy, _make_game, board_control_opponent, n_steps=150)

    assert len(set(buffer.episode_seats)) == 2, "the agent must play both seats"


def test_opponents_alternate_and_pair_with_both_seats():
    """Every seat-by-opponent combination must occur, or the score is partly a
    measure of which bot the agent happened to face from which chair."""
    policy = _policy()
    buffer = collect_rollout(
        policy,
        _make_game,
        [board_control_opponent, greedy_face_opponent],
        n_steps=300,
    )

    combos = set(zip(buffer.episode_seats, buffer.episode_opponents))
    assert combos == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_a_single_opponent_is_still_accepted():
    policy = _policy()
    buffer = collect_rollout(policy, _make_game, board_control_opponent, n_steps=60)

    assert set(buffer.episode_opponents) == {0}


def test_rollout_records_at_least_one_completed_episode():
    policy = _policy()
    buffer = collect_rollout(policy, _make_game, board_control_opponent, n_steps=120)

    assert buffer.dones.sum() >= 1
    assert len(buffer.episode_returns) == int(buffer.dones.sum())


# ----------------------------------------------------------------- update


def test_recomputed_log_prob_matches_the_stored_one():
    """PPO's ratio must be exactly 1 before the first gradient step."""
    policy = _policy()
    buffer = collect_rollout(policy, _make_game, board_control_opponent, n_steps=20)

    with torch.no_grad():
        log_probs, _, _ = policy.evaluate_actions(
            torch.from_numpy(buffer.observations),
            torch.from_numpy(buffer.kind_codes),
            torch.from_numpy(buffer.source_codes),
            torch.from_numpy(buffer.target_codes),
            torch.from_numpy(buffer.kind_masks),
            torch.from_numpy(buffer.source_masks),
            torch.from_numpy(buffer.target_masks),
        )

    torch.testing.assert_close(
        log_probs, torch.from_numpy(buffer.log_probs), rtol=1e-4, atol=1e-5
    )


def test_update_changes_the_trainable_parameters():
    policy = _policy()
    buffer = collect_rollout(policy, _make_game, board_control_opponent, n_steps=40)
    optimiser = torch.optim.AdamW(policy.parameters(), lr=1e-2)

    before = policy.coupling.A.detach().clone()
    ppo_update(policy, buffer, optimiser, epochs=2, minibatch=16)

    assert not torch.equal(before, policy.coupling.A.detach())


def test_update_leaves_the_frozen_connectome_untouched():
    """Only the adapter and the interface may move; the wiring is evidence."""
    policy = _policy()
    buffer = collect_rollout(policy, _make_game, board_control_opponent, n_steps=40)
    optimiser = torch.optim.AdamW(policy.parameters(), lr=1e-2)

    before = policy.coupling.weights.detach().clone()
    ppo_update(policy, buffer, optimiser, epochs=2, minibatch=16)

    torch.testing.assert_close(policy.coupling.weights.detach(), before)


def test_update_reports_its_loss_components():
    policy = _policy()
    buffer = collect_rollout(policy, _make_game, board_control_opponent, n_steps=40)
    optimiser = torch.optim.AdamW(policy.parameters(), lr=1e-3)

    stats = ppo_update(policy, buffer, optimiser, epochs=1, minibatch=16)

    assert {"policy_loss", "value_loss", "entropy", "clip_fraction"} <= set(stats)
    assert np.isfinite(stats["policy_loss"])


def test_buffer_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="length"):
        RolloutBuffer(
            observations=np.zeros((3, 4), dtype=np.float32),
            kind_codes=np.zeros(2, dtype=np.int64),
            source_codes=np.zeros(3, dtype=np.int64),
            target_codes=np.zeros(3, dtype=np.int64),
            kind_masks=np.zeros((3, 3), dtype=bool),
            source_masks=np.zeros((3, 13), dtype=bool),
            target_masks=np.zeros((3, 12), dtype=bool),
            log_probs=np.zeros(3, dtype=np.float32),
            values=np.zeros(3, dtype=np.float32),
            rewards=np.zeros(3, dtype=np.float32),
            dones=np.zeros(3, dtype=bool),
            episode_returns=[],
            episode_seats=[],
        )
