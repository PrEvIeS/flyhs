"""Tests for the observation encoder and the hierarchical action decoder.

This is the joint between the environment and the substrate. Two properties
matter most: a sampled action must always be legal (so invalid-action rate is
structurally zero rather than penalised), and the hidden-information boundary
must be honoured exactly, or Stage 3 silently becomes Stage 2.
"""

import numpy as np
import pytest
import torch

from flywire_rl.minicard import ActionKind, MiniCard, TargetKind, action
from flywire_rl.policy import (
    OBS_DIM,
    ActionCodec,
    SpikingPolicy,
    encode_observation,
    select_io_indices,
)


def _game(seed=0):
    game = MiniCard(seed=seed)
    game.reset()
    return game


def _rich_state(seed=0):
    """A state with several legal options at every level.

    On turn 1 a player often has one mana and nothing affordable, leaving
    END_TURN as the only legal action. That is correct behaviour but it makes a
    poor fixture for testing masks and gradients.
    """
    game = _game(seed)
    game.current.mana = 10
    game.current.max_mana = 10
    for _ in range(2):
        minion = game.summon_for_test(game.current, "Scout")
        minion.sick = False
    game.summon_for_test(game.opponent, "Knight")
    return game


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
        # Weights are synapse counts. A single presynaptic spike displaces the
        # postsynaptic membrane by 0.1575 * count * w_syn (the peak of the exact
        # two-variable solution, at t = 9.24 ms), against the 7 mV gap from rest
        # to threshold. At the published w_syn = 0.275 mV that needs about 162
        # synapses on one edge, so this toy graph uses 200 to make the readout
        # reachable in a single hop rather than by coincidence.
        weights=torch.ones(int(keep.sum())) * 200.0,
        input_indices=torch.arange(0, 12),
        output_indices=torch.arange(n - 12, n),
        rank=4,
        steps=300,
        # Tonic drive in mV per step. It settles at drive / (1 - exp(-dt/t_mbr))
        # above rest, so 0.06 mV/step reaches about 12 mV against the 7 mV gap
        # and the input neurons fire. Without that the substrate is silent,
        # every feature sits at v_rest, and the heads get no weight gradient.
        input_drive_mv=0.06,
    )


# ------------------------------------------------------------- observation


def test_observation_has_a_fixed_width():
    game = _game()
    vector = encode_observation(game.observe(0, hide_opponent_hand=True))

    assert vector.shape == (OBS_DIM,)
    assert vector.dtype == np.float32


def test_observation_is_deterministic():
    game = _game()
    observation = game.observe(0, hide_opponent_hand=True)

    assert np.array_equal(
        encode_observation(observation), encode_observation(observation)
    )


def test_hidden_hand_is_zeroed_and_flagged():
    game = _game()
    game.step(action(ActionKind.END_TURN))  # give the opponent a hand

    visible = encode_observation(game.observe(0, hide_opponent_hand=False))
    hidden = encode_observation(game.observe(0, hide_opponent_hand=True))
    block = slice(*ActionCodec.OPPONENT_HAND_SLICE)

    assert np.any(visible[block] != 0)
    assert np.all(hidden[block] == 0)
    # The flag is what lets the network tell "hidden" from "genuinely empty".
    assert visible[ActionCodec.HAND_VISIBLE_INDEX] == 1.0
    assert hidden[ActionCodec.HAND_VISIBLE_INDEX] == 0.0


def test_hand_size_survives_hiding():
    game = _game()
    game.step(action(ActionKind.END_TURN))
    hidden = encode_observation(game.observe(0, hide_opponent_hand=True))

    assert hidden[ActionCodec.OPPONENT_HAND_SIZE_INDEX] > 0


def test_observation_values_are_bounded():
    game = _game()
    for _ in range(12):
        if game.is_over:
            break
        game.step(game.legal_actions()[0])
        vector = encode_observation(game.observe(0, hide_opponent_hand=True))
        assert np.all(np.abs(vector) <= 2.0)


# -------------------------------------------------------------------- codec


def test_codec_round_trips_every_legal_action():
    game = _game()
    for _ in range(40):
        if game.is_over:
            break
        for candidate in game.legal_actions():
            levels = ActionCodec.encode(candidate)
            assert ActionCodec.decode(*levels) == candidate
        game.step(game.legal_actions()[-1])


def test_level_one_mask_matches_the_legal_kinds():
    game = _game()
    mask = ActionCodec.kind_mask(game.legal_actions())
    expected = {a.kind for a in game.legal_actions()}

    for kind in ActionKind:
        assert bool(mask[int(kind)]) == (kind in expected)


def test_level_two_mask_is_conditioned_on_the_chosen_kind():
    game = _game()
    legal = game.legal_actions()
    mask = ActionCodec.source_mask(legal, ActionKind.PLAY_CARD)
    expected = {
        ActionCodec.encode(a)[1] for a in legal if a.kind is ActionKind.PLAY_CARD
    }

    assert {int(i) for i in np.flatnonzero(mask)} == expected


def test_level_three_mask_is_conditioned_on_kind_and_source():
    game = _rich_state()
    legal = game.legal_actions()
    plays = [a for a in legal if a.kind is ActionKind.PLAY_CARD]
    source = ActionCodec.encode(plays[0])[1]

    mask = ActionCodec.target_mask(legal, ActionKind.PLAY_CARD, source)
    expected = {
        ActionCodec.encode(a)[2] for a in plays if ActionCodec.encode(a)[1] == source
    }

    assert {int(i) for i in np.flatnonzero(mask)} == expected


def test_end_turn_has_no_source_or_target():
    levels = ActionCodec.encode(action(ActionKind.END_TURN))

    assert levels == (int(ActionKind.END_TURN), ActionCodec.NO_SLOT, 0)


def test_target_codes_distinguish_the_two_sides():
    enemy = ActionCodec.encode(
        action(ActionKind.ATTACK, 0, TargetKind.ENEMY_MINION, 2)
    )[2]
    own = ActionCodec.encode(action(ActionKind.ATTACK, 0, TargetKind.OWN_MINION, 2))[2]

    assert enemy != own


# ------------------------------------------------------------------ policy


def test_policy_samples_only_legal_actions():
    policy = _policy()
    torch.manual_seed(0)
    for seed in range(5):
        game = _game(seed)
        steps = 0
        while not game.is_over and steps < 60:
            chosen, _, _ = policy.act(game)
            assert chosen in game.legal_actions()
            game.step(chosen)
            steps += 1


def test_policy_returns_a_value_estimate():
    policy = _policy()
    game = _game()
    _, _, value = policy.act(game)

    assert value.shape == ()


@pytest.mark.xfail(reason="fly-4s8: a constant drive fires the population in lockstep, and the 1.8 ms axonal delay lands inside the 2.2 ms refractory period, so every spike is dropped on arrival. Needs the Poisson input the reference model uses, not a tonic current.", strict=True)
def test_gradient_reaches_encoder_adapter_and_heads():
    policy = _policy()
    game = _rich_state()
    observation = encode_observation(game.observe(0, hide_opponent_hand=True))
    # Stated explicitly: a silent substrate zeroes the head weight gradients
    # while leaving the bias gradients intact, which is a confusing way to
    # discover that the drive is too low. Features are membrane potentials in
    # mV and rest at v_rest = -52, so the test is that they have *moved*, not
    # that they are positive -- a sum above zero would mean the readout sat
    # above threshold throughout.
    features = policy.features(torch.from_numpy(observation).unsqueeze(0))
    assert (features != policy.params.v_rest_mv).any(), "the substrate is silent"

    _, log_prob, value = policy.act(game)
    (log_prob + value).backward()

    assert policy.encoder.weight.grad.abs().sum() > 0
    assert policy.head_kind.weight.grad.abs().sum() > 0
    assert policy.head_value.weight.grad.abs().sum() > 0
    assert policy.coupling.A.grad is not None


def test_a_forced_choice_produces_no_policy_gradient():
    """Correct, not a defect: one legal option is a degenerate distribution.

    With a single admissible action the masked softmax is identically 1, so its
    log-probability is a constant 0 and carries no gradient. Nothing is being
    learned at such a state because nothing could have been chosen differently.
    """
    policy = _policy()
    game = _game()
    game.current.mana = 0  # nothing affordable, no minions: only END_TURN
    assert len(game.legal_actions()) == 1

    _, log_prob, _ = policy.act(game)

    assert log_prob.item() == 0.0


def test_trainable_parameter_count_is_identical_for_any_topology():
    """P3: arms may differ in edges, never in trainable capacity."""
    counts = []
    for seed in (0, 1):
        policy = _policy(seed=seed)
        counts.append(sum(p.numel() for p in policy.parameters() if p.requires_grad))

    assert counts[0] == counts[1]


@pytest.mark.skipif(
    not torch.backends.mps.is_available() and not torch.cuda.is_available(),
    reason="no accelerator on this host",
)
def test_act_works_when_the_policy_has_been_moved_off_cpu():
    """Regression: act() built the observation tensor on CPU regardless of the
    module's device, which crashed the first real end-to-end run."""
    device = "mps" if torch.backends.mps.is_available() else "cuda"
    policy = _policy().to(device)
    game = _rich_state()

    chosen, _, _ = policy.act(game)

    assert chosen in game.legal_actions()


def test_io_selection_splits_afferent_from_efferent_classes():
    labels = ["sensory", "central", "descending", "optic", "ascending", "motor", None]
    inputs, outputs = select_io_indices(labels)

    assert set(inputs.tolist()) == {0, 4}
    assert set(outputs.tolist()) == {2, 5}


def test_io_selection_refuses_a_graph_with_no_output_neurons():
    with pytest.raises(ValueError, match="no input or no output"):
        select_io_indices(["sensory", "central", "central"])


def test_log_prob_sums_the_three_levels():
    policy = _policy()
    game = _game()
    torch.manual_seed(3)
    _, log_prob, _ = policy.act(game)

    assert log_prob.item() <= 0.0
    assert torch.isfinite(log_prob)
