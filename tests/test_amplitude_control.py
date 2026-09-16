"""Amplitude control: readout standardisation (fly-ky7.11).

The pilot's arm ordering reproduced the *pre-learning* amplitude ordering
exactly (amendment A4: median s.d. 70.1 / 13.7 / 4.88, feature RMS 1510 / 839 /
242), so the result may say "the real connectome delivers more signal" rather
than "the real connectome computes better". Standardising each arm's readout to
zero mean and unit variance removes the amplitude difference while preserving
the pattern across units, which separates the two claims.

These tests pin the properties that make that separation valid: the reference
set is shared and policy-independent, the statistics are frozen rather than
adaptive, and an uncalibrated policy is bit-identical to the pre-amendment one.
"""

import numpy as np
import pytest
import torch

from flywire_rl.minicard import MiniCard
from flywire_rl.policy import (
    OBS_DIM,
    SpikingPolicy,
    encode_observation,
    reference_observations,
)


# A standardisation test is vacuous on a silent substrate: every unit sits at
# v_rest = -52 mV, every variance is zero, and dividing by the eps floor would
# "pass". So this fixture is built to discriminate, and the tests assert that it
# does before relying on it.
#
# Two departures from the fixture in test_policy.py, both necessary:
#
# 1. Explicit input->output edges. Over a random 40-node graph there may be no
#    path from the input set to the output set at all, in which case the readout
#    never leaves rest whatever the weights.
# 2. A far larger ``input_drive_mv``. The drive reaching a neuron is
#    ``encoder(obs) * input_drive_mv``, and the encoder's random init puts
#    |encoder(obs)| around 0.4, so the 0.06 used elsewhere delivers ~0.002 mV
#    per step against the 7 mV gap from rest to threshold -- measured: zero
#    spikes anywhere in the network. 15.0 gets all 12 output units varying.
_DRIVE_MV = 15.0


def _policy(n=40, seed=0, drive_mv=_DRIVE_MV):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    pre = list(rng.integers(0, n, size=200))
    post = list(rng.integers(0, n, size=200))
    for i in range(12):
        pre.append(i)
        post.append(n - 12 + i)
    pre, post = np.array(pre), np.array(post)
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
        input_drive_mv=drive_mv,
    )


def _observations(count=24, seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(count):
        game = MiniCard(seed=seed + i)
        game.reset()
        for _ in range(int(rng.integers(0, 10))):
            legal = game.legal_actions()
            if game.is_over or not legal:
                break
            game.step(legal[int(rng.integers(0, len(legal)))])
        rows.append(encode_observation(game.observe(game.to_move)))
    return torch.from_numpy(np.stack(rows))


# --- the reference set -------------------------------------------------------


def test_reference_set_is_deterministic_and_shaped_for_the_encoder():
    first = reference_observations(16)
    second = reference_observations(16)
    assert first.shape == (16, OBS_DIM)
    assert torch.equal(first, second)


def test_reference_set_does_not_depend_on_any_policy():
    """The whole control fails if arms are standardised on different states.

    ``reference_observations`` takes no policy argument, so states cannot be
    drawn from an arm's own behaviour. This pins that as a contract, not an
    accident of the current implementation.
    """
    import inspect

    params = inspect.signature(reference_observations).parameters
    assert "policy" not in params
    assert "arm" not in params


def test_reference_states_are_distinct_positions():
    """Correlated steps of one match would standardise on a narrow slice of the
    state space; the set is collected one position per game instead."""
    states = reference_observations(32)
    unique = {tuple(row.tolist()) for row in states}
    assert len(unique) > 24, f"only {len(unique)} distinct positions of 32"


def test_reference_set_refuses_a_single_state():
    with pytest.raises(ValueError, match="at least 2"):
        reference_observations(1)


# --- calibration -------------------------------------------------------------


def test_uncalibrated_policy_is_identical_to_the_raw_readout():
    """Default-off keeps every pre-amendment result reproducible bit for bit."""
    policy = _policy()
    obs = _observations(8)
    assert not bool(policy.standardised)
    assert torch.equal(policy.features(obs), policy.raw_features(obs))


def test_calibration_gives_the_reference_set_zero_mean_and_unit_variance():
    policy = _policy()
    reference = reference_observations(16)
    policy.calibrate_readout(reference)

    with torch.no_grad():
        features = policy.features(reference)
        live = policy.raw_features(reference).std(0, unbiased=False) > 1e-6

    assert live.all(), (
        f"fixture substrate discriminates on only {int(live.sum())} of 12 units; "
        "a standardisation test on constant features would be vacuous"
    )

    # Tolerance is set by float32, not by the method. Features sit around
    # v_rest = -52 mV, so the centring subtraction at inference cancels four
    # leading digits, and a unit whose s.d. is small divides that residue by a
    # small number. The statistics themselves are accumulated in float64. This
    # is harmless downstream: the heads are linear and learn their own scale,
    # and what the control equalises between arms is the s.d., which is exact
    # to 1e-5 here.
    assert torch.allclose(features.mean(0), torch.zeros(12), atol=1e-3)
    assert torch.allclose(
        features.std(0, unbiased=False), torch.ones(12), atol=1e-5
    )


def test_calibration_removes_an_amplitude_difference_between_two_arms():
    """The point of the control, stated as a test.

    Two substrates whose readouts differ only in gain must become
    indistinguishable in amplitude once standardised -- otherwise re-running the
    pilot could not tell transmission from computation.
    """
    reference = reference_observations(16)
    # Same wiring and same seed; only the input gain differs. Any amplitude
    # gap between them is therefore transmission, never topology.
    loud = _policy(seed=0, drive_mv=_DRIVE_MV)
    quiet = _policy(seed=0, drive_mv=_DRIVE_MV / 8.0)

    def spread(policy):
        with torch.no_grad():
            features = policy.raw_features(reference)
        return float(features.std(0, unbiased=False).median())

    raw_gap = abs(spread(loud) - spread(quiet))
    assert raw_gap > 1e-3, "fixture produced no amplitude difference to remove"

    loud.calibrate_readout(reference)
    quiet.calibrate_readout(reference)

    def spread_std(policy):
        with torch.no_grad():
            features = policy.features(reference)
        return float(features.std(0, unbiased=False).median())

    # Both are now unit-variance by construction, so the gap must vanish.
    assert abs(spread_std(loud) - spread_std(quiet)) < 1e-4
    assert abs(spread_std(loud) - 1.0) < 1e-4


def test_statistics_are_frozen_not_adaptive():
    """A running normaliser would keep adapting per arm during training and so
    would smuggle back the arm-dependent signal this control exists to remove."""
    policy = _policy()
    policy.calibrate_readout(reference_observations(16))
    mean_before = policy.readout_mean.clone()
    scale_before = policy.readout_scale.clone()

    for _ in range(3):
        policy.features(_observations(8))

    assert torch.equal(policy.readout_mean, mean_before)
    assert torch.equal(policy.readout_scale, scale_before)


def test_statistics_never_receive_a_gradient():
    policy = _policy()
    policy.calibrate_readout(reference_observations(16))
    assert not policy.readout_mean.requires_grad
    assert not policy.readout_scale.requires_grad
    names = {n for n, _ in policy.named_parameters()}
    assert "readout_mean" not in names and "readout_scale" not in names


def test_calibration_survives_a_state_dict_round_trip():
    """A reloaded policy that forgot it was calibrated would silently mix a
    standardised arm with unstandardised ones inside one comparison."""
    source = _policy()
    source.calibrate_readout(reference_observations(16))

    target = _policy()
    target.load_state_dict(source.state_dict())

    assert bool(target.standardised)
    obs = _observations(6)
    assert torch.allclose(target.features(obs), source.features(obs), atol=1e-6)


def test_a_dead_unit_is_passed_through_rather_than_amplified():
    """Dividing a constant unit's zero deviation by eps would turn float noise
    into a large feature. It keeps scale 1.0 and stays at zero instead."""
    policy = _policy()
    reference = reference_observations(16)
    policy.calibrate_readout(reference)
    # Force a constant unit and recalibrate through the public path.
    raw = policy.raw_features(reference)
    policy.standardised.fill_(False)
    stats = policy.calibrate_readout(reference)

    std = raw.std(0, unbiased=False)
    dead = std <= 1e-6
    assert stats["dead_units"] == float(int(dead.sum()))
    assert torch.all(policy.readout_scale[dead] == 1.0)
    assert torch.all(policy.readout_scale > 0), "a zero scale would produce inf/NaN"


def test_calibration_rejects_a_degenerate_reference_set():
    policy = _policy()
    with pytest.raises(ValueError, match="at least 2"):
        policy.calibrate_readout(reference_observations(16)[:1])
    with pytest.raises(ValueError, match="n_states, obs_dim"):
        policy.calibrate_readout(reference_observations(16)[0])


def test_a_failed_calibration_does_not_leave_the_flag_flipped():
    policy = _policy()
    policy.calibrate_readout(reference_observations(16))
    # Wrong observation width survives the shape guards and fails inside the
    # encoder, so this is torch's error, not ours -- which is exactly the case
    # worth pinning: the flag must be restored on an exception we did not raise.
    with pytest.raises(RuntimeError):
        policy.calibrate_readout(torch.zeros(4, OBS_DIM + 1))
    assert bool(policy.standardised), "an aborted recalibration disabled a live control"


def test_standardised_policy_still_produces_legal_actions_and_gradients():
    """The control must not break the interface it sits inside."""
    policy = _policy()
    policy.calibrate_readout(reference_observations(16))

    game = MiniCard(seed=11)
    game.reset()
    chosen, log_prob, value = policy.act(game)
    assert chosen in game.legal_actions()
    assert torch.isfinite(log_prob) and torch.isfinite(value)

    (log_prob + value).backward()
    grads = [
        p.grad for p in policy.parameters() if p.requires_grad and p.grad is not None
    ]
    assert grads, "no parameter received a gradient through the standardised readout"
    assert all(torch.isfinite(g).all() for g in grads)
