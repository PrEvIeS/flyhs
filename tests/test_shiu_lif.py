"""Tests for the Shiu et al. (2024) absolute LIF parameterisation.

These pin the physics that the pilot got wrong. The pilot normalised the graph
to a spectral radius of 0.95 and then to a branching ratio of 1, both of which
are criteria for a *linear rate* network. On a hard-threshold LIF they left one
presynaptic spike moving the membrane by 6.6e-4 of the threshold gap, so the
recurrent graph transmitted nothing and all three arms produced byte-identical
results.

The fix is not another dynamical target: it is the absolute biophysical scale
that Shiu et al. fit once, ``w_syn = 0.275 mV`` per anatomical synapse against a
``v_th - v_rest = 7 mV`` gap. ``test_one_connection_is_a_substantial_fraction_
of_threshold`` is the regression test for the blocker itself.

Reference: Shiu, P.K. et al., Nature 634, 210-219 (2024); implementation
``philshiu/Drosophila_brain_model`` (MIT), ``model.py``. The discrete-time
semantics below were verified against Brian2 by ``liuzihe02/fly-craftax``.
"""

import math

import pytest
import torch

from flywire_rl.spiking import (
    CouplingMode,
    LIFSubstrate,
    LowRankCoupling,
    ShiuParams,
)


def _substrate(pre, post, signed_counts, n, params=None, **kwargs):
    coupling = LowRankCoupling(
        n,
        torch.tensor(pre, dtype=torch.long),
        torch.tensor(post, dtype=torch.long),
        torch.tensor(signed_counts, dtype=torch.float32),
        rank=2,
        mode=CouplingMode.GATHER,
    )
    return LIFSubstrate(coupling, params=params or ShiuParams(**kwargs))


def _silent_drive(steps, n, batch=1):
    return torch.zeros(steps, batch, n)


# --------------------------------------------------------------------------
# Parameters


def test_parameters_match_the_published_model():
    p = ShiuParams()

    assert p.v_rest_mv == -52.0
    assert p.v_reset_mv == -52.0
    assert p.v_threshold_mv == -45.0
    assert p.t_mbr_ms == 20.0
    assert p.tau_syn_ms == 5.0
    assert p.t_refractory_ms == 2.2
    assert p.t_delay_ms == 1.8
    assert p.w_syn_mv == 0.275
    # The quantity the whole calibration is measured against.
    assert p.threshold_gap_mv == pytest.approx(7.0)


def test_decay_coefficients_are_the_exact_linear_solution():
    """a, b, c solve dv/dt = (g - (v - v_rest))/t_mbr and dg/dt = -g/tau exactly.

    Explicit Euler would be wrong here by construction: the pilot's substrate
    pushed synaptic input through a ``dt/tau`` gain, which attenuated it ~20x.
    """
    p = ShiuParams(dt_ms=0.1)
    a, b, c = p.decay_coefficients()

    assert a == pytest.approx(math.exp(-0.1 / 5.0))
    assert b == pytest.approx(math.exp(-0.1 / 20.0))
    assert c == pytest.approx((b - a) * 5.0 / (20.0 - 5.0))


def test_step_counts_round_the_published_durations():
    p = ShiuParams(dt_ms=0.1)
    assert p.n_delay_steps == 18
    assert p.n_refractory_steps == 22


# --------------------------------------------------------------------------
# The blocker


def test_one_connection_is_a_substantial_fraction_of_threshold():
    """A 25-synapse connection must move the postsynaptic membrane appreciably.

    This is the pilot's failure expressed as a number. Under rho = 0.95 one
    spike moved the membrane by 6.6e-4 of the gap, which is why activity never
    left the directly driven input neurons. Under the published scale a
    25-synapse connection deposits 25 * 0.275 = 6.875 mV into g against a 7 mV
    gap, and the membrane must see a double-digit percentage of that.
    """
    p = ShiuParams(dt_ms=0.1)
    sub = _substrate(pre=[0], post=[1], signed_counts=[25.0], n=2, params=p)

    # Fire neuron 0 once, by driving it over threshold, then let neuron 1 run.
    drive = _silent_drive(400, 2)
    drive[0, 0, 0] = 10.0  # mV, enough to cross the 7 mV gap on its own

    result = sub.rollout(drive)

    assert result.spikes[:, 0, 0].sum() == 1, "neuron 0 must fire exactly once"
    peak_gap = result.peak_v[0, 1] - p.v_rest_mv
    assert peak_gap / p.threshold_gap_mv > 0.1, (
        f"a 25-synapse connection moved the membrane only "
        f"{peak_gap / p.threshold_gap_mv:.2e} of the way to threshold; this is "
        "the pilot's non-transmitting substrate"
    )


def test_a_strong_connection_propagates_a_spike():
    """Enough synapses on one edge must actually fire the postsynaptic neuron.

    The exact solution puts the peak displacement at ``0.1575 * g0`` (t* = 9.24
    ms), so a single presynaptic spike reaches the 7 mV gap only from about 162
    synapses upward. Below that, transmission needs coincidence or repetition --
    which is the real constraint the calibration sweep has to satisfy, given a
    mean in-degree of 20 on the v783 subset.
    """
    sub = _substrate(
        pre=[0], post=[1], signed_counts=[200.0], n=2, params=ShiuParams(dt_ms=0.1)
    )

    drive = _silent_drive(400, 2)
    drive[0, 0, 0] = 10.0

    result = sub.rollout(drive)

    assert result.spikes[:, 0, 1].sum() >= 1, "the spike did not propagate"


# --------------------------------------------------------------------------
# Model semantics


def test_an_unstimulated_network_is_exactly_silent():
    """No intrinsic noise and no background current: baseline is exactly 0 Hz.

    This is why an empty readout is a property of the model rather than a bug,
    and why the readout needs a variance floor instead of a switch to membrane
    potential.
    """
    sub = _substrate(pre=[0, 1], post=[1, 0], signed_counts=[50.0, 50.0], n=2)

    result = sub.rollout(_silent_drive(500, 2))

    assert result.spikes.sum() == 0
    torch.testing.assert_close(
        result.final_v, torch.full((1, 2), ShiuParams().v_rest_mv)
    )


def test_inhibition_does_nothing_to_an_already_silent_neuron():
    """Stated consequence of the model: inhibition is only visible where there
    is activity, because there is no baseline to suppress."""
    p = ShiuParams(dt_ms=0.1)
    inhibited = _substrate(pre=[0], post=[1], signed_counts=[-80.0], n=2, params=p)
    isolated = _substrate(pre=[0], post=[1], signed_counts=[0.0], n=2, params=p)

    drive = _silent_drive(300, 2)
    drive[0, 0, 0] = 10.0

    a = inhibited.rollout(drive)
    b = isolated.rollout(drive)

    # The membrane does hyperpolarise; what must be identical is the firing.
    assert a.spikes[:, 0, 1].sum() == 0
    torch.testing.assert_close(a.spikes, b.spikes)
    assert a.final_v[0, 1] < b.final_v[0, 1]


def test_axonal_delay_is_uniform_and_published():
    """A presynaptic spike at step t reaches the target at t + n_delay_steps."""
    p = ShiuParams(dt_ms=0.1)
    sub = _substrate(pre=[0], post=[1], signed_counts=[25.0], n=2, params=p)

    drive = _silent_drive(60, 2)
    drive[0, 0, 0] = 10.0
    result = sub.rollout(drive, record_g=True)

    g = result.g_trace[:, 0, 1]
    first = int((g > 0).nonzero()[0])
    # The spike happens on step 0; the jump lands n_delay_steps later.
    assert first == p.n_delay_steps


def test_refractory_input_is_dropped_not_accumulated():
    """Brian2's ``(unless refractory)`` skips *every* write to v and g.

    A synaptic jump arriving at a refractory target is discarded outright: it
    must not sit in g and fire the neuron once refractoriness ends.
    """
    p = ShiuParams(dt_ms=0.1)
    sub = _substrate(pre=[], post=[], signed_counts=[], n=1, params=p)

    drive = _silent_drive(p.n_refractory_steps + 5, 1)
    drive[0, 0, 0] = 10.0  # fires on step 0, then refractory
    # Land a large input squarely inside the refractory window.
    drive[2, 0, 0] = 100.0

    result = sub.rollout(drive)

    assert result.spikes[:, 0, 0].sum() == 1, (
        "input delivered during refractoriness was accumulated instead of "
        "dropped, producing a spike Brian2 does not produce"
    )


def test_refractory_period_caps_the_firing_rate():
    """Driven every step, the neuron fires at most once per refractory period."""
    p = ShiuParams(dt_ms=0.1)
    sub = _substrate(pre=[], post=[], signed_counts=[], n=1, params=p)

    steps = 1000
    drive = torch.full((steps, 1, 1), 10.0)
    result = sub.rollout(drive)

    n_spikes = int(result.spikes.sum())
    ceiling = steps / p.n_refractory_steps
    assert 0 < n_spikes <= ceiling + 1


# --------------------------------------------------------------------------
# Calibration surface


def test_w_syn_is_the_single_free_parameter():
    """Doubling w_syn must double the synaptic drive, with the graph untouched.

    Calibration has one knob and it is an absolute scale, not a per-arm
    rescaling to a shared spectral radius or branching ratio.
    """
    counts = [25.0]
    drive = _silent_drive(200, 2)
    drive[0, 0, 0] = 10.0

    low = _substrate([0], [1], counts, 2, params=ShiuParams(dt_ms=0.1, w_syn_mv=0.275))
    high = _substrate([0], [1], counts, 2, params=ShiuParams(dt_ms=0.1, w_syn_mv=0.550))

    g_low = low.rollout(drive, record_g=True).g_trace[:, 0, 1].max()
    g_high = high.rollout(drive, record_g=True).g_trace[:, 0, 1].max()

    torch.testing.assert_close(g_high, g_low * 2.0)


def test_weights_stay_in_anatomical_units():
    """The coupling holds signed synapse counts; w_syn is applied by the substrate.

    Keeping the anatomy and the fitted scale separate is what makes the scale a
    single declared free parameter rather than something baked into each arm.
    """
    sub = _substrate(pre=[0], post=[1], signed_counts=[25.0], n=2)

    torch.testing.assert_close(sub.coupling.effective_weights(), torch.tensor([25.0]))


def test_gradients_reach_the_adapter_through_spikes():
    """PPO trains the low-rank adapter, so the rollout must stay differentiable."""
    sub = _substrate(
        pre=[0], post=[1], signed_counts=[60.0], n=2, params=ShiuParams(dt_ms=0.1)
    )
    with torch.no_grad():
        sub.coupling.B.copy_(torch.randn_like(sub.coupling.B) * 0.1)

    drive = _silent_drive(300, 2)
    drive[0, 0, 0] = 10.0

    result = sub.rollout(drive)
    result.membrane.sum().backward()

    assert sub.coupling.B.grad is not None
    assert torch.isfinite(sub.coupling.B.grad).all()
    assert sub.coupling.B.grad.abs().sum() > 0
