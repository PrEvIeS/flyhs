"""The decision window is pre-registered in milliseconds, not in steps.

The protocol fixed a 30 ms decision window at dt = 1.0 ms, so 30 steps. The
Shiu re-parameterisation cut dt to 0.1 ms and left the step count at 30, which
silently shortened the window to 3 ms -- less than two synaptic delays, so no
signal could cross more than a single hop. On the real subset that left every
one of the 44 output units pinned at v_rest in all three arms, and raising
w_syn a hundredfold did not help: the binding constraint was time, not gain.

These tests keep the window anchored to the protocol's milliseconds, so the
same dt change cannot break it again silently.
"""

from dataclasses import replace

import pytest

from flywire_rl.experiment import (
    DECISION_WINDOW_MS,
    DECISION_WINDOW_STEPS,
    decision_window_steps,
)
from flywire_rl.spiking import ShiuParams


def test_window_matches_the_pre_registered_thirty_milliseconds():
    assert DECISION_WINDOW_MS == 30.0


def test_step_count_is_derived_from_dt_not_hardcoded():
    """The regression that caused the outage: a dt change must move the step
    count, not shorten the window."""
    params = ShiuParams()
    assert DECISION_WINDOW_STEPS == round(DECISION_WINDOW_MS / params.dt_ms)
    assert decision_window_steps(replace(params, dt_ms=1.0)) == 30
    assert decision_window_steps(replace(params, dt_ms=0.1)) == 300
    assert decision_window_steps(replace(params, dt_ms=0.05)) == 600


def test_window_spans_enough_synaptic_delays_to_cross_the_graph():
    """A4 measured the output population at 1 to 3 hops from the input set, so
    a window that fits fewer than 3 delays cannot reach all of it. At dt = 0.1
    the broken 30-step window fitted 1."""
    params = ShiuParams()
    hops = DECISION_WINDOW_STEPS // params.n_delay_steps
    assert hops >= 3, f"window fits only {hops} synaptic delays"


def test_the_broken_configuration_is_recognisably_too_short():
    """Pins the failure mode itself, so a future edit that reintroduces it
    fails here rather than in a 466-hour confirmatory run."""
    params = ShiuParams()
    assert 30 // params.n_delay_steps < 2, (
        "30 steps at this dt now fits 2+ delays; the guard above needs rechecking"
    )


def test_defaults_use_the_derived_window():
    import inspect

    from flywire_rl.experiment import make_policy, run_experiment, run_seed

    for fn in (make_policy, run_seed, run_experiment):
        got = inspect.signature(fn).parameters["steps"].default
        assert got == DECISION_WINDOW_STEPS, f"{fn.__name__} defaults to {got}"


def test_input_drive_actually_reaches_threshold():
    """At the 0.06 this replaced, no neuron in the subset fired at all."""
    from flywire_rl.experiment import INPUT_DRIVE_MV

    params = ShiuParams()
    # A sustained drive settles at drive / (1 - exp(-dt/t_mbr)) above rest.
    import math

    settle = 1.0 - math.exp(-params.dt_ms / params.t_mbr_ms)
    # The encoder's random init puts |encoder(obs)| on the order of 0.1.
    reached = 0.1 * INPUT_DRIVE_MV / settle
    assert reached > params.threshold_gap_mv, (
        f"sustained drive settles {reached:.2f} mV above rest, below the "
        f"{params.threshold_gap_mv} mV gap: the substrate would be silent"
    )
