"""The standardise flag must survive the trip from the CLI to the result file.

``calibrate_readout`` itself is well covered in test_amplitude_control, but
every one of those tests calls it directly on a policy. Nothing exercised the
path the pilot actually takes -- run_experiment -> run_seed -> make_policy --
so a pass-through that silently dropped the flag would leave the two fly-ky7.11
arms identical while both result files still claimed to differ. The comparison
would then measure nothing, and say so nowhere.
"""

from __future__ import annotations

import numpy as np
import torch

from flywire_rl.controls import Graph
from flywire_rl.experiment import make_policy, run_experiment
from flywire_rl.spiking import ShiuParams

# Small enough to train in seconds, large enough that the readout has something
# to standardise: the amplitude control is meaningless on a silent substrate.
N = 120
M = 1200
INPUTS = torch.arange(0, 10)
OUTPUTS = torch.arange(N - 6, N)
BUDGET = dict(
    n_seeds=1,
    total_steps=30,
    n_eval_episodes=2,
    rank=2,
    steps=5,
    rollout_steps=15,
    minibatch=5,
    epochs=1,
)


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


def test_make_policy_marks_the_policy_as_standardised():
    """The innermost link in the chain, asserted on its own."""
    raw = make_policy(_graph(), INPUTS, OUTPUTS, seed=0, rank=2, steps=5)
    standardised = make_policy(
        _graph(), INPUTS, OUTPUTS, seed=0, rank=2, steps=5, standardise=True
    )

    assert not bool(raw.standardised)
    assert bool(standardised.standardised)


def test_run_experiment_records_the_flag_it_was_given():
    result = run_experiment(
        _graph(), INPUTS, OUTPUTS, label="pilot", standardise=True, **BUDGET
    )

    assert result.config["standardise"] is True


def test_run_experiment_defaults_to_the_raw_readout():
    """The default must stay raw: standardisation is the control, not the arm."""
    result = run_experiment(_graph(), INPUTS, OUTPUTS, label="pilot", **BUDGET)

    assert result.config["standardise"] is False


def test_run_experiment_records_the_window_in_milliseconds():
    """Steps alone cannot be checked against the protocol without dt."""
    result = run_experiment(_graph(), INPUTS, OUTPUTS, label="pilot", **BUDGET)

    assert result.config["decision_window_steps"] == BUDGET["steps"]
    assert result.config["decision_window_ms"] == BUDGET["steps"] * ShiuParams().dt_ms


def test_both_arms_are_produced_for_every_topology_control():
    """A dropped arm would make the comparison incomparable, not merely smaller."""
    result = run_experiment(
        _graph(), INPUTS, OUTPUTS, label="pilot", standardise=True, **BUDGET
    )

    assert set(result.scores) == {"real", "shuffled", "random"}
    for arm, scores in result.scores.items():
        assert scores.shape == (BUDGET["n_seeds"], BUDGET["n_eval_episodes"]), arm


# --- the calibration must leave a trace in the result --------------------


def test_calibration_statistics_reach_the_result_provenance():
    """A dead readout makes calibration a no-op; the boolean flag hides that.

    calibrate_readout falls back to scale = 1.0 for units that never move, by
    design. So ``standardise: true`` on its own cannot distinguish a real
    standardisation from one that did nothing, and the pilot's conclusion
    depends on which it was.
    """
    result = run_experiment(
        _graph(), INPUTS, OUTPUTS, label="pilot", standardise=True, **BUDGET
    )

    calibration = result.provenance["calibration"]
    assert set(calibration) == {"real", "shuffled", "random"}
    for arm, per_seed in calibration.items():
        assert len(per_seed) == BUDGET["n_seeds"], arm
        stats = per_seed[0]
        assert stats["n_units"] == len(OUTPUTS)
        assert stats["n_states"] > 0
        assert 0 <= stats["dead_units"] <= stats["n_units"]


def test_a_raw_run_records_no_calibration():
    result = run_experiment(_graph(), INPUTS, OUTPUTS, label="pilot", **BUDGET)

    assert result.provenance["calibration"] == {}


def test_calibrate_readout_records_what_it_returns():
    policy = make_policy(_graph(), INPUTS, OUTPUTS, seed=0, rank=2, steps=5)
    assert policy.calibration is None

    returned = policy.calibrate_readout(
        torch.stack([torch.zeros(policy.encoder.in_features) for _ in range(4)])
    )
    assert returned == policy.calibration
    assert policy.calibration["n_states"] == 4


def test_calibration_does_not_travel_in_the_state_dict():
    """It is a record for the result file, not state the model computes with."""
    policy = make_policy(
        _graph(), INPUTS, OUTPUTS, seed=0, rank=2, steps=5, standardise=True
    )
    assert policy.calibration is not None
    assert not any("calibration" in key for key in policy.state_dict())


# --- the coupling mode must be selectable and recorded --------------------


def test_coupling_mode_defaults_to_sparse_and_is_recorded():
    """The default is unchanged on purpose, so earlier results stay comparable."""
    result = run_experiment(_graph(), INPUTS, OUTPUTS, label="pilot", **BUDGET)

    assert result.config["coupling"] == "sparse"


def test_coupling_mode_can_be_selected_and_reaches_the_result():
    """It decides whether a configuration is runnable, not merely how fast.

    Measured on a 12 GB card at the 300-step window: gather runs in 274 ms and
    961 MB, sparse does not finish a single forward+backward in 240 s. A result
    file that did not say which mode produced it could not be compared with one
    that used the other.
    """
    result = run_experiment(
        _graph(), INPUTS, OUTPUTS, label="pilot", coupling="gather", **BUDGET
    )

    assert result.config["coupling"] == "gather"


def test_make_policy_builds_the_requested_mode():
    from flywire_rl.spiking import CouplingMode

    policy = make_policy(
        _graph(), INPUTS, OUTPUTS, seed=0, rank=2, steps=5, coupling="gather"
    )

    assert policy.substrate.coupling.mode is CouplingMode.GATHER
