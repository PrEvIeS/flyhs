"""The experiment harness: build arms, train seeds, evaluate, analyse.

Every choice here exists to keep one property true — **arms differ in their
edges and in nothing else**. Same node count, same edge count, same sign vector,
same weight multiset, same per-synapse scale, same tonic input, same interface
indices, same adapter rank, same optimiser, same seeds, same budget. Anything
else that differed would be a second explanation for whatever the experiment
measures, and the result would not be about topology.

Note what is deliberately *not* matched any more: the arms are no longer
rescaled to a shared spectral radius or branching ratio. Both were criteria for
a linear rate network, both required a different weight scale per arm, and the
first left the substrate transmitting nothing at all.

A run must declare itself ``pilot`` or ``confirmatory``. Exploratory numbers
later described as confirmatory are the ordinary way pre-registration fails, so
the label is required rather than optional.
"""

from __future__ import annotations

import json
import os
import platform
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from flywire_rl.controls import (
    Graph,
    random_control,
    shuffled_control,
)
from flywire_rl.minicard import MiniCard, board_control_opponent, greedy_face_opponent
from flywire_rl.policy import SpikingPolicy, reference_observations
from flywire_rl.ppo import train
from flywire_rl.spiking import ShiuParams

VALID_LABELS = ("pilot", "confirmatory")
#: Tonic sensory drive, identical across arms, scaling the encoder's output. It
#: is dt-bound: a constant drive settles at ``drive / (1 - exp(-dt/t_mbr))``
#: above rest, so it must be recalibrated if ``ShiuParams.dt_ms`` changes.
#:
#: Restored to the 20.0 used before the Shiu re-parameterisation. At the 0.06
#: that replaced it the drive reaching a neuron peaked at 0.03 mV against the
#: 7 mV gap from rest to threshold, and **not one neuron in the 15,400-node
#: subset fired** -- not even the directly driven inputs. At 20.0, 914 of 1042
#: input neurons fire.
#:
#: Stated plainly: 20.0 drives the input population well past threshold rather
#: than to a calibrated rate. Amendment A4 removed the 5 Hz rate calibration as
#: ill-posed once branching is fixed, and nothing replaced it, so this is a
#: working value and not a calibrated one. What it must not do is differ
#: between arms, and it does not.
INPUT_DRIVE_MV = 20.0

#: The decision window **in milliseconds**, as the protocol pre-registers it.
#:
#: Expressed in time rather than steps on purpose. The protocol fixed "30 ms
#: decision window" at dt = 1.0 ms, so 30 steps. Moving to the Shiu
#: parameterisation cut dt to 0.1 ms and left the step count at 30, which
#: silently shortened the window to 3 ms -- shorter than two synaptic delays
#: (t_delay = 1.8 ms), so no signal could cross more than one hop. Measured on
#: the real subset at that window: zero non-input neurons fired, and every one
#: of the 44 output units sat at v_rest in all three arms. Raising w_syn a
#: hundredfold did not help, because the binding constraint was time, not gain.
#:
#: Deriving the step count from dt keeps that from recurring: change dt and the
#: window stays 30 ms.
DECISION_WINDOW_MS = 30.0


#: How far the realised window may sit from the pre-registered one before the
#: run is refused, in milliseconds. Tight on purpose: a whole step at the
#: published dt is 0.1 ms, so anything a tenth of that size is a dt that does
#: not divide the window, not a rounding artefact.
WINDOW_TOLERANCE_MS = 1e-2


def decision_window_steps(params: ShiuParams | None = None) -> int:
    """Steps spanning ``DECISION_WINDOW_MS`` at the substrate's own dt.

    Deriving the count from dt stops the *old* failure -- a hardcoded 30 that
    silently became 3 ms when dt moved to 0.1 -- but it does not stop the
    mirror image of it. ``round`` absorbs any remainder without complaint, so a
    future dt that does not divide 30 ms evenly would drift the window off the
    pre-registered value with nothing to show for it: dt = 0.13 rounds 230.77
    up to 231 steps, a 30.03 ms window presented as 30. That is the same class
    of silent protocol drift, so it raises rather than rounds.
    """
    dt_ms = (params or ShiuParams()).dt_ms
    steps = round(DECISION_WINDOW_MS / dt_ms)
    realised_ms = steps * dt_ms
    if abs(realised_ms - DECISION_WINDOW_MS) > WINDOW_TOLERANCE_MS:
        raise ValueError(
            f"dt_ms = {dt_ms} does not divide the pre-registered "
            f"{DECISION_WINDOW_MS} ms decision window: {steps} steps span "
            f"{realised_ms:.4f} ms. Choose a dt that divides the window, or "
            f"amend DECISION_WINDOW_MS deliberately."
        )
    return steps


#: 300 at the published dt = 0.1 ms.
DECISION_WINDOW_STEPS = decision_window_steps()
DEFAULT_OPPONENTS = (board_control_opponent, greedy_face_opponent)


@dataclass
class ExperimentResult:
    """Per-arm ``(n_seeds, n_episodes)`` evaluation scores plus provenance."""

    label: str
    scores: dict[str, np.ndarray]
    provenance: dict
    config: dict

    def __post_init__(self) -> None:
        if self.label not in VALID_LABELS:
            raise ValueError(
                f"label must be one of {VALID_LABELS}, not {self.label!r}: a run "
                "that does not declare itself pilot or confirmatory can be "
                "reinterpreted later"
            )


def build_arms(
    real: Graph,
    seed: int = 0,
    swaps_per_edge: int = 100,
) -> dict[str, Graph]:
    """Construct the three P4 arms. No arm is reweighted.

    The pilot gave each arm its own weight scale, chosen so that all three
    transmitted at the same branching ratio. That is now deliberately gone, for
    two reasons.

    The scale it solved for was a *dynamical* target on a hard-threshold LIF,
    where branching is the wrong criterion, and it needed visibly different
    scales per arm (real 143.3, shuffled 40.7, random 38.2 at branching 0.70) --
    a difference in gain smuggled in alongside the difference in topology.

    Under the Shiu parameterisation the anatomy keeps its own units: weights are
    synapse counts and the single absolute scale ``w_syn`` is shared by every
    arm. Both controls permute edges while preserving the weight multiset
    exactly, so total synaptic mass is matched by construction -- the "matched
    wiring budget" that Correig-Fraga et al. show is what separates a topology
    result from an artefact of how much wire each graph was allowed. Whatever
    transmission difference remains between arms is then the effect under test.
    """
    return {
        "real": real,
        "shuffled": shuffled_control(real, seed=seed, swaps_per_edge=swaps_per_edge),
        "random": random_control(real, seed=seed),
    }


def make_policy(
    arm: Graph,
    input_indices: torch.Tensor,
    output_indices: torch.Tensor,
    seed: int = 0,
    rank: int = 8,
    steps: int = DECISION_WINDOW_STEPS,
    input_drive_mv: float = INPUT_DRIVE_MV,
    device: str = "cpu",
    standardise: bool = False,
    n_reference_states: int = 64,
) -> SpikingPolicy:
    """Wrap an arm as a policy. Dale signs are applied; the wiring stays frozen.

    With ``standardise=True`` the readout is calibrated on a fixed, shared set
    of reference states before any training, so every arm enters training with
    a zero-mean unit-variance readout. That is the amplitude control of
    fly-ky7.11: see ``SpikingPolicy.calibrate_readout`` for what it decides.
    """
    torch.manual_seed(seed)
    signed = arm.sign[arm.pre].astype(np.float32) * arm.weight
    policy = SpikingPolicy(
        n_neurons=arm.n,
        pre_idx=torch.from_numpy(np.ascontiguousarray(arm.pre)),
        post_idx=torch.from_numpy(np.ascontiguousarray(arm.post)),
        weights=torch.from_numpy(signed),
        input_indices=input_indices,
        output_indices=output_indices,
        rank=rank,
        steps=steps,
        input_drive_mv=input_drive_mv,
    ).to(device)
    # Attached rather than returned, so that every existing caller keeps the
    # bare policy it expects. Dropping these numbers was the defect: a readout
    # that collapses to mostly dead units makes calibration fall back to
    # scale = 1.0 by design, which turns standardisation into a no-op for that
    # arm -- and the saved result would have recorded only ``standardise: true``
    # and shown nothing of it.
    policy.calibration = (
        policy.calibrate_readout(reference_observations(n_reference_states).to(device))
        if standardise
        else None
    )
    return policy


def _make_game(seed: int, archetype: str = "aggro", max_turns: int = 30):
    """Mirror matchup: both players hold the same deck, so only policy differs."""
    game = MiniCard(seed=seed, archetypes=(archetype, archetype), max_turns=max_turns)
    game.reset()
    return game


def evaluate_policy(
    policy: SpikingPolicy,
    opponents=DEFAULT_OPPONENTS,
    n_episodes: int = 100,
    seed: int = 0,
    archetype: str = "aggro",
    max_actions: int = 600,
    truncation_sink: list[int] | None = None,
) -> np.ndarray:
    """Play ``n_episodes`` and return each one's terminal return, agent's seat.

    Seats and opponents alternate on different periods so every combination is
    represented, matching how the policy was trained.

    An episode that hits ``max_actions`` scores 0.0, which is the same value a
    genuine draw produces. That is the right score -- an unfinished game has no
    winner -- but the two are not the same event, and a batch of truncations
    reads downstream as a clean null result rather than as a policy cycling
    through legal actions without ever ending a turn. ``truncation_sink``
    collects how many episodes were cut off, so the difference is visible in
    the saved result instead of only in the scores.
    """
    pool = list(opponents)
    scores = np.zeros(n_episodes, dtype=np.float32)
    truncated = 0
    torch.manual_seed(seed)

    for episode in range(n_episodes):
        agent_seat = episode % 2
        foe = pool[(episode // 2) % len(pool)]
        game = _make_game(seed * 100_003 + episode, archetype)

        actions = 0
        while not game.is_over and actions < max_actions:
            if game.to_move == agent_seat:
                chosen, _ = policy.act_with_record(game)
            else:
                chosen = foe(game)
            game.step(chosen)
            actions += 1

        outcome = game.result() if agent_seat == 0 else -game.result()
        scores[episode] = float(outcome) if game.is_over else 0.0
        truncated += not game.is_over

    if truncation_sink is not None:
        truncation_sink.append(truncated)
    return scores


def run_seed(
    arm: Graph,
    input_indices: torch.Tensor,
    output_indices: torch.Tensor,
    seed: int,
    total_steps: int,
    n_eval_episodes: int,
    rank: int = 8,
    steps: int = DECISION_WINDOW_STEPS,
    device: str = "cpu",
    input_drive_mv: float = INPUT_DRIVE_MV,
    standardise: bool = False,
    calibration_sink: list[dict[str, float]] | None = None,
    truncation_sink: list[int] | None = None,
    **train_kwargs,
) -> np.ndarray:
    """Train one policy on one arm with one seed, then evaluate it.

    ``calibration_sink`` collects the readout statistics measured before
    training, so the saved result can show whether standardisation actually did
    anything for this arm rather than only that it was requested.
    """
    policy = make_policy(
        arm,
        input_indices,
        output_indices,
        seed,
        rank,
        steps,
        input_drive_mv,
        device,
        standardise=standardise,
    )
    if calibration_sink is not None and policy.calibration is not None:
        calibration_sink.append(policy.calibration)
    train(
        policy,
        lambda s: _make_game(s),
        list(DEFAULT_OPPONENTS),
        total_steps=total_steps,
        seed=seed,
        **train_kwargs,
    )
    return evaluate_policy(
        policy,
        n_episodes=n_eval_episodes,
        seed=seed,
        truncation_sink=truncation_sink,
    )


def run_experiment(
    real: Graph,
    input_indices: torch.Tensor,
    output_indices: torch.Tensor,
    label: str,
    n_seeds: int = 10,
    total_steps: int = 500_000,
    n_eval_episodes: int = 100,
    rank: int = 8,
    steps: int = DECISION_WINDOW_STEPS,
    device: str = "cpu",
    arm_seed: int = 0,
    swaps_per_edge: int = 100,
    manifest_path: Path | None = None,
    progress=None,
    standardise: bool = False,
    **train_kwargs,
) -> ExperimentResult:
    """Run every arm across every seed and return the score matrices."""
    arms = build_arms(real, seed=arm_seed, swaps_per_edge=swaps_per_edge)
    scores: dict[str, np.ndarray] = {}

    calibration: dict[str, list[dict[str, float]]] = {}
    truncated: dict[str, list[int]] = {}

    for name, arm in arms.items():
        rows = []
        sink: list[dict[str, float]] = []
        cut_off: list[int] = []
        for seed in range(n_seeds):
            if progress is not None:
                progress(name, seed)
            rows.append(
                run_seed(
                    arm,
                    input_indices,
                    output_indices,
                    seed,
                    total_steps,
                    n_eval_episodes,
                    rank,
                    steps,
                    device,
                    standardise=standardise,
                    calibration_sink=sink,
                    truncation_sink=cut_off,
                    **train_kwargs,
                )
            )
        scores[name] = np.stack(rows)
        truncated[name] = cut_off
        if sink:
            calibration[name] = sink

    manifest = {}
    if manifest_path is not None and Path(manifest_path).exists():
        manifest = json.loads(Path(manifest_path).read_text()).get("files", {})

    return ExperimentResult(
        label=label,
        scores=scores,
        provenance={
            "device": device,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
            "host": platform.platform(),
            "manifest": {k: v.get("sha256") for k, v in manifest.items()},
            # Empty on a raw run. On a standardised one these are what say
            # whether standardisation was real for each arm or a no-op over a
            # dead readout, which the boolean flag alone cannot distinguish.
            "calibration": calibration,
            # Per arm, per seed: how many evaluation episodes hit the action
            # cap instead of ending. They score 0.0, the same as a draw, so
            # without this an arm full of unfinished games is indistinguishable
            # from an arm that genuinely drew.
            "truncated_episodes": truncated,
        },
        config={
            "n_seeds": n_seeds,
            "total_steps": total_steps,
            "n_eval_episodes": n_eval_episodes,
            "rank": rank,
            "decision_window_steps": steps,
            "decision_window_ms": steps * ShiuParams().dt_ms,
            "arm_seed": arm_seed,
            "swaps_per_edge": swaps_per_edge,
            "standardise": standardise,
            "n_neurons": real.n,
            "n_edges": real.n_edges,
            "train_kwargs": {k: str(v) for k, v in train_kwargs.items()},
        },
    )


def save_result(result: ExperimentResult, path: Path) -> None:
    """Write a result so that an interrupted write cannot destroy a good one.

    ``write_text`` truncates the destination before it writes a byte, so a
    crash, a reboot or a full disk mid-write leaves nothing at a path that held
    a complete run a moment earlier. These runs cost GPU-hours and the filename
    is deterministic, so a rerun with the same label and tag aims at exactly
    the file it would be most expensive to lose. Writing to a sibling temp file
    and moving it into place makes the replacement atomic on the same
    filesystem: the path holds either the old result or the new one, never a
    half of either.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            {
                "label": result.label,
                "scores": {k: v.tolist() for k, v in result.scores.items()},
                "provenance": result.provenance,
                "config": result.config,
            },
            indent=2,
        )
        + "\n"
    )

    # Same directory, so os.replace is a rename within one filesystem and
    # therefore atomic. A temp file in the system temp dir would not be.
    handle = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".partial",
        delete=False,
    )
    try:
        with handle as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


def load_result(path: Path) -> ExperimentResult:
    payload = json.loads(Path(path).read_text())
    return ExperimentResult(
        label=payload["label"],
        scores={
            k: np.asarray(v, dtype=np.float64) for k, v in payload["scores"].items()
        },
        provenance=payload["provenance"],
        config=payload["config"],
    )
