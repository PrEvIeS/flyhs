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
import platform
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

VALID_LABELS = ("pilot", "confirmatory")
#: Tonic sensory drive, millivolts per step, identical across arms. It is
#: dt-bound: a constant drive settles at ``drive / (1 - exp(-dt/t_mbr))`` above
#: rest, so it must be recalibrated if ``ShiuParams.dt_ms`` changes.
INPUT_DRIVE_MV = 0.06
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
    steps: int = 30,
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
    if standardise:
        policy.calibrate_readout(
            reference_observations(n_reference_states).to(device)
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
) -> np.ndarray:
    """Play ``n_episodes`` and return each one's terminal return, agent's seat.

    Seats and opponents alternate on different periods so every combination is
    represented, matching how the policy was trained.
    """
    pool = list(opponents)
    scores = np.zeros(n_episodes, dtype=np.float32)
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

    return scores


def run_seed(
    arm: Graph,
    input_indices: torch.Tensor,
    output_indices: torch.Tensor,
    seed: int,
    total_steps: int,
    n_eval_episodes: int,
    rank: int = 8,
    steps: int = 30,
    device: str = "cpu",
    input_drive_mv: float = INPUT_DRIVE_MV,
    standardise: bool = False,
    **train_kwargs,
) -> np.ndarray:
    """Train one policy on one arm with one seed, then evaluate it."""
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
    train(
        policy,
        lambda s: _make_game(s),
        list(DEFAULT_OPPONENTS),
        total_steps=total_steps,
        seed=seed,
        **train_kwargs,
    )
    return evaluate_policy(policy, n_episodes=n_eval_episodes, seed=seed)


def run_experiment(
    real: Graph,
    input_indices: torch.Tensor,
    output_indices: torch.Tensor,
    label: str,
    n_seeds: int = 10,
    total_steps: int = 500_000,
    n_eval_episodes: int = 100,
    rank: int = 8,
    steps: int = 30,
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

    for name, arm in arms.items():
        rows = []
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
                    **train_kwargs,
                )
            )
        scores[name] = np.stack(rows)

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
        },
        config={
            "n_seeds": n_seeds,
            "total_steps": total_steps,
            "n_eval_episodes": n_eval_episodes,
            "rank": rank,
            "decision_window_steps": steps,
            "arm_seed": arm_seed,
            "swaps_per_edge": swaps_per_edge,
            "standardise": standardise,
            "n_neurons": real.n,
            "n_edges": real.n_edges,
            "train_kwargs": {k: str(v) for k, v in train_kwargs.items()},
        },
    )


def save_result(result: ExperimentResult, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
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
