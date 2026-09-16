"""The experiment harness: build arms, train seeds, evaluate, analyse.

Every choice here exists to keep one property true — **arms differ in their
edges and in nothing else**. Same node count, same edge count, same sign vector,
same spectral radius, same calibrated firing rate, same interface indices, same
adapter rank, same optimiser, same seeds, same budget. Anything else that
differed would be a second explanation for whatever the experiment measures, and
the result would not be about topology.

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

from dataclasses import replace as _replace

from flywire_rl.controls import (
    Graph,
    calibrate_branching,
    random_control,
    shuffled_control,
)
from flywire_rl.minicard import MiniCard, board_control_opponent, greedy_face_opponent
from flywire_rl.policy import SpikingPolicy
from flywire_rl.ppo import train

VALID_LABELS = ("pilot", "confirmatory")
#: Amendment A3 target. Branching near 1 is the spiking analogue of a unit
#: spectral radius; P4's original rho = 0.95 understated the required weight
#: scale by a factor of ~1780 and left the graph transmitting nothing.
BRANCHING_TARGET = 1.0
#: Amendment A4 removed the 5 Hz input calibration: once branching is fixed the
#: firing rate is set by the network rather than the input, so that calibration
#: is ill-posed. A single constant, identical across arms, replaces it.
INPUT_GAIN = 20.0
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
    target_branching: float = BRANCHING_TARGET,
    tolerance: float = 0.12,
    device: str = "cpu",
) -> dict[str, Graph]:
    """Construct the three P4 arms, each scaled to transmit at the same rate.

    Each arm gets its *own* weight scale, chosen so all arms carry one spike
    per spike. Matching the scale instead would leave the arms transmitting at
    different rates, which is a larger difference than the topology under test.
    """
    raw = {
        "real": real,
        "shuffled": shuffled_control(real, seed=seed, swaps_per_edge=swaps_per_edge),
        "random": random_control(real, seed=seed),
    }
    arms = {}
    for name, arm in raw.items():
        scale = calibrate_branching(
            arm, target=target_branching, tolerance=tolerance, device=device
        )
        arms[name] = _replace(arm, weight=(arm.weight * scale).astype(np.float32))
    return arms


def make_policy(
    arm: Graph,
    input_indices: torch.Tensor,
    output_indices: torch.Tensor,
    seed: int = 0,
    rank: int = 8,
    steps: int = 30,
    input_gain: float = 1.0,
    device: str = "cpu",
) -> SpikingPolicy:
    """Wrap an arm as a policy. Dale signs are applied; the wiring stays frozen."""
    torch.manual_seed(seed)
    signed = arm.sign[arm.pre].astype(np.float32) * arm.weight
    return SpikingPolicy(
        n_neurons=arm.n,
        pre_idx=torch.from_numpy(np.ascontiguousarray(arm.pre)),
        post_idx=torch.from_numpy(np.ascontiguousarray(arm.post)),
        weights=torch.from_numpy(signed),
        input_indices=input_indices,
        output_indices=output_indices,
        rank=rank,
        steps=steps,
        input_gain=input_gain,
    ).to(device)


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
    input_gain: float = INPUT_GAIN,
    **train_kwargs,
) -> np.ndarray:
    """Train one policy on one arm with one seed, then evaluate it."""
    policy = make_policy(
        arm, input_indices, output_indices, seed, rank, steps, input_gain, device
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
    **train_kwargs,
) -> ExperimentResult:
    """Run every arm across every seed and return the score matrices."""
    arms = build_arms(
        real, seed=arm_seed, swaps_per_edge=swaps_per_edge, device=device
    )
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
