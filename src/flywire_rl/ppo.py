"""PPO with GAE over the spiking policy.

Reward is the **unshaped terminal return** (+1/-1/0) that P2 names as the
primary metric. Potential-based shaping is available but off by default and
never enters the reported outcome.

Two details are easy to get wrong and are handled explicitly here.

*Legality masks are stored, not recomputed.* By update time the game has moved
on, and a mask that differs from the one used at sampling silently changes the
importance ratio. A regression test asserts the replayed log-probability
reproduces the stored one exactly, so the ratio starts at 1.

*Seats alternate between episodes.* Amendment A2 measured a residual
second-player advantage of 60.8% in the control mirror, so a policy trained from
one seat would be scored partly on its seat. Per-seat returns are kept separate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from flywire_rl.policy import SpikingPolicy


@dataclass
class RolloutBuffer:
    """One batch of experience, with everything needed to replay each decision."""

    observations: np.ndarray
    kind_codes: np.ndarray
    source_codes: np.ndarray
    target_codes: np.ndarray
    kind_masks: np.ndarray
    source_masks: np.ndarray
    target_masks: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    episode_returns: list[float] = field(default_factory=list)
    episode_seats: list[int] = field(default_factory=list)
    episode_opponents: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        lengths = {
            len(self.observations),
            len(self.kind_codes),
            len(self.source_codes),
            len(self.target_codes),
            len(self.kind_masks),
            len(self.source_masks),
            len(self.target_masks),
            len(self.log_probs),
            len(self.values),
            len(self.rewards),
            len(self.dones),
        }
        if len(lengths) != 1:
            raise ValueError(f"buffer fields disagree in length: {sorted(lengths)}")


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
    lam: float = 0.95,
    last_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalised advantage estimation that does not bootstrap across episodes."""
    horizon = len(rewards)
    advantages = np.zeros(horizon, dtype=np.float32)
    running = 0.0

    for t in reversed(range(horizon)):
        if dones[t]:
            next_value, non_terminal = 0.0, 0.0
        else:
            next_value = values[t + 1] if t + 1 < horizon else last_value
            non_terminal = 1.0
        delta = rewards[t] + gamma * next_value * non_terminal - values[t]
        running = delta + gamma * lam * non_terminal * running
        advantages[t] = running

    return advantages, advantages + values


def collect_rollout(
    policy: SpikingPolicy,
    make_game,
    opponent,
    n_steps: int,
    start_seed: int = 0,
    max_actions_per_game: int = 600,
) -> RolloutBuffer:
    """Play until ``n_steps`` agent decisions have been recorded.

    ``opponent`` may be one callable or several. Seat and opponent advance on
    different periods so every seat-by-opponent combination appears equally
    often: with two of each, all four recur every four episodes. Training
    against a single scripted bot would let the policy specialise against that
    bot's habits, which the Stage 4 opponent swap is meant to detect rather than
    inherit.
    """
    pool = list(opponent) if isinstance(opponent, (list, tuple)) else [opponent]

    records: list[dict] = []
    rewards: list[float] = []
    dones: list[bool] = []
    episode_returns: list[float] = []
    episode_seats: list[int] = []
    episode_opponents: list[int] = []

    episode = 0
    game = make_game(start_seed)
    seat = 0
    opponent_index = 0
    first_index = 0
    actions_this_game = 0

    def finish(result: int) -> None:
        nonlocal episode, game, seat, opponent_index, first_index, actions_this_game
        if len(records) > first_index:
            rewards[-1] = float(result)
            dones[-1] = True
            episode_returns.append(float(result))
            episode_seats.append(seat)
            episode_opponents.append(opponent_index)
        episode += 1
        seat = episode % 2  # alternate seats, per amendment A2's residual
        opponent_index = (episode // 2) % len(pool)
        game = make_game(start_seed + episode)
        first_index = len(records)
        actions_this_game = 0

    while len(records) < n_steps:
        if game.is_over or actions_this_game >= max_actions_per_game:
            # result() is reported from player 0's seat; flip it for the agent.
            outcome = game.result() if seat == 0 else -game.result()
            finish(outcome if game.is_over else 0)
            continue

        if game.to_move == seat:
            chosen, record = policy.act_with_record(game)
            records.append(record)
            rewards.append(0.0)
            dones.append(False)
        else:
            chosen = pool[opponent_index](game)
        game.step(chosen)
        actions_this_game += 1

    def stack(key: str, dtype) -> np.ndarray:
        return np.asarray([r[key] for r in records], dtype=dtype)

    return RolloutBuffer(
        observations=np.stack([r["observation"] for r in records]).astype(np.float32),
        kind_codes=stack("kind_code", np.int64),
        source_codes=stack("source_code", np.int64),
        target_codes=stack("target_code", np.int64),
        kind_masks=np.stack([r["kind_mask"] for r in records]),
        source_masks=np.stack([r["source_mask"] for r in records]),
        target_masks=np.stack([r["target_mask"] for r in records]),
        log_probs=stack("log_prob", np.float32),
        values=stack("value", np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        dones=np.asarray(dones, dtype=bool),
        episode_returns=episode_returns,
        episode_seats=episode_seats,
        episode_opponents=episode_opponents,
    )


def ppo_update(
    policy: SpikingPolicy,
    buffer: RolloutBuffer,
    optimiser: torch.optim.Optimizer,
    epochs: int = 4,
    minibatch: int = 64,
    clip: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    gamma: float = 0.99,
    lam: float = 0.95,
    max_grad_norm: float = 1.0,
) -> dict[str, float]:
    """One PPO update over the buffer, returning its loss components."""
    advantages, returns = compute_gae(
        buffer.rewards, buffer.values, buffer.dones, gamma, lam
    )
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    tensors = {
        "observations": torch.from_numpy(buffer.observations),
        "kind_codes": torch.from_numpy(buffer.kind_codes),
        "source_codes": torch.from_numpy(buffer.source_codes),
        "target_codes": torch.from_numpy(buffer.target_codes),
        "kind_masks": torch.from_numpy(buffer.kind_masks),
        "source_masks": torch.from_numpy(buffer.source_masks),
        "target_masks": torch.from_numpy(buffer.target_masks),
        "old_log_probs": torch.from_numpy(buffer.log_probs),
        "advantages": torch.from_numpy(advantages),
        "returns": torch.from_numpy(returns),
    }

    horizon = len(buffer.rewards)
    stats = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "clip_fraction": 0.0,
    }
    batches = 0

    for _ in range(epochs):
        order = torch.randperm(horizon)
        for start in range(0, horizon, minibatch):
            index = order[start : start + minibatch]
            log_probs, values, entropy = policy.evaluate_actions(
                tensors["observations"][index],
                tensors["kind_codes"][index],
                tensors["source_codes"][index],
                tensors["target_codes"][index],
                tensors["kind_masks"][index],
                tensors["source_masks"][index],
                tensors["target_masks"][index],
            )
            device = log_probs.device
            old_log_probs = tensors["old_log_probs"][index].to(device)
            advantage = tensors["advantages"][index].to(device)

            ratio = (log_probs - old_log_probs).exp()
            unclipped = ratio * advantage
            clipped = ratio.clamp(1 - clip, 1 + clip) * advantage
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = (values - tensors["returns"][index].to(device)).pow(2).mean()
            entropy_mean = entropy.mean()

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_mean

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimiser.step()

            stats["policy_loss"] += float(policy_loss)
            stats["value_loss"] += float(value_loss)
            stats["entropy"] += float(entropy_mean)
            stats["clip_fraction"] += float(((ratio - 1.0).abs() > clip).float().mean())
            batches += 1

    return {key: value / max(batches, 1) for key, value in stats.items()}


def train(
    policy: SpikingPolicy,
    make_game,
    opponent,
    total_steps: int,
    rollout_steps: int = 512,
    lr: float = 3e-4,
    seed: int = 0,
    log=None,
    **update_kwargs,
) -> list[dict]:
    """Run PPO for ``total_steps`` agent decisions, returning per-rollout stats."""
    optimiser = torch.optim.AdamW(policy.parameters(), lr=lr)
    history: list[dict] = []
    collected = 0
    iteration = 0

    while collected < total_steps:
        buffer = collect_rollout(
            policy,
            make_game,
            opponent,
            n_steps=min(rollout_steps, total_steps - collected),
            start_seed=seed + iteration * 1000,
        )
        stats = ppo_update(policy, buffer, optimiser, **update_kwargs)
        collected += len(buffer.rewards)
        iteration += 1

        returns = np.asarray(buffer.episode_returns, dtype=np.float32)
        seats = np.asarray(buffer.episode_seats)
        versus = np.asarray(buffer.episode_opponents)
        for index in sorted(set(buffer.episode_opponents)):
            picked = returns[versus == index]
            stats[f"mean_return_vs_{index}"] = (
                float(picked.mean()) if picked.size else float("nan")
            )
        stats.update(
            steps=collected,
            episodes=len(returns),
            mean_return=float(returns.mean()) if returns.size else float("nan"),
            mean_return_seat0=(
                float(returns[seats == 0].mean())
                if returns.size and (seats == 0).any()
                else float("nan")
            ),
            mean_return_seat1=(
                float(returns[seats == 1].mean())
                if returns.size and (seats == 1).any()
                else float("nan")
            ),
        )
        history.append(stats)
        if log is not None:
            log(stats)

    return history
