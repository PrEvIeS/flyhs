"""Throughput probe for P5, at the N the P1 rule actually returned.

P5's budget arithmetic assumed ``N ~ 8k``; rule P1 returns 15,400 with 310,867
edges (result R1). This measures what that costs per coupling mode, so the
budget question is settled by a number rather than by an estimate.

Run with ``python -m flywire_rl.bench``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from flywire_rl import connectome as C
from flywire_rl.spiking import CouplingMode, LIFSubstrate, LowRankCoupling

DATA_RAW = Path("data/raw")
SUBSET_CACHE = Path("data/subset_v783.npz")


def build_subset_arrays(force: bool = False) -> dict[str, np.ndarray]:
    """Materialise the P1 subset as edge arrays, caching the pandas pipeline."""
    if SUBSET_CACHE.exists() and not force:
        with np.load(SUBSET_CACHE) as cached:
            return {key: cached[key] for key in cached.files}

    C.verify_manifest(DATA_RAW, Path("data/manifest.json"))
    connections = C.load_connections(DATA_RAW / "connections.csv.gz")
    subset = C.select_subset(connections)

    neuron_ids = np.array(sorted(subset.neuron_ids), dtype=np.int64)
    position = {int(rid): i for i, rid in enumerate(neuron_ids)}
    edges = subset.edges

    arrays = {
        "pre": edges.pre_root_id.map(position).to_numpy(dtype=np.int64),
        "post": edges.post_root_id.map(position).to_numpy(dtype=np.int64),
        "weight": edges.syn_count.to_numpy(dtype=np.float32),
        "n_neurons": np.array(len(neuron_ids), dtype=np.int64),
    }
    SUBSET_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(SUBSET_CACHE, **arrays)
    return arrays


def _sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def _allocated_mb(device: str) -> float:
    """Bytes currently held, sampled while the autograd graph is still alive.

    Sampling after ``backward`` reports the post-release figure, which is not a
    high-water mark: an earlier version of this probe did exactly that and
    reported an identical 128 MB for all three modes, including one holding a
    0.95 GB dense matrix.
    """
    if device == "mps":
        return torch.mps.driver_allocated_memory() / 1e6
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / 1e6
    return float("nan")


def benchmark_mode(
    arrays: dict[str, np.ndarray],
    mode: CouplingMode,
    device: str,
    batch: int,
    steps: int,
    rank: int,
    repeats: int,
) -> dict:
    """Time forward+backward for one coupling mode, or report why it cannot run."""
    n = int(arrays["n_neurons"])
    result = {"mode": mode.value, "device": device, "batch": batch}

    try:
        # P4 requires a gain normalisation before training; a crude scale is
        # enough here because only timing is being measured.
        weights = torch.from_numpy(arrays["weight"])
        weights = weights / weights.abs().max()

        coupling = LowRankCoupling(
            n,
            torch.from_numpy(arrays["pre"]),
            torch.from_numpy(arrays["post"]),
            weights,
            rank=rank,
            mode=mode,
        ).to(device)
        substrate = LIFSubstrate(coupling)
        drive = torch.randn(steps, batch, n, device=device) * 5.0

        held_mb = 0.0

        def one_rollout() -> None:
            nonlocal held_mb
            coupling.zero_grad(set_to_none=True)
            raster, _ = substrate(drive)
            # Sample while the graph is alive; after backward it is released.
            held_mb = max(held_mb, _allocated_mb(device))
            raster.sum().backward()

        one_rollout()  # warm up kernels and compilation
        _sync(device)

        start = time.perf_counter()
        for _ in range(repeats):
            one_rollout()
        _sync(device)
        elapsed = (time.perf_counter() - start) / repeats

        result.update(
            ok=True,
            seconds_per_rollout=elapsed,
            env_steps_per_second=batch / elapsed,
            hours_per_500k_steps=(500_000 / batch) * elapsed / 3600,
            peak_memory_mb=held_mb,
        )
    except Exception as error:  # noqa: BLE001 - the failure IS the measurement
        result.update(ok=False, error=f"{type(error).__name__}: {error}"[:200])

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--modes", default="gather,sparse,dense")
    args = parser.parse_args()

    device = args.device or (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    arrays = build_subset_arrays()
    n = int(arrays["n_neurons"])
    n_edges = len(arrays["pre"])
    print(f"device={device}  N={n:,}  |E|={n_edges:,}  density={n_edges / n**2:.5f}")
    print(f"batch={args.batch}  steps={args.steps}  rank={args.rank}")
    print(f"trainable adapter params: {2 * n * args.rank:,}")
    print(
        f"dense W would be {n * n * 4 / 1e9:.2f} GB; edges are {n_edges * 4 / 1e6:.1f} MB"
    )
    print()

    for name in args.modes.split(","):
        outcome = benchmark_mode(
            arrays,
            CouplingMode(name.strip()),
            device,
            args.batch,
            args.steps,
            args.rank,
            args.repeats,
        )
        if outcome["ok"]:
            print(
                f"{outcome['mode']:>7}  "
                f"{outcome['seconds_per_rollout'] * 1000:8.1f} ms/rollout  "
                f"{outcome['env_steps_per_second']:8.1f} env-steps/s  "
                f"{outcome['hours_per_500k_steps']:6.2f} h per 5e5 steps  "
                f"peak {outcome['peak_memory_mb']:7.0f} MB"
            )
        else:
            print(f"{outcome['mode']:>7}  FAILED  {outcome['error']}")


if __name__ == "__main__":
    main()
