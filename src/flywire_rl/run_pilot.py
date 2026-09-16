"""Run a pilot on the real FlyWire subset and its two matched controls.

Written to be run on the CUDA host, because nothing smaller will do: the
restored 30 ms decision window is 300 steps at the published dt, the substrate
is exactly linear in steps, and a single forward-plus-backward at batch 1 takes
146 s on two CPU cores.

Run with ``python -m flywire_rl.run_pilot``.

The default is the fly-ky7.11 comparison: the same pilot twice, once with the
readout standardised per arm and once without. That is what decides whether an
arm ordering is computational or merely transmission, so running only one half
of it answers nothing.

Nothing here chooses a protocol parameter. The window, the drive, the P1 subset
rule and the arm construction all come from the modules that own them, so this
script cannot silently disagree with the experiment it is running.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from flywire_rl import connectome as C
from flywire_rl.controls import Graph
from flywire_rl.experiment import (
    DECISION_WINDOW_MS,
    DECISION_WINDOW_STEPS,
    INPUT_DRIVE_MV,
    run_experiment,
    save_result,
)
from flywire_rl.policy import select_io_indices

DATA_RAW = Path("data/raw")
MANIFEST = Path("data/manifest.json")


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_real_graph() -> tuple[Graph, torch.Tensor, torch.Tensor]:
    """The real arm plus the interface indices, exactly as the protocol defines them.

    The manifest is verified first: a silently different FlyWire release would
    otherwise reach the results file with the provenance of the pinned one.
    """
    C.verify_manifest(DATA_RAW, MANIFEST)

    connections = C.load_connections(DATA_RAW / "connections.csv.gz")
    subset = C.select_subset(connections)
    ids = sorted(subset.neuron_ids)
    position = {int(rid): i for i, rid in enumerate(ids)}

    # Dale signs come from the per-neuron transmitter in neurons.csv.gz, not
    # from the per-edge prediction, so one neuron cannot be half-excitatory.
    nt = pd.read_csv(
        DATA_RAW / "neurons.csv.gz", usecols=["root_id", "nt_type"]
    ).set_index("root_id")["nt_type"]
    sign = C.dale_signs(ids, nt)

    edges = subset.edges
    real = Graph(
        n=len(ids),
        pre=edges.pre_root_id.map(position).to_numpy(np.int64),
        post=edges.post_root_id.map(position).to_numpy(np.int64),
        weight=edges.syn_count.to_numpy(np.float32),
        sign=sign,
    )

    # select_io_indices wants labels in neuron order, not a root_id-indexed
    # Series: passing the Series iterates its own order and silently yields the
    # wrong indices.
    classes = pd.read_csv(
        DATA_RAW / "classification.csv.gz", usecols=["root_id", "super_class"]
    ).set_index("root_id")["super_class"]
    labels = classes.reindex(ids).to_numpy(dtype=object)
    inputs, outputs = select_io_indices(labels)

    return real, torch.from_numpy(inputs), torch.from_numpy(outputs)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--steps", type=int, default=1500, help="env steps per seed")
    parser.add_argument("--eval-episodes", type=int, default=40)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--label", default="pilot", choices=["pilot", "confirmatory"])
    parser.add_argument("--out", default="data/results")
    parser.add_argument("--tag", default="", help="suffix for the output filenames")
    parser.add_argument(
        "--only",
        choices=["both", "raw", "standardised"],
        default="both",
        help="which arms of the fly-ky7.11 comparison to run",
    )
    args = parser.parse_args(argv)

    device = pick_device(args.device)
    print(f"device: {device}", flush=True)
    if device == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}", flush=True)
    elif device == "cpu":
        print(
            "  WARNING: on CPU a single forward+backward at batch 1 takes ~146 s "
            "at this window. This will not finish in reasonable time.",
            flush=True,
        )

    print("loading connectome and verifying the manifest...", flush=True)
    t0 = time.time()
    real, inputs, outputs = build_real_graph()
    print(
        f"  N={real.n:,} |E|={real.n_edges:,} inputs={inputs.numel()} "
        f"outputs={outputs.numel()}  [{time.time() - t0:.1f}s]",
        flush=True,
    )
    print(
        f"window: {DECISION_WINDOW_MS:.0f} ms = {DECISION_WINDOW_STEPS} steps, "
        f"drive {INPUT_DRIVE_MV}",
        flush=True,
    )

    variants = {"both": [False, True], "raw": [False], "standardised": [True]}[args.only]
    out_dir = Path(args.out)
    written = []

    for standardise in variants:
        name = "standardised" if standardise else "raw"
        print(f"\n=== {name} readout ===", flush=True)
        started = time.time()

        def progress(arm, seed, _name=name, _start=None):
            print(f"  [{_name}] {arm} seed {seed}", flush=True)

        result = run_experiment(
            real,
            inputs,
            outputs,
            label=args.label,
            n_seeds=args.seeds,
            total_steps=args.steps,
            n_eval_episodes=args.eval_episodes,
            rank=args.rank,
            device=device,
            manifest_path=MANIFEST,
            progress=progress,
            standardise=standardise,
        )

        for arm, scores in result.scores.items():
            per_seed = scores.mean(axis=1)
            print(
                f"  {arm:9} per-seed {np.round(per_seed, 4).tolist()}  "
                f"overall {scores.mean():+.4f}",
                flush=True,
            )

        suffix = f"_{args.tag}" if args.tag else ""
        path = out_dir / f"{args.label}_{C.RELEASE_TAG}_{name}{suffix}.json"
        save_result(result, path)
        written.append(str(path))
        print(f"  saved {path}  [{time.time() - started:.0f}s]", flush=True)

    print("\nwrote: " + ", ".join(written), flush=True)
    if len(written) == 2:
        print(
            "fly-ky7.11 reads the two together: if the arm ordering survives "
            "standardisation the effect is computational, if it vanishes it was "
            "transmission.",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
