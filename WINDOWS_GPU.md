# Running the pilot on the CUDA desktop

The GPU is not reachable from any agent shell: the cloud container has no
accelerator, and the desktop Cowork workspace is an isolated Linux VM with no
GPU passthrough (no `nvidia-smi`, no `/dev/nvidia*`, 2 cores and 3 GB). The
RTX 4070 lives on the Windows host, outside that sandbox. So this runs natively
on Windows.

## One-time setup

In PowerShell, from the repository root:

```powershell
# uv installs and pins Python 3.13 itself; nothing else is needed.
irm https://astral.sh/uv/install.ps1 | iex

uv sync --python 3.13
```

`pyproject.toml` pins `torch>=2.14`, which ships a CUDA build on Windows, so the
lockfile resolves without a separate index. Check that it actually sees the card:

```powershell
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If that prints `False`, the CPU wheel was resolved. Force the CUDA index:

```powershell
uv pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu130
```

## The data

`data/raw/` is not committed. The four v783 tables are pinned by SHA-256 in
`data/manifest.json` and the loader refuses to run on a mismatch:

```powershell
mkdir data\raw
$B = "https://storage.googleapis.com/flywire-data/codex/data/fafb/783"
foreach ($f in "connections.csv.gz","classification.csv.gz","neurons.csv.gz","cell_stats.csv.gz") {
  Invoke-WebRequest -Uri "$B/$f" -OutFile "data\raw\$f"
}
```

## The run

```powershell
uv run python -m flywire_rl.run_pilot
```

That is the fly-ky7.11 comparison: the same pilot twice, once with the readout
standardised per arm and once without, writing
`data/results/pilot_v783_raw.json` and `data/results/pilot_v783_standardised.json`.
Defaults match `pilot_v783_a4`: 2 seeds, 1500 env steps, 40 eval episodes,
rank 8.

Useful flags: `--seeds`, `--steps`, `--eval-episodes`, `--only raw|standardised`,
`--tag` to avoid overwriting an earlier run.

## What to expect, and what to check first

The decision window is 300 steps (30 ms at the published dt = 0.1 ms). It was
3 ms until commit a095f35, which is why the substrate transmitted nothing; see
`data/results/decision_window_v783.json`. The substrate is exactly linear in
steps, so every earlier timing estimate in the README multiplies by ten.

Before committing to a long run, measure one seed:

```powershell
uv run python -m flywire_rl.run_pilot --seeds 1 --steps 500 --eval-episodes 5 --only raw --tag probe
```

Two things are worth watching. Memory: the spec's 250 MB BPTT estimate assumed
8k neurons and 30 steps; at 15,400 neurons, batch 64 and 300 steps it is roughly
4.8 GB, which fits 12 GB but with far less headroom than the spec assumed. And
throughput: if a seed takes longer than about fifteen minutes, the
pre-registered confirmatory budget (5e5 steps x 10 seeds x 3 arms) will not fit
in any reasonable wall clock, and the budget has to come from seeds, per-seed
steps, subset size or batch — not from dt, which
`data/results/dt_sensitivity_v783.json` rules out because its error is
topology-correlated and favours the real arm over its own controls.

## Reading the result

If the arm ordering survives standardisation the effect is computational; if it
vanishes it was transmission. Either way the outcome belongs in the spec, per
the acceptance criteria of fly-ky7.11.

One caution. At the corrected window the three arms differ in pre-learning
readout amplitude by 11 percent (median s.d. 0.5194 / 0.4946 / 0.4659), not the
14x that amendment A4 recorded — that spread was the per-arm branching
calibration, which commit 7dd3bd6 removed. With the confound that small, a null
result from the standardisation comparison is weak evidence either way, and the
more informative number may simply be whether any arm ordering appears after
learning at all.
