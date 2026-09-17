# Running the pilot on the CUDA desktop

The GPU is not reachable from any agent shell: the cloud container has no
accelerator, and the desktop Cowork workspace is an isolated Linux VM with no
GPU passthrough (no `nvidia-smi`, no `/dev/nvidia*`, 2 cores and 3 GB). The
RTX 4070 Ti lives on the Windows host, outside that sandbox. So this runs natively
on Windows.

## One-time setup

In PowerShell, from the repository root:

```powershell
# uv installs and pins Python 3.13 itself; nothing else is needed.
irm https://astral.sh/uv/install.ps1 | iex

uv sync --python 3.13
```

On Windows the default PyPI wheel for `torch` is **CPU-only**, so `uv sync`
alone leaves the card unused. Measured on this host: the sync installed
`2.14.0+cpu` and `torch.cuda.is_available()` returned `False`. Install the CUDA
build over it, pinning the same version the lockfile resolved:

```powershell
uv pip install --force-reinstall torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
```

Then confirm the card is visible:

```powershell
uv run --no-sync python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Observed here: `2.14.0+cu130 True NVIDIA GeForce RTX 4070 Ti`, compute
capability (8, 9), driver 616.92.

### `--no-sync` is not optional

`uv run` syncs the environment against `uv.lock` before every command, and the
lock pins plain `torch==2.14.0` — which on Windows resolves to the CPU wheel. A
bare `uv run` therefore **uninstalls the CUDA build and reinstalls the CPU one**,
then proceeds on CPU without raising anything. `uv sync --dry-run` shows exactly
that:

```
- torch==2.14.0+cu130
+ torch==2.14.0
```

So every command below uses `uv run --no-sync`. Set `$env:UV_NO_SYNC = "1"` for
the session if you would rather not have to remember it.

The CUDA wheel is deliberately kept out of `pyproject.toml` and `uv.lock`: that
lockfile is shared with the macOS dev host and the Linux confirmatory host per
amendment A1, and a Windows-only index entry would rewrite it for all three.

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

## Run it in gather, not in the default

This is the single thing to know before starting a run on this card. Measured
on the 4070 Ti at the restored 300-step window, forward+backward, batch 1:

| coupling | 200 steps | 250 steps | 300 steps |
|---|---|---|---|
| `sparse` (the default) | 1279 ms, 4291 MB | did not finish in 240 s | did not finish in 240 s |
| `gather` | 183 ms, 650 MB | 229 ms, 806 MB | **274 ms, 961 MB** |

`SpikingPolicy` has always defaulted to `CouplingMode.SPARSE`, and until now
nothing above it could change that, so the pilot could only ever run the mode
that cannot reach the window. The modes are not approximations of each other:
on the real subset at 300 steps they produce byte-identical rasters -- 4,620,000
entries, zero disagreements, final membrane potential differing by exactly 0.0
mV (`data/results/mode_equivalence_v783.json`). The default is left alone so
earlier results stay comparable; pass the flag.

```powershell
uv run --no-sync python -m flywire_rl.run_pilot --coupling gather
```

Batch is nearly free in time and linear in memory: at 300 steps, gather costs
274 ms at batch 1 and 280 ms at batch 8, because the rollout is bound by
kernel-launch latency across sequential steps rather than by arithmetic. Batch
8 is the ceiling on 12 GB. Batch 16 reports a 14,926 MB peak on a 12,282 MiB
card -- a spill to host memory under WDDM, which does not raise and costs 12x
the time -- and batch 32 raises `OutOfMemoryError`. Note that the
pre-registered minibatch is 32, so running at 8 is a documented deviation.

## The run

Three flags are not optional on this card, and the run fails without them:

```powershell
uv run --no-sync python -m flywire_rl.run_pilot --coupling gather --minibatch 8
```

`--coupling gather` because sparse cannot reach the window at all.
`--minibatch 8` because the pre-registered 32 is the batch dimension of a
300-step BPTT and raises `OutOfMemoryError` in the PPO update -- measured, not
predicted: the first attempt died having failed to find two more megabytes.
`--no-sync` because a bare `uv run` reinstalls the CPU wheel over the CUDA one.

Running at a minibatch of 8 is a deviation from the registered `train_kwargs`,
and `run_pilot` records it in the result's `config.train_kwargs` so it cannot
be lost.

Measured end to end with those flags: 3 arms, 1 seed, 300 env steps and 5
evaluation episodes took 415 s and peaked at about 8.1 GB. That run scored
-1.0 on all three arms with zero truncated episodes -- genuine losses by a
policy trained for 300 steps against a registered budget of 500,000, not a
mechanical hang. The distinction is only readable because `provenance` now
carries `truncated_episodes`; a truncated episode scores 0.0, exactly like a
draw.

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
uv run --no-sync python -m flywire_rl.run_pilot --seeds 1 --steps 500 --eval-episodes 5 --only raw --tag probe
```

Watch throughput, and check the drive. On the memory question the guesswork is
over: the numbers above replace the spec's 250 MB BPTT estimate, which assumed
8k neurons and 30 steps.

There is one open problem the probe will not show you. At the production
`INPUT_DRIVE_MV = 20.0` and the restored window, the three arms are not in the
same dynamical regime: active fraction 0.1762 for real, 0.2104 for shuffled and
0.6037 for random, against the 0.25 ceiling `controls.calibrate_w_syn` uses to
decide an arm is transmitting rather than running away. The random control is
2.4x over it. That is a confound, since the arms are supposed to differ in their
edges and in nothing else, and nothing in the run path checks it --
`calibrate_w_syn` raises on exactly this and is never called. See
`data/results/drive_at_window_v783.json`. The drive was chosen at the broken
30-step window, where nothing propagated past the input population and so
nothing downstream could run away.

On throughput: if a seed takes longer than about fifteen minutes, the
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
