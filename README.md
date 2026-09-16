# flyhs — does real connectome topology help a spiking network make decisions?

A FlyWire-derived spiking substrate learns to play a small, hidden-information
card game, and is compared against **strictly matched** shuffled and random
topologies. The question is not "can a fly brain play cards" but whether the
*real wiring* measurably outperforms controls that share its degree sequence,
weights, Dale signs and transmission rate.

The pre-registered protocol, every amendment, and every measurement that refuted
an assumption live in
[`.omc/specs/deep-interview-flywire-hearthstone.md`](.omc/specs/deep-interview-flywire-hearthstone.md).
Read it before changing anything: several apparently sensible choices in it were
already measured and found wrong, and the reasons are recorded.

## Setup

Requires Python 3.13 (pinned: `torch`, `brian2` and `scipy` all ship cp313
wheels for macOS arm64 *and* linux x86_64, which amendment A1 needs).

```bash
uv sync --python 3.13
uv run pytest -q          # 146 tests
```

The FlyWire source tables are not committed — they are large and carry their own
licence. Fetch them into `data/raw/`:

```bash
B=https://storage.googleapis.com/flywire-data/codex/data/fafb/783
mkdir -p data/raw
for f in connections.csv.gz classification.csv.gz neurons.csv.gz cell_stats.csv.gz; do
  curl -sS -o "data/raw/$f" "$B/$f"
done
```

`data/manifest.json` pins the SHA-256 of each file and the loader refuses to run
on a mismatch, so a silently different release cannot slip into a result.

## Layout

| Module | What it does |
|---|---|
| `connectome.py` | FlyWire v783 loading, the P1 subset rule, manifest, Dale signs |
| `controls.py` | Shuffled and random arms, branching-ratio calibration, graph stats |
| `spiking.py` | LIF with surrogate gradients; low-rank adapter masked to each arm's own edges |
| `minicard.py` | The environment: 10 cards, hidden hands, legality-masked actions |
| `policy.py` | Observation encoder, hierarchical action decoder, spiking policy |
| `ppo.py` | PPO with GAE, alternating seats and opponents |
| `analysis.py` | IQM, stratified bootstrap, probability of improvement, Holm |
| `experiment.py` | Builds arms, trains seeds, evaluates, records provenance |
| `bench.py` | Throughput probe |

## Where it stands

The pipeline runs end to end. A **pilot** (2 seeds, 1/333 of the pre-registered
budget) gives real +0.025 > shuffled -0.450 > random -1.000, and only the
comparison against *random* is significant.

**That is not a result.** Two seeds is far below the power measured for five;
the random arm has zero variance across seeds, so its interval is degenerate;
and the ordering reproduces the representational ordering measured *before any
learning*, which means it may reflect how much signal reaches the readout rather
than better computation. Separating those two is the next task.

## Next

1. **Amplitude control** — standardise readout features per arm and re-run. If
   the ordering survives, the effect is computational; if it vanishes, it is
   transmission. This decides what the confirmatory series is even asking.
2. **CUDA host** — verify the lockfile there, and close the cross-device
   agreement gate amendment A1 requires (identical spike trains on `mps` and
   `cuda`).
3. **Confirmatory series** — 10 seeds x 3 arms at 5e5 steps. Roughly 466 h on an
   M1 Pro, so it belongs on the desktop.

Issues are tracked with [beads](https://github.com/gastownhall/beads) (`bd ready`).
