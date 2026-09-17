# What the neighbouring projects did, and what they left undone

Read from source on 2026-09-17, from the repositories listed in
[cobanov/awesome-fly](https://github.com/cobanov/awesome-fly). Everything below
is either quoted from those repositories' code or quoted from their own
documentation and marked as their claim. None of it is independently
reproduced.

The short version: **no one has run this experiment.** Each project holds one
or two of the four pieces it needs, and the two that came closest to a topology
comparison report a null or report nothing.

| | topology controls | non-connectome baseline | ablation ladder | outcome |
|---|---|---|---|---|
| cobanov/flyjump | no | a hand-written rule only | **yes, exemplary** | works, claims nothing about topology |
| 5p00kyy/neuroterrarium | yes, five shuffles | no | yes | **null**: arms indistinguishable |
| eganeganegan/flydoom | yes, three kinds | yes, but unmatched budget | no | **no results committed** |
| seanphan/flyt3 | no | claimed, code absent | no | connectome 60.8% vs MLP 85.7% |
| eudald-seeslab/connectome | yes, four ensembles | no | no | vision, not RL; one instance per ensemble |

flyjump states the gap in its own words
(`docs/experiment.md:93`): *"A topology-benefit claim would require matched
artificial/rewired controls trained with equal budgets; that study is not
included."* That study is this one.

## The reference model drives its inputs differently than we do

`philshiu/Drosophila_brain_model` is where our LIF constants come from, and
every one of them matches ours exactly — `v_0` -52 mV, `v_th` -45 mV, `t_mbr`
20 ms, `tau` 5 ms, `t_rfc` 2.2 ms, `t_dly` 1.8 ms, `w_syn` 0.275 mV. The input
does not match. `model.py:84-92`:

```python
for i in exc:
    p = PoissonInput(target=neu[i], target_var='v', N=1,
                     rate=params['r_poi'],                    # 150 Hz
                     weight=params['w_syn']*params['f_poi'])  # 0.275 * 250 = 68.75 mV
    neu[i].rfc = 0 * ms  # no refractory period for Poisson targets
```

Three differences, each load-bearing. One independent Poisson process per
driven neuron (`N=1` inside the loop), so the driven population is
desynchronised from the first step. An event weight of 68.75 mV against a 7 mV
rest-to-threshold gap, so each event forces a spike rather than charging a
membrane. And the refractory period switched off for exactly the neurons being
driven.

The reference has the *same* 1.8 ms delay inside the *same* 2.2 ms refractory
period, so it does not dodge the collision numerically — it dissolves it with
those last two lines. That is what our two `xfail(strict=True)` markers mean by
"the Poisson input the reference model uses, not a tonic current". Filed as
fly-2y4.9.

## The closest method trains only a readout

`liuzihe02/fly-craftax`, which `controls.py` already cites for its liveness
criterion, never puts the substrate in the gradient graph. `ppo.py:12-14`:

```python
def init_params(n_dn, n_actions=7):
    """Zeros, so the initial policy is uniform and the initial value is zero."""
    return dict(W=jnp.zeros((n_dn, n_actions)), b=jnp.zeros(n_actions),
                vw=jnp.zeros(n_dn), vb=jnp.zeros(()))
```

The brain produces rates, the rates enter the batch as ordinary numbers, and a
linear readout is differentiated. No surrogate gradient, no backprop through
time. That is why they run under 1 GB where our 300-step BPTT needs 7.5 GB at
batch 8. Their window is 200 steps at dt 0.1, i.e. 20 ms; ours is 300, i.e. 30.

Our low-rank adapter is deliberately more than they do — it lets the arms
become *differently trainable* rather than only differently informative — and
the whole BPTT bill is the price of exactly that. Worth knowing what we are
buying.

Their calibration rule (`scripts/calibrate.py:48`) is the one ours borrowed:

```python
ok = [r for r in rows if r["nonzero"] >= 4 and r["active"] < 0.25
      and r["l1_black"] > r["l1_full"]]
```

Note their weight grid is `(0.275, 0.44)` — like ours, it searches only at or
above the published value. See fly-2y4.8.

**And their own verdict on their trained policy** (`tracker.md:107`): *"on this
one seed the trained readout is best described as a clock with a small
drive-dependent jitter, not a visually guided policy."* They found that by
replaying it with vision blacked out and with vision frozen: 27 and 22
achievements against 25 for the intact run. Earlier, `tracker.md:99`: *"pure
noop out-survives the wired fly (168 vs 155), so no adaptive control was
shown."*

## The ablation table to copy

`cobanov/flyjump`, `docs/experiment.md:84-93`, on 100 held-out seeds absent
from training and validation, with architecture and weights frozen beforehand:

| Controller | Completed / 100 | Mean survival | Mean score |
|---|---:|---:|---:|
| Connectome + trained readout | **99** | **179.4 s** | **2885.75** |
| Same readout, circuit silenced | 0 | 4.5 s | 41.00 |
| Initial untrained readout | 0 | 4.5 s | 41.00 |
| Hand-written rule baseline | 0 | 46.5 s | 535.27 |
| Uniform random actions | 0 | 4.7 s | 42.77 |
| Idle | 0 | 4.5 s | 41.00 |

Two rows carry the argument: the *same trained weights* with the substrate
silenced, and the *untrained readout* on a live substrate. Together they show
the circuit and the training are both necessary. We have neither row. Filed as
fly-2y4.11.

## The null, and what its authors did about it

`5p00kyy/neuroterrarium` hit our exact symptom. `docs/PHASE3.md:66`:

> Biological and all five shuffled controls each emit 20 total spikes,
> including 10 GF spikes, with identical mean-trace responses despite 502 to
> 561 changed target slots.

Their response was not more seeds. They added coarse perturbations that must
break anything — no recurrence, all-excitatory signs, unit-contact weights,
silencing the stimulated population — which gave 10/0, 20/10 and 0 spikes. That
established the evaluation *could* detect degradation, which is what turned a
tie into a reportable null rather than a broken instrument.

They also interrogate their own null model (`PHASE3.md:64`): *"Swaps are not a
uniform sampler and preserve strong unique-weight connections; mixing is
limited."* We have evidence ours does mix: `decision_window_v783.json` records
2379, 2570 and 6527 spiking neurons for real, shuffled and random.

## Our control construction is not the weak point

`flydoom/src/flydoom/data/controls.py` matches N, edge count and the weight
multiset (Erdős–Rényi) *or* the degree sequence (Maslov–Sneppen) — one property
family at a time — and never touches sign. `neuroterrarium/src/sim/network.ts:96-105`
gates each swap on equal weight and equal source sign, so it preserves degree,
signed degree and weight multisets together; that is roughly our level. Nobody
matches spectral properties. `flyt3` has no topology control at all.

The weak point is elsewhere: seeds, an ablation ladder, and a
parameter-matched baseline. `flydoom`'s baselines claim budget matching in a
docstring (`baselines.py:1`) but are sized from a static `baseline_hidden_size:
128` unrelated to the graph. `flyt3`'s widely quoted 85.7% MLP cannot be
checked for budget matching because its training code is not in the repository.
Sizing a baseline to our actual trainable-parameter count would be stricter
than any of them. Filed as fly-2y4.12.

## Gradient-free training is not available to us at this scale

flyjump's CEM optimises 243 parameters by running an 80-node, 1,296-edge fixed
circuit three times per decision at roughly 11 µs a decision, so 15,680 full
episodes finish in 456 s on a laptop (`public/benchmarks/training.json`). We
have 246,400 parameters in the adapter alone, 15,400 neurons, 300 sequential
steps, and 0.44 s per decision. The sample complexity of a diagonal-Gaussian
CEM grows with dimension and our per-evaluation cost is four orders of
magnitude higher. It becomes an option only if the trained surface shrinks to
something of their size.

Notably, `eudald-seeslab`, working on the same FlyWire v783 at comparable
scale, kept gradients for its per-edge gains (`configs/config.py:11-13`,
`train_edges = True`). Freezing the wiring is standard; freezing the training
method is not.
