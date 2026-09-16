# Deep Interview Spec: FlyWire Topology in a Hearthstone-Like Card Game

## Metadata
- Interview ID: `deep-interview-flywire-hearthstone`
- Rounds: 7 scored rounds plus Round 0 topology gate
- Final Ambiguity Score: 14.1%
- Type: greenfield
- Generated: 2026-09-15
- Threshold: 0.2
- Threshold Source: `default`
- Initial Context Summarized: yes
- Status: **PROTOCOL LOCKED** (2026-09-16) — all six pre-registration items fixed; see "Pre-Registered Protocol"
- Protocol Amendments: **A1** (2026-09-16) — two-host execution model; see "Protocol Amendments"

## Clarity Breakdown

| Dimension | Score | Weight | Weighted |
|---|---:|---:|---:|
| Goal Clarity | 0.85 | 0.40 | 0.340 |
| Constraint Clarity | 0.88 | 0.30 | 0.264 |
| Success Criteria | 0.85 | 0.30 | 0.255 |
| **Total Clarity** |  |  | **0.859** |
| **Ambiguity** |  |  | **0.141 / 14.1%** |

The greenfield formula is `ambiguity = 1 - (goal * 0.40 + constraints * 0.30 + criteria * 0.30)`.

## Topology

Round 0 confirmed four active, independent components. No component was deferred.

| Component | Status | Description | Coverage / Deferral Note |
|---|---|---|---|
| FlyWire substrate and reproducible connectome data | active | Select and reproduce a functional connectome subset, annotations, weights, delays, and provenance. | Functional subset selected in Round 3. The exact subset selection rule and data release must be pre-registered. |
| SNN simulation and learning | active | Use spiking dynamics with a primary low-rank adaptation method whose graph-preserving parameterization is explicitly controlled. | Low-rank adaptation made primary in Round 5. Adapter placement, rank, and parameter budget remain protocol parameters to specify before experiments. |
| Hearthstone-like environment and interfaces | active | Build an independent toy card game with partial observation, observation encoding, hierarchical action decoding, legality, and curriculum stages. | Toy card game selected in Round 2. It is the first publishable experimental environment, not full Hearthstone. |
| Evaluation, datasets, baselines, and ablations | active | Test topology value against strict matched controls using game return, sample efficiency, neural, biological, and compute metrics. | Primary claim and strict matching contract selected in Rounds 1 and 7. |

## Goal

Determine whether a real FlyWire-derived functional connectome topology, used as the recurrent substrate of a spiking neural system with a primary low-rank adaptation mechanism, provides measurable value for sequential decisions in a partially observable toy card game compared with strictly matched shuffled and random topology controls.

The first publishable claim is about **topology value**, not about full Hearthstone competence, human-level play, or a claim that the system is a literal biological fly brain. Full FlyWire-scale simulation and a progression toward richer Hearthstone subsets are later scale-up stages.

## Constraints

- This is a research/neuroscience experiment, not an ordinary LLM-agent project.
- The first publishable environment is an independently implemented and validated toy card game.
- The first neural substrate is a functional FlyWire-derived subset, not an assumption that the complete approximately 140k-neuron brain is a practical first training target.
- Real, shuffled, and random topology arms must be strictly matched. The matching protocol must preserve, where applicable, degree/edge-count, excitatory/inhibitory sign, weight distribution, and other declared graph statistics.
- All topology arms must use the same neuron dynamics, input/output interface, adapter rank and parameterization, initialization family, optimizer, random-seed protocol, number of trainable parameters, sample budget, and compute budget.
- The recurrent substrate must remain identifiable as the manipulated factor. Any trainable low-rank adapter must have an explicitly reported location, rank, parameter count, and effect on the base graph.
- The environment must expose only the declared partial observation. Hidden opponent information must not leak through the API, logging, seeding, or action legality helpers.
- Action selection must be hierarchical or autoregressive rather than an uncontrolled flat Cartesian action space.
- Reward shaping must be documented and tested for reward hacking; the primary result must include the environment's terminal objective.
- Training, evaluation, seeds, data versions, graph construction, simulator parameters, and hardware/software versions must be reproducible.
- Public statistics or card metadata may be used only within their actual access and license boundaries. Unavailable or restricted replay data must not be assumed.

## Non-Goals

- Full end-to-end reinforcement learning on the complete Hearthstone game as the MVP or first publishable result.
- Claiming that a reduced artificial circuit is biologically equivalent to a living fly brain.
- Using HSReplay as an assumed source of unrestricted private replay sequences or hidden player data.
- Treating card metadata libraries as a complete game simulator.
- Optimizing only for peak game score while ignoring topology controls, matched budgets, learning curves, neural dynamics, and failure modes.
- Allowing a learned adapter to vary independently between topology arms in a way that confounds the topology comparison.
- Presenting a single successful seed as evidence of a topology effect.

## Acceptance Criteria

### Environment and interfaces

- [ ] The toy card game has a written rules specification covering state, turn/phase transitions, legal actions, terminal conditions, hidden variables, and reward.
- [ ] The environment passes deterministic transition tests, legality tests, terminal/reward tests, and hidden-information leakage tests.
- [ ] Observation encoding and hierarchical action decoding are versioned interfaces independent of the neural implementation.
- [ ] The environment supports fixed evaluation seeds and held-out generalization conditions.

### FlyWire substrate

- [ ] A versioned data manifest records FlyWire release/version, source URLs or identifiers, selected neuron/cell-type subset, neuropil or functional rationale, filtering rules, edge inclusion rules, sign/weight handling, delays, and licensing/access assumptions.
- [ ] The real topology can be rebuilt from the manifest without manual GUI state.
- [ ] The functional subset has a machine-readable graph summary: node count, edge count, in/out-degree distributions, weight/sign distributions, connected components, and input/output interface nodes.
- [ ] Shuffled and random control generators produce auditable graph summaries and pass the pre-registered matching checks.

### SNN and low-rank adaptation

- [ ] The simulation backend reproduces the selected neuron and synapse dynamics deterministically for fixed inputs and seeds.
- [ ] The low-rank adapter is documented with its insertion point, factor shapes, rank, initialization, constraints, parameter count, update rule, and whether base recurrent weights remain frozen.
- [ ] Real, shuffled, and random topology arms use the same adapter contract and trainable parameter budget.
- [ ] The implementation records spikes, adapter parameters or sufficient statistics, episode outcomes, wall-clock time, GPU memory, and configuration hashes.
- [ ] At least one frozen-reservoir/readout or otherwise topology-isolating control is retained as a secondary ablation, so the primary result can distinguish graph value from adapter capacity.

### Evaluation and scientific claim

- [ ] The primary analysis reports mean terminal return and sample-efficiency curves under a fixed sample and compute budget.
- [ ] Results include multiple independent seeds, confidence intervals or bootstrap intervals, and a pre-declared statistical comparison between real topology and each matched control.
- [ ] The evaluation reports learning stability, held-out seed/rule/opponent generalization, legality rate, episode length, and reward decomposition.
- [ ] Neural metrics include firing-rate distributions, sparsity, synchrony, temporal activity, representation similarity, and adapter-induced changes.
- [ ] Biological/structural metrics include preservation of declared FlyWire graph statistics and a clearly labeled biological plausibility score with its rubric.
- [ ] Compute metrics include simulator throughput, episode latency, peak GPU memory, host RAM, and reproducibility on the target RTX 4070 Ti / Ryzen 9 7900X / 32 GB RAM class machine.
- [ ] Required ablations cover real versus random topology, real versus shuffled connectivity, homogeneous versus annotated neuron types where available, frozen versus adapted weights, and functional subset versus scale-up variants.
- [ ] The conclusion states whether the topology effect is supported, absent, or inconclusive; it must not convert a non-significant result into a biological claim.

## Assumptions Exposed and Resolved

| Assumption | Challenge | Resolution |
|---|---|---|
| Full FlyWire plus full Hearthstone is the natural first target. | Compute, interface, and credit-assignment costs would confound the first result. | Use a functional subset and toy card game first; treat full-scale work as later feasibility/scale-up. |
| A score improvement alone proves that topology matters. | A larger or more trainable adapter could explain the improvement. | Use strict matched controls and report a topology-isolating frozen/readout ablation. |
| A flat Hearthstone action space is acceptable. | It creates avoidable combinatorial exploration and credit-assignment problems. | Use hierarchical/autoregressive actions with explicit legality masking. |
| Low-rank adaptation is only a later optional trick. | The user selected it as the primary method, which changes the causal design. | Make it primary, but fully pre-register its graph interaction and parameter budget. |
| HSReplay can supply all required behavior data. | Public availability, API stability, licensing, and replay granularity may be limited. | Treat data access as an explicit constraint; do not depend on restricted replay data for the first experiment. |

## Technical Context

The workspace is greenfield with respect to product/research code. Existing repository files are instruction and task-tracking files; there is no application implementation to extend.

The prior research identified `eonsystemspbc/fly-brain` as a reusable simulation foundation rather than a completed learning system. Its GeNN path provides sparse recurrent connectivity, delayed synapses, batched simulation, CUDA execution, bounded spike recording, Parquet export, and benchmark/ground-truth comparison patterns. It does not currently provide the toy game, observation encoder, action decoder, RL loop, low-rank adapter, partial-observation memory protocol, or topology-control experiment required here.

Relevant external foundations include FlyWire/Codex connectome and annotation data, CAVE-style data access, Brian2/GeNN-compatible spiking simulation patterns, and HearthSim card/deck metadata. These are inputs or implementation references, not evidence that a complete environment or unrestricted replay corpus is already available.

## Experimental Design

### Arms

1. **Real topology:** the declared FlyWire functional subset.
2. **Connectivity shuffle:** a topology-preserving shuffle under the declared matching contract.
3. **Matched random topology:** a random graph generated under the same declared graph-statistic constraints.
4. **Optional secondary controls:** homogeneous neuron annotations, frozen reservoir plus readout, and other pre-registered controls.

### Learning contract

The primary method is a low-rank adapter over a frozen or partially frozen spiking substrate, with the exact placement and rank specified before the run. Every arm receives the same adapter architecture and parameter budget. A frozen-reservoir/readout arm remains necessary as a secondary causal check because otherwise adapter learning may dominate the observed result.

### Curriculum

- Stage 0: unit tests and deterministic toy-game transitions.
- Stage 1: neural sanity checks on synthetic event streams and controlled memory tasks.
- Stage 2: fully observable toy card game.
- Stage 3: partial observation, hidden opponent state, and delayed consequences.
- Stage 4: held-out rule/deck/opponent variations.
- Later: richer mini-Hearthstone subset and only then evaluation of whether a bridge to real Hearthstone tooling is scientifically justified.

### Primary analysis

The primary comparison is the difference in mean terminal return and sample-efficiency area-under-the-learning-curve between real topology and each strict matched control at a fixed compute/sample budget. The analysis must include multiple seeds and uncertainty intervals. A topology result is supported only if the pre-registered comparison survives the declared statistical and reproducibility checks; otherwise it is reported as absent or inconclusive.

## Metrics

### Game

- Terminal return and win/loss or task-success rate.
- Sample efficiency and area under the learning curve.
- Legality rate, invalid-action attempts, episode length, and reward decomposition.
- Performance on held-out seeds, rules, card distributions, and opponent policies.

### Learning

- Convergence rate, variance across seeds, instability, catastrophic forgetting, and adapter norm/rank utilization.
- Training samples, optimization steps, wall-clock time, and compute-normalized progress.

### Neural

- Firing-rate distribution, population sparsity, active-neuron fraction, synchrony, temporal precision, and recurrent activity stability.
- Readout/adapter representation similarity across topology arms and across curriculum stages.
- Working-memory or predictive-state task performance under partial observation.

### Structural and biological

- Degree, motif, component, sign, weight, delay, and cell-type statistics relative to the declared FlyWire source.
- A 0-10 biological plausibility score with separate axes for anatomy/topology, neuron dynamics, synapse dynamics, learning rule, temporal scale, and interface assumptions. The score is descriptive, not a substitute for evidence of biological equivalence.

### Compute and reproducibility

- Throughput, episode latency, peak VRAM, peak RAM, compile time, spike-recording cost, and scaling with node/edge count.
- Source/data/configuration hashes, software versions, hardware description, random seeds, and exact commands.

## Ontology (Key Entities)

| Entity | Type | Fields | Relationships |
|---|---|---|---|
| Connectome | core domain | release, nodes, edges, annotations, provenance | Connectome contains NeuralCircuit and Synapse records |
| NeuralCircuit | core domain | subset, topology, dynamics, interface | NeuralCircuit contains Neurons and is assigned a TopologyControl |
| Neuron | core domain | id, type, region, dynamics, sign | Neuron participates in Synapses and emits spikes |
| Synapse | core domain | source, target, weight, delay, sign | Synapse connects Neurons |
| Environment | core domain | rules, hidden state, seed, version | Environment emits Observations and accepts Actions |
| Observation | core domain | visible state, event encoding, timestamp | Observation is provided to Policy |
| Action | core domain | hierarchy, arguments, legality | Action is selected by Policy and applied to Environment |
| Policy | core domain | encoder, SNN, adapter, decoder | Policy maps Observations to Actions |
| Reward | core domain | immediate value, terminal value, shaping terms | Environment emits Reward to a TrainingRun |
| Episode | core domain | seed, trajectory, outcome, length | Episode belongs to a TrainingRun |
| Baseline | evaluation entity | architecture, parameters, contract | Baseline is compared with NeuralCircuit arms |
| Ablation | evaluation entity | changed factor, control, hypothesis | Ablation isolates a causal factor |
| Metric | evaluation entity | definition, aggregation, uncertainty | Metric evaluates Episodes, TrainingRuns, and NeuralCircuits |
| Dataset | external/supporting | version, source, license, manifest | Dataset supplies Connectome or evaluation inputs |
| TrainingRun | evaluation entity | seed, config hash, budget, artifacts | TrainingRun produces Episodes and Metrics |
| CurriculumStage | supporting | stage id, rules, unlock criteria | CurriculumStage constrains Environment and TrainingRun |
| TopologyControl | evaluation entity | kind, matching statistics, seed | TopologyControl constructs a NeuralCircuit arm |
| LowRankAdapter | core method entity | rank, factors, placement, constraints | LowRankAdapter modifies Policy parameters under a fixed contract |

## Ontology Convergence

| Round | Entity Count | New | Changed | Stable | Stability Ratio |
|---:|---:|---:|---:|---:|---:|
| 1 | 15 | 15 | 0 | 0 | N/A |
| 2 | 16 | 1 | 0 | 15 | 93.75% |
| 3 | 17 | 1 | 0 | 16 | 94.12% |
| 4 | 18 | 1 | 0 | 17 | 94.44% |
| 5 | 18 | 0 | 0 | 18 | 100% |
| 6 | 18 | 0 | 0 | 18 | 100% |
| 7 | 18 | 0 | 0 | 18 | 100% |

## Pre-Registered Protocol (LOCKED 2026-09-16)

All six previously open items are now fixed. Any deviation after this point must be recorded as a labeled protocol amendment with its date and rationale, and any run performed under a superseded protocol must be reported as exploratory, not confirmatory.

### P1. Connectome data and subset selection rule

| Item | Value |
|---|---|
| Release | FlyWire FAFB **v783** (Codex/CAVE public materialization) |
| Pinned tables | `root_id` @ v783; synapse table (`pre_pt_root_id`, `post_pt_root_id`, `syn_count`); Schlegel et al. cell-type annotations; neurotransmitter predictions |
| Provenance | Every source file hashed SHA-256 into `data/manifest.json`; loader refuses to run on hash mismatch |
| Edge threshold | `syn_count >= 5` (FlyWire noise floor) |
| Seed neuropils | **MB** (CA, PED, ML, VL) + **CX** (EB, FB, PB, NO) |
| Membership rule | Neuron included if its **majority-synapse neuropil** is in the seed set |
| Interface layer | Plus all 1-hop neighbours with >= 5 synapses to/from the seed set |
| Size | **Reported outcome of the rule, not a target.** If over compute budget, truncate by total `syn_count` and report the cutoff explicitly |

Rationale for MB+CX: the mushroom body is a sparse-coding associative memory under dopaminergic modulation; the central complex is a ring-attractor system for working memory and action selection. Together they are the functional prior a partially observable card game actually requires. MB alone is nearly feedforward (KC -> MBON), which would leave no recurrent substrate for working memory and could drive the topology effect to zero by construction.

### P2. Environment - MiniCard v1

| Parameter | Value |
|---|---|
| Players | 2, alternating turns |
| HP | ~~20 / 20~~ **30 / 30** (amendment A2) |
| Mana | `min(turn, 10)`, refreshed each turn |
| Deck | 30 cards from a fixed 10-type pool; composition set by archetype seed |
| Hand | start 3 / 4; max 7; draw 1 at turn start; fatigue on empty deck |
| Minions | 6 types, cost 1-5, stats ~ cost x [1.0 .. 1.5]; summoning sickness; 1 attack/turn |
| Spells | 4 types: 3 dmg / cost 2; AoE 2 / cost 3; heal 5 / cost 2; draw 2 / cost 2 |
| Board cap | 5 per side |
| Turn limit | 30 turns -> draw, return 0 |

Keywords (taunt, charge, deathrattle) are **deliberately absent in v1** and reserved as held-out rule variations for Stage 4.

**Observation (single engine, mask toggled):**

- Stage 2 (full): own hand, both boards, both HP, both mana, deck counts, **opponent hand contents**, turn index.
- Stage 3 (partial): opponent hand contents masked to a count only. Deck order never observable in either stage.

**Action space - hierarchical, autoregressive, legality-masked at every level:**

```
L1: {play_card, attack, end_turn}
L2: source  (hand 0-6 | board 0-4)
L3: target  (enemy_face | enemy_minion 0-4 | own_minion 0-4 | none)
```

**Reward.** The primary metric is the **unshaped terminal return**: +1 win / -1 loss / 0 draw. Training may use shaping, but only in potential-based form `r' = r + gamma*Phi(s') - Phi(s)` with `Phi = (HP_own - HP_opp)/20 + 0.1*(stats_own - stats_opp)`, which leaves the optimal policy invariant. Both shaped and unshaped curves are reported.

**Held-out generalization (Stage 4):** train on deck archetypes A,B -> test on C,D; introduce `taunt` at test time only; swap scripted opponent (greedy-face -> board-control) and evaluate against a self-play snapshot.

### P3. Learning contract

**Adapter is masked to each arm's own support.** A dense `Delta_W = BA` would add all-to-all connections and erase exactly the inter-arm difference under measurement. Therefore:

```
Delta_W_rec = (B @ A) * M_arm        B in R^{N x r},  A in R^{r x N}
```

`M_arm` is the binary adjacency mask of that arm, and `*` is elementwise. Trainable parameter count `2Nr` is identical across arms by construction, while the expressible delta stays confined to each arm's own graph.

| Item | Value |
|---|---|
| Primary rank | **r = 8** |
| Rank sweep | r in {2, 8, 32}, secondary, 5 seeds x 2 arms (real, shuffled) |
| Init | `B = 0`, `A ~ N(0, 1/sqrt(N))` -> `Delta_W = 0` at t=0; substrate starts as pure connectome |
| Input/output nodes | Identical **index sets** across arms (edges are shuffled, nodes are not); linear encoder -> current injection; linear readout |
| Frozen | Recurrent `W_rec` weights, neuron parameters |
| Optimizer | AdamW; lr 3e-4 (adapter) / 1e-3 (readout); cosine decay; grad-clip 1.0 |
| RL algorithm | PPO, actor-critic |
| Primary learning rule | **Surrogate-gradient truncated BPTT** (fast-sigmoid, beta=10) |
| Secondary learning rule | **e-prop / three-factor eligibility traces** - tests whether the topology effect survives a biologically plausible rule |

v1 does not claim biologically plausible learning; it tests it as a secondary arm.

### P4. Topology controls and matching tolerances

**Shuffled (degree-preserving double-edge swap):**

| Property | Tolerance |
|---|---|
| Per-node in-degree and out-degree | **0** (hard) |
| Edge count | **0** |
| Weight multiset | **0** (weights travel with edges) |
| E/I sign per **presynaptic** neuron | **0** - Dale's law: a neuron cannot become half-excitatory |
| Swap count | 100 x abs(E) |

Clustering, assortativity and reciprocity **will** change - that is the measured difference. Reported before and after.

**Random:** matched on N, edge count (relative delta <= 1e-3), E/I fraction (<= 1 pp), and weight distribution (KS <= 0.01). Degree sequence deliberately not preserved.

**Mandatory gain normalization - two levels.** If the real graph has spectral radius 3 and the ER control has 0.5, one substrate saturates and the other is silent, and "topology matters" becomes an artifact of gain rather than structure.

1. **Structural:** global weight rescale so `rho(W_rec) = 0.95` in every arm.
2. **Dynamical:** calibrate on a fixed reference input to population rate **5 +/- 0.5 Hz** in every arm before training begins.

Both are reported; disagreement between them is itself informative.

### P5. Statistics and budget

| Item | Value |
|---|---|
| Seeds | **10 per arm**; 4 arms x 10 = 40 confirmatory runs |
| Sample budget | **5e5 environment steps per run**, identical across arms |
| Compute budget | ~0.7 GPU-h per run; ~30 GPU-h total on one RTX 4070 Ti |
| Escalation rule | If the pilot shows learning curves have not plateaued by 5e5, the budget is raised to 1e6 and **all** arms are re-run. Partial escalation of a subset of arms is forbidden |
| Primary statistic | **IQM of terminal return** with **stratified bootstrap** over seeds (Agarwal et al. 2021), 95% CI, plus probability of improvement |
| Primary hypothesis | `H0: IQM(real) - IQM(shuffled) <= 0`; rejected if the 95% bootstrap CI of the difference excludes 0 |
| Secondary comparison | real vs. random, same procedure |
| Multiplicity | Holm correction over the 2 primary comparisons, alpha = 0.05 |

Bare t-tests on means are excluded: RL outcome distributions are heavy-tailed and seed-dominated.

### P6. Simulator backends

| Role | Backend |
|---|---|
| Learning runs | **PyTorch CUDA**, custom LIF + surrogate gradient; dt = 1.0 ms; 30 ms decision window; truncated BPTT over the window; batch 64 environments |
| Cross-check | **Brian2 CPU**, identical parameters, no learning; PyTorch LIF must reproduce reference spike trains with >= 95% spike agreement within +/- 1 dt on a fixed stimulus |
| Reserved | GeNN, forward-only scaling beyond v1 |

Brian2 and GeNN are not candidates for the learning run: neither provides autodiff through spike generation.

Memory estimate: 8k neurons x 64 batch x 30 steps x 4 state variables ~ 250 MB for the BPTT graph, comfortably inside 12 GB.

## Protocol Amendments

### A1 — Two-host execution model (2026-09-16)

**Trigger.** Hardware audit of the machine this work is being developed on.

**Finding.** The development host is an **Apple M1 Pro, 8 cores, 16 GB unified memory, arm64, no CUDA** (Python 3.14.7; torch 2.14.0 available for this interpreter via venv). The **RTX 4070 Ti / Ryzen 9 7900X / 32 GB** machine named in P5 and P6 is a **separate desktop host**, confirmed available. P6 as written implied a single CUDA machine and was therefore false about the development environment.

**Change to P6 — backend becomes device-agnostic:**

| Role | Host | Device |
|---|---|---|
| Confirmatory runs (P5 budget: 40 runs) | CUDA desktop | `cuda` |
| Development, unit tests, Stage 0/1, pilots | M1 Pro laptop | `mps`, fallback `cpu` |
| Brian2 CPU spike-train cross-check | either host | `cpu` |
| GeNN / Brian2CUDA (reserved, forward-only) | **CUDA desktop only** | `cuda` |

GeNN and Brian2CUDA do not exist on the development host. Any claim depending on them is confirmatory-host-only.

**Two new gates introduced by this amendment:**

1. **Cross-device agreement.** Identical seed and identical input must produce matching spike trains on `mps` and `cuda` within the same tolerance already required of the Brian2 cross-check (>= 95% spike agreement within +/- 1 dt). Until this passes, results measured on one device may not be compared with results measured on the other.
2. **Dense/sparse equivalence.** MPS lacks efficient sparse matmul, so development runs carry `W_rec` dense while confirmatory runs may use a sparse kernel on CUDA. The dense and sparse code paths must be verified numerically equivalent on a fixed stimulus before any confirmatory run. Otherwise "topology matters" could reduce to "the two hosts computed different things."

**New reproducibility requirement.** A single pinned dependency lockfile is shared by both hosts; resolved package versions, device, driver and host identifier are recorded in every run's metadata alongside the `data/manifest.json` hashes from P1.

**Compute note (informational, not a protocol change).** On M1 Pro with `N ~ 8k` and dense `W_rec` (256 MB), a pilot run is estimated at roughly 1 h — same order as the 4070 Ti figure in P5, because the dense path discards the sparsity advantage that CUDA can exploit. At `N ~ 20k`, dense `W_rec` reaches 1.6 GB and laptop runs become unviable under thermal throttling. This reinforces the existing P1 risk: subset size is an outcome of the rule, and if it lands far above the assumed scale, P5 requires its own amendment **before** any confirmatory run.

**Unchanged:** P1, P2, P3, P4, P5.

### A2 — Starting HP raised from 20 to 30 (2026-09-16)

**Trigger.** Sanity simulation of the implemented MiniCard v1 engine, before any
training run.

**Finding.** At P2's original 20 HP the environment is strategically degenerate.
Measured over 300-400 scripted games per cell:

| Matchup (HP 20) | P0 win | Median turns |
|---|---:|---:|
| greedy-face vs board-control | 88.0% | 7 |
| greedy mirror | 74.3% | 5 |
| board-control mirror | 51.0% | 12 |

Racing beats trading 88%, so the optimal policy is "attack the face", and games
end by turn 5-7. An environment whose optimum is that shallow gives every arm
the same ceiling and leaves no room for a topology effect to appear. Worse for
this specific experiment, a 5-turn game places no demand on working memory,
which is the capability the central complex was selected into the substrate to
provide (R1). The environment would not have exercised the hypothesis.

An initial diagnosis blamed P2's deliberate removal of `taunt`, on the theory
that an unprotectable face makes board control worthless. Measurement refuted
this: the board-control mirror is 51.0% and runs 12 turns, so the game itself is
sound. The 74% skew was an artefact of the greedy bot's pure-race policy, not a
first-player advantage in the rules.

A sweep isolated a crossover: aggro dominates below ~25 HP, control dominates
above it.

**Change.** `START_HP` 20 -> 30. Nothing else in P2 changes; keywords stay
reserved for Stage 4.

**Verification at 30 HP** (400 games per cell):

| Matchup (HP 30) | P0 / P1 | Median turns | Median actions |
|---|---|---:|---:|
| control mirror | 39.2 / 60.8 | 22 | 139 |
| aggro mirror | 48.8 / 51.2 | 15 | 112 |
| control vs aggro | 90.0 / 10.0 | 11 | 85 |
| aggro vs control | 51.8 / 48.2 | 10 | 72 |

The strategic policy now dominates the shallow one, and games run 10-22 turns
over 72-139 decisions.

**Residual imbalance, recorded not hidden.** The control mirror favours the
second player 60.8%, because P2's extra starting card for player two is worth
more in slow games than in fast ones. Seats must therefore be alternated during
training and evaluation, and per-seat win rates reported separately; this is not
corrected by a rules change.

**Timing.** Applied before any training run, so no result was produced under the
superseded value.

### A3 — Branching ratio replaces the spectral radius (2026-09-16)

**Trigger.** The first pilot on the real substrate returned byte-identical
scores for all three arms (-0.933 each, both seeds). Independent graphs with
independently seeded policies cannot agree by chance.

**Finding.** At P4's `rho = 0.95` the substrate is not a network. Mean synaptic
magnitude is 0.0132, the membrane integrates at `dt/tau = 0.05`, so one
presynaptic spike moves the potential by 6.6e-4 against a threshold of 1.0 —
about **1,515 coincident inputs** would be needed to fire a neuron whose mean
in-degree is 20. Measured directly: of 15,710 spikes in a policy rollout,
**100% were in directly-driven input neurons and none propagated**. With the
recurrent graph transmitting nothing, topology cannot affect anything, and the
pre-registered experiment was guaranteed to return a null for reasons having
nothing to do with the connectome.

A spectral radius near 1 is the right criterion for a linear *rate* network,
where the question is whether activity decays or explodes. It is the wrong scale
for leaky integrate-and-fire units with a hard threshold. A sweep located the
onset of transmission at 100-200x the `rho = 0.95` weights; calibration put the
matched point at **1,780x**, i.e. P4's target was off by three orders of
magnitude.

**Change.** The structural match becomes the **branching ratio**: each arm's
weights are scaled so that one spike produces on average one further spike,
measured while the network runs with **no external drive at all**. This is the
spiking analogue of unit gain, and it is the regime where structure most
influences dynamics. Spectral radius is still reported, as an observable.

Each arm receives its *own* scale. Matching the scale instead would leave arms
transmitting at different rates — a larger difference than the topology under
test.

### A4 — Membrane readout; the input-gain calibration is removed (2026-09-16)

**Trigger.** Fixing A3 exposed a second blocker: output neurons still produced
no spikes, and P4's 5 Hz input calibration stopped converging.

**Finding, after two wrong hypotheses.** It is not distance — the 44 descending
neurons sit 1 to 3 hops from the input set, 25 of them one hop, none
unreachable. It is not gain — raising the input a thousandfold changed nothing.
The cause is sparsity: only ~534 of 15,400 neurons spike, so a neuron with
in-degree 41 sees roughly 1.4 active inputs and never reaches threshold.

Readout variants were then compared on whether they *discriminate game states*
(standard deviation across 16 distinct positions), not merely whether they are
non-zero:

| Branching | Readout | Units that vary | Median s.d. |
|---|---|---:|---:|
| 0.70 | spike counts | 0.02-0.20 | **0.00** |
| 0.70 | membrane | 0.59-0.99 | 0.014-0.071 |
| 1.00 | spike counts | 0.64-1.00 | 0.48-1.34 |
| 1.00 | membrane | 0.98-1.00 | **1.12-2.97** |

A spike-count readout leaves the median output unit with exactly zero variance
unless the network sits at criticality. The membrane is graded and varies
everywhere.

**Change.** The readout becomes the **mean pre-reset membrane potential** of the
output neurons. The 5 Hz input calibration is **removed**: once branching is
fixed the firing rate is set by the network rather than by the input, which
makes that calibration ill-posed, and a single constant gain shared by all arms
replaces it. The output population stays the biologically motivated descending
and motor neurons; enlarging it to 256 units did not help.

**Cost, stated plainly.** The claim weakens from "the fly's descending neurons
discharge to drive the action" to "the readout is their sub-threshold
potential". This was chosen over criticality-dependent spike readout because a
membrane readout keeps working if the adapter drifts the branching ratio during
training, whereas a spike readout would silently go dead.

**Verification on the real subset after both amendments:**

| Arm | Branching | Units that vary | Median s.d. | Feature RMS |
|---|---:|---:|---:|---:|
| real | 1.011 | 1.00 | 70.10 | 1509.9 |
| shuffled | 1.002 | 1.00 | 13.72 | 838.7 |
| random | 1.000 | 1.00 | 4.88 | 242.2 |

Every readout unit now varies with the game state, and the arms produce plainly
different representations (mean absolute difference 384 to 866). **This is not a
topology result:** no learning has occurred, and the amplitude rises together
with the variance (relative s.d. 0.046 / 0.016 / 0.020), so the ordering may
reflect gain rather than richer representation. It establishes only that the
substrate is no longer degenerate and the experiment can now be run.

## Probe Results and Clarifications

These are outcomes of executing the protocol, plus disambiguations of prose that
turned out to be underdetermined. They are **not** amendments: no pre-registered
decision changed. Recorded here because they are material and must not be
rediscovered later.

### C1 — P1 file identity (clarification, 2026-09-16)

The Codex v783 bucket ships **nine** `connections*` variants (`princeton`,
`buhmann`, `no_threshold`, `ol_min_2`, ...). P1 named a threshold without naming
a file, which leaves "syn_count >= 5" meaningless. The canonical Codex export
`connections.csv.gz` is used, pinned by SHA-256 in `data/manifest.json` together
with `classification.csv.gz`, `neurons.csv.gz` and `cell_stats.csv.gz`.

### C2 — P1 threshold semantics (clarification, 2026-09-16)

The bulk CSV holds one row per **(pre, post, neuropil)** triple, and those rows
routinely fall below 5 synapses (observed minimum: 1). The threshold applies to
the **aggregated neuron-to-neuron pair**, per FlyWire convention. On v783 all
2,700,513 aggregated pairs already clear 5, so the filter is a no-op on this
release; it is retained explicitly so the rule stays portable to a release where
it is not.

Correspondingly, "majority-synapse neuropil" counts each neuron's presynaptic
**and** postsynaptic mass. Both points are pinned by tests in
`tests/test_connectome.py`.

### R1 — P1 measured outcome (result, 2026-09-16)

| Quantity | Value |
|---|---|
| Seed (majority neuropil in MB+CX) | 7,918 |
| 1-hop interface | 7,482 |
| **N** | **15,400** |
| **\|E\|** | **310,867** |
| Density | 0.00131 |
| Excitatory / inhibitory / modulatory | 10,795 / 3,312 / 1,293 (E fraction 0.765) |
| Degrees | mean 20.2, median 8-9, p95 ~72, max 2,763 |
| Largest SCC | 14,841 (96.4% of N) |
| rho(W) on raw synapse counts | 1,091.8 |

The interface layer resolves to 6,353 `central`, 581 `sensory`, 433
`visual_projection`, 39 `descending`, 24 `ascending`, 2 `motor` and only 13
`optic` neurons: a genuine sensory-to-central-to-motor shell rather than optic
lobe bloat. The 96.4% single strongly-connected component confirms the MB+CX
choice over MB-only, which the interview flagged as risking a near-feedforward
substrate with nowhere for working memory to live.

The P4 normalization to `rho = 0.95` is therefore a rescale by a factor of
~1,150. It is load-bearing, not cosmetic.

### R2 — N is 1.9x the scale P5 assumed (open, 2026-09-16)

P5's budget arithmetic assumed `N ~ 8k`; the rule returns 15,400. At 0.13%
density a dense `W_rec` is 0.95 GB and wastes 763x the necessary compute.

**Decision (2026-09-16): keep N, change the implementation, not the biology.**
P3's `Delta_W_rec = (B @ A) * M_arm` is realised as a **per-edge gather**,
`Delta_ij = B[i] . A[:,j]` evaluated only on the 310,867 edges (~2.5M MAC at
r=8), never as a dense product followed by a mask. The mathematics of P3 is
unchanged; only its realisation is specified. Whether P5's budget survives at
this N is deferred to the throughput probe, which must benchmark the gather
formulation rather than the dense one.

> **Partly superseded by R4.** Computing the adapter delta per edge is correct
> and is retained. Propagating activity through those edges with
> `index_select` + `index_add` is **not**: measurement showed it to be the
> slowest of the three options, 12.9x slower than `torch.sparse.mm`. The two
> operations are distinct and only the second was wrong.

### R3 — Prior art: DOOMFLY (2026-09-16)

`github.com/nftechie/doomfly` runs the **whole** MaleCNS v1.0 graph (166,700
neurons, 25.6M edges) against live ViZDoom, with a dopamine-gated rule on 4,184
KC->MBON11 synapses. It reports failing its own visual, conditioning and
survival validation gates: "changing weights and longer individual rounds do not
establish learning."

It carries **no matched topology controls**, so success could not have been
attributed to the connectome even had it occurred. That absence is precisely
what P4 supplies. Its other four weaknesses map onto choices this protocol
already makes: untrained "inferred proxy" sensory interface versus P3's trained
encoder; "engineered controller assignments" versus P3's trained readout; 0.016%
of synapses plastic versus masked adaptation over every edge; a local rule alone
versus surrogate BPTT as the primary. Its restriction of plasticity to
*existing* synapses independently converges with P3's `M_arm` masking.

To be cited as a prior negative result.

### R4 — Throughput measured, and R2's propagation advice overturned (result, 2026-09-16)

Measured on the M1 Pro dev host at the real `N = 15,400` / `|E| = 310,867`,
batch 64, 30-step window, rank 8, forward **and** backward:

| Coupling | ms / rollout | h per 5e5 env steps | Verdict |
|---|---:|---:|---|
| `torch.sparse.mm` | **952** | **2.07** | fastest; **the default** |
| dense `N x N` | 2,425 | 5.26 | 2.5x slower, 0.95 GB resident |
| gather (`index_select`+`index_add`) | 12,279 | 26.65 | 12.9x slower than sparse |

Two predictions in R2 were wrong and are corrected here rather than quietly
dropped. First, `torch.sparse.mm` was expected to be unsupported or degraded on
MPS; it is neither, and it wins outright. Second, the per-edge gather was
expected to win on FLOP count; it loses badly, because a 20M-element scattered
`index_add` per timestep is dominated by atomics rather than arithmetic. FLOP
counting did not predict this. The distinction preserved from R2 is that
computing `Delta_ij` per edge remains correct and cheap; only the *propagation*
step changes.

**Trainable adapter parameters at r=8: 246,400** — 59x DOOMFLY's 4,184 plastic
synapses (R3), on a graph 82x smaller.

**A1 gate 2 (dense/sparse equivalence): PASSED, on MPS, at full scale.** All
three couplings produce bit-identical spike rasters (agreement 1.000000, max
absolute difference 0.00e+00 over a 30-step, 8-batch rollout producing 48,727
spikes). This exceeds A1's stated tolerance of >= 95% agreement within +/- 1 dt.

**Thermal behaviour:** over a sustained 120 s load (127 consecutive
forward+backward rollouts) the last quarter ran 1.01x slower than the first.
No meaningful throttling at this window. This does not certify multi-hour
behaviour, which remains unmeasured.

**Resident memory:** 2,056 MB driver-allocated under the sparse path. An earlier
revision of the probe sampled memory after `backward` had already released the
graph and reported an identical 128 MB for all three modes; that figure was
wrong and the measurement was corrected.

**P5 verdict.** P5 assumed ~0.7 h per run and ~30 GPU-h total. On the dev host
the real figure is 2.07 h per run, so the 40-run confirmatory series would cost
~83 h there — too slow for a laptop, exactly as amendment A1 anticipated when it
placed the confirmatory series on the CUDA host. P5's budget is not yet
contradicted for that host, but it remains unverified there; `fly-ky7.4` must
measure it before the budget can be treated as settled.

**New learned constraint.** The coupling multiplies presynaptic spikes, so
`d(out)/d(w_eff) = s_pre`. A substrate that does not spike gives its adapter
*exactly zero* gradient, not a small one. An arm silenced by unnormalised gain
would therefore produce a clean-looking topology difference while measuring
nothing at all. This is the concrete mechanism that makes P4's dynamical
calibration to 5 +/- 0.5 Hz load-bearing, and it is pinned by a regression test.

### D1 — MaleCNS v1.0 considered and deferred (2026-09-16)

MaleCNS v1.0 is publicly downloadable without a neuprint token (three feather
files, ~1.1 GB, `storage.googleapis.com/flyem-male-cns/v1.0/...`; paper
doi:10.1016/j.cell.2026.08.015), despite `natverse/malecns` requiring one.

Rejected as **primary**: its headline advantage is the ventral nerve cord and
real descending-to-motor pathways, which a symbolic card game with no motor
periphery cannot use, at the cost of roughly 50% more neurons of no relevance.
FAFB v783 is already measured, DOI-pinned and CC BY 4.0.

Retained as the strongest available **cross-dataset replication** arm — a
different specimen, sex, reconstruction pipeline and neurotransmitter model,
which would answer "you found an artifact of the FlyWire pipeline." Deferred by
decision on 2026-09-16; if adopted later it is no longer pre-registered, and
that cost was accepted knowingly.

### R5 — P4 controls built and matched at real scale (result, 2026-09-16)

All three arms constructed from the v783 subset. Shuffle: 31M attempted swaps in
11.6 s; 2.21% of original edges survive. Random control: 0.3 s.

| Quantity | real | shuffled | random |
|---|---:|---:|---:|
| N | 15,400 | 15,400 | 15,400 |
| \|E\| | 310,867 | 310,867 | 310,867 |
| density | 0.00131 | 0.00131 | 0.00131 |
| in/out degree mean | 20.18617 | 20.18617 | 20.18617 |
| in-degree max | 2,843 | **2,843** | 42 |
| out-degree max | 2,763 | **2,763** | 65 |
| rho(W) | 0.95000 | 0.95000 | 0.95000 |
| E / I / modulatory | 10,795 / 3,312 / 1,293 | identical | identical |
| **reciprocity** | **0.22536** | 0.02106 | 0.00122 |
| **clustering** | **0.26702** | 0.07520 | 0.00439 |
| largest SCC | 14,841 | 14,846 | 7,703 |

Every tolerance-0 quantity matches exactly, and the controls form a graded
ladder rather than two arbitrary graphs. The shuffled arm reproduces the degree
sequence bit for bit, heavy tail included, and holds the giant strongly-connected
component (14,846 vs 14,841) while losing 10.7x of the reciprocity and 3.6x of
the clustering. `real vs shuffled` therefore isolates higher-order structure at
constant recurrence; `real vs random` additionally removes the degree tail and
halves the giant component.

**Dynamical calibration** (target 5.0 +/- 0.5 Hz):

| Arm | input gain | achieved rate |
|---|---:|---:|
| real | 3.5866 | 5.449 Hz |
| shuffled | 3.5866 | 5.465 Hz |
| random | 3.4755 | 4.975 Hz |

**The rationale for P4's second step was stated imprecisely and is corrected
here.** It was justified as preventing arms from differing in saturation. What
the measurement shows is starker: at unit input **every arm fires 0.000 Hz**
after the spectral normalisation. Combined with the zero-gradient property of a
silent substrate (R4), omitting this step would leave *every* arm unable to
learn, not merely unmatched. Matching between arms is in fact delivered by the
structural step -- the calibrated gains differ by only 3%.

### R6 — First end-to-end run (result, 2026-09-16)

A FlyWire-derived substrate played a complete game of MiniCard.

| | |
|---|---|
| Substrate | 15,400 neurons, 310,867 edges, `rho = 0.95`, calibrated gain 3.587 |
| Input neurons | 1,042 (`sensory`, `visual_projection`, `ascending`) |
| Output neurons | 44 (`descending`, `motor`) |
| Trainable parameters | 542,591 |
| Outcome | 34 actions, 8 turns, result -1, **0 illegal actions** |
| Speed | 13 ms per decision at batch 1 |

The policy lost, which is what an untrained policy should do. The result that
matters is the zero: legality is structural, so the agent cannot emit an illegal
action even at random initialisation, and no part of the reward has to be spent
teaching it the rules.

**Input/output wiring rule.** The game enters through afferent populations and
acts through descending neurons — the pathway the animal itself uses — rather
than through indices chosen by degree. The rule runs once on the real graph and
the resulting *indices* are reused unchanged by every arm, which satisfies P3
because the controls shuffle edges and never nodes.

**Defect found and fixed.** `act()` built its observation tensor on the CPU
regardless of where the module had been moved, which crashed the first attempt
on MPS. Unit tests had not caught it because they never moved the policy off the
CPU; a regression test now does.

**Two silent-substrate traps, both hit during this work.** The head weight
gradients are `d(loss)/d(logits) * features`, so when the substrate does not
spike the *weight* gradients vanish while the *bias* gradients survive — an
obscure signature for "the gain is too low". A toy fixture with 40 neurons
needed recurrent weights of 25 before signal reached the readout at all (4 and
10 left it silent). The real substrate gets this from P4's spectral
normalisation plus its 5 Hz calibration, which is now the third independent
confirmation that P4's second step is load-bearing.

### R7 — The pipeline learns; deck assignment was confounding the seat analysis

**Learning confirmed** (500-neuron substrate, mirror `aggro` deck, opponents
alternating between `board_control` and `greedy`, 24k agent decisions):

| Steps | overall | vs board_control | vs greedy | episodes/rollout |
|---:|---:|---:|---:|---:|
| baseline | -1.000 | -1.000 | -1.000 | 153 |
| 8,000 | -0.735 | -0.471 | -1.000 | 68 |
| 16,000 | -0.556 | -0.143 | -1.000 | 54 |
| 24,000 | **-0.244** | **+0.391** | -0.909 | 45 |

The scripted `board_control` bot scores exactly 0.000 against itself, so a
trained policy at +0.391 is beating the bot it trained against, in roughly 70%
of games. Episodes per rollout fall by a factor of 3.4, i.e. the agent survives
far longer. Against `greedy` it has barely moved, which is the cost accepted
when the alternating-opponent configuration was chosen: a racing bot is brutal
against a weak policy (uniform-random scores -0.993 there).

**Confound found and removed.** `MiniCard` defaulted to `("aggro", "control")`,
so although the agent alternated *seats*, the decks stayed pinned to the seats
and it always held `aggro` in seat 0. Measurement showed that after amendment
A2 raised HP to 30, **no policy wins with the aggro deck against a control
deck** — `board_control` scores 0.0% over 200 games from either seat. Half of
every rollout was therefore unwinnable regardless of the policy, and half the
gradient signal was noise.

A2 fixed one degeneracy and created another: it was recorded as "the strategic
policy now dominates the shallow one", which was true, but the corollary that
the aggro archetype had become *unplayable* rather than merely weaker went
unchecked.

P2 never required the two players to hold different decks, so the primary
configuration is now a **mirror matchup** and no amendment is needed. Deck
variation stays where P2 already put it, in the Stage 4 held-out set. Mirror
matchups also give a clean engine check: any policy against itself scores
exactly +0.000 / 50.0% in both mirrors.

**Open item.** An untrained *spiking* policy scores -1.000 where uniform-random
scores -0.580, so the initialisation is degenerate rather than random: with
`B = 0` the adapter is a no-op and the heads read spike counts that may barely
vary across states. This costs nothing in the comparison, since every arm starts
the same way, but it wastes early samples and should be characterised.

### R8 — Power of the pre-registered design, measured before any run

The P5 analysis was implemented and then characterised on synthetic outcomes
shaped like real ones (terminal return +/-1, per-seed win probability wobbling
around the arm mean), 200 trials per cell.

**False positive rate under the null: 0.010** at a nominal alpha of 0.05. The
test is *conservative*, not inflated. The reason is structural: P5 requires both
a Holm-adjusted p below alpha **and** a 95% interval excluding zero, and a
one-sided test at 0.05 corresponds to a 90% two-sided interval rather than a
95% one, so the effective level is nearer 0.025. This is a property of the
pre-registered rule, not a defect.

**Power at the pre-registered 10 seeds:**

| True advantage of the real arm | Detected |
|---|---:|
| +5 percentage points of win rate | 0.195 |
| +10 pp | **0.715** |
| +15 pp | 0.975 |
| +20 pp | 1.000 |

**Power against seed count at +10 pp:** 5 seeds 0.465, 10 seeds 0.715, 20 seeds
0.935.

**Decision (2026-09-16): keep 10 seeds.** The budget stays as pre-registered and
the power curve is published beside the result. The consequence is stated in
advance so it cannot be glossed later: this design detects a topology advantage
of roughly 10 percentage points about seven times in ten and will usually miss
anything at 5 points. **A null result must therefore be reported as "no effect
of at least ~10 pp was detected", never as "there is no effect."**

### D2 — Training against real Hearthstone is not available (2026-09-16)

Checked rather than assumed, after the deck `AAECAeHJBgjNngapiAeEmQfQvweb1Aet2QeO3Afb4AcL...`
(Control Warlock, Standard) was proposed as a training target. The deckstring
parses correctly — `FT_STANDARD`, hero 107745, 19 unique cards, 30 total — but
nothing exists to play it in:

| Package | Result |
|---|---|
| `fireplace` | not installable; unmaintained, predates this card pool by years |
| `hearthbreaker` | not installable |
| `sabberstone` | not installable |
| `hearthstone-sim` | not installable |
| `hslog` | installs, but parses logs; it is not a simulator |
| `python-hearthstone` | installs; **metadata only** — card definitions and deckstrings, no game |

Automating the retail client is refused: it violates the publisher's terms and
risks the user's account. HSReplay supplies logs of games already played, not an
environment to act in, so it cannot host reinforcement learning either.

The blocker is the absence of an environment, not the amount of work. Closing it
would mean writing a Hearthstone simulator — rules engine, trigger ordering,
secrets, and the card text of both decks — a project comparable in size to
everything built here, contributing nothing to the topology question.

Deferred by decision on 2026-09-16 in favour of finishing the pre-registered
experiment. Real card definitions remain usable for raising MiniCard's fidelity
at the mini-Hearthstone stage P2 already schedules.

## Interview Transcript

<details>
<summary>Q&A (Round 0 plus 7 rounds)</summary>

### Round 0
**Q:** I’m reading this as four top-level components: FlyWire substrate/data, SNN simulation/learning, Hearthstone-like environment/interfaces, and evaluation/data/baselines/ablations. Is that topology right?

**A:** Topology is correct.

### Round 1
**Q:** What should be the primary claim of the first publishable version?

**A:** Topology value: test whether real FlyWire topology provides an advantage over matched random/shuffled controls.

**Ambiguity after round:** 58.5%.

### Round 2
**Q:** What environment boundary is accepted for the first publishable topology experiment?

**A:** Toy card game.

**Ambiguity after round:** 50.0%.

### Round 3
**Q:** What connectome scope is fixed for the first topology experiment?

**A:** Functional subset.

**Ambiguity after round:** 50.0%.

### Round 4
**Q:** What identical learning contract should be used across topology arms?

**A:** Initially selected frozen reservoir plus trainable readout, then asked whether LoRA could help.

**Ambiguity after round:** 31.0%.

### Round 5
**Q:** How should a LoRA-like low-rank adapter be positioned in the causal claim?

**A:** Make low-rank adaptation the primary method.

**Ambiguity after round:** 33.0%.

### Round 6
**Q:** What is success in the first toy-game topology experiment?

**A:** Real topology plus adapter should be compared with matched random/shuffled controls on mean return and sample efficiency under fixed compute/sample budget.

**Ambiguity after round:** 24.0%.

### Round 7
**Q:** What matching contract is required for causal topology controls?

**A:** Strict matched controls: preserve degree/edge-count/sign/weight-distribution properties where applicable, and match adapter rank, initialization family, optimizer, seeds, trainable parameter count, and compute/sample budgets.

**Ambiguity after round:** 14.1%.

</details>
