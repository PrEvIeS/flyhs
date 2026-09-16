"""P4 topology controls: matched shuffled and random graphs, plus gain matching.

This is the causal core of the experiment. Where P4 says "tolerance 0" the
quantity is preserved exactly rather than approximately, because a control that
differs from the real graph in anything but topology makes the comparison
uninterpretable.

A useful identity makes the shuffle cheap and provably correct. A directed
double-edge swap takes ``(a->b, c->d)`` to ``(a->d, c->b)``, which means each
edge keeps its **source** and only exchanges its **target**. In array terms only
``post`` is permuted; ``pre`` never moves. Three of P4's tolerance-0 constraints
then hold by construction rather than by checking:

* out-degree is the multiset of ``pre``, which is untouched;
* in-degree is the multiset of ``post``, which is permuted among edges;
* the presynaptic neuron of every edge is fixed, so a neuron cannot acquire a
  second transmitter and **Dale's law survives automatically**.

Weights stay in their slot, so each weight also keeps its own source neuron.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from flywire_rl.spiking import (
    CouplingMode,
    LIFSubstrate,
    LowRankCoupling,
    ShiuParams,
)


@dataclass(frozen=True)
class Graph:
    """A signed, weighted directed graph in edge-list form.

    ``sign`` is per *neuron*, not per edge: Dale's law makes excitation and
    inhibition a property of the presynaptic cell.
    """

    n: int
    pre: np.ndarray  # (E,) int64
    post: np.ndarray  # (E,) int64
    weight: np.ndarray  # (E,) float32, magnitude only
    sign: np.ndarray  # (N,) int8 in {-1, 0, +1}

    @property
    def n_edges(self) -> int:
        return int(self.pre.size)

    def signed_matrix(self) -> sp.csr_matrix:
        values = self.sign[self.pre].astype(np.float64) * self.weight
        return sp.csr_matrix((values, (self.pre, self.post)), shape=(self.n, self.n))


def _keys(pre: np.ndarray, post: np.ndarray, n: int) -> np.ndarray:
    return pre.astype(np.int64) * n + post.astype(np.int64)


def _exists(sorted_keys: np.ndarray, probe: np.ndarray) -> np.ndarray:
    """Vectorised membership test against a sorted key array."""
    if sorted_keys.size == 0:
        return np.zeros(probe.shape, dtype=bool)
    slot = np.searchsorted(sorted_keys, probe)
    slot = np.clip(slot, 0, sorted_keys.size - 1)
    return sorted_keys[slot] == probe


def shuffled_control(
    graph: Graph,
    seed: int = 0,
    swaps_per_edge: int = 100,
    batch: int = 200_000,
) -> Graph:
    """Degree-preserving directed double-edge swap.

    P4 fixes the budget at ``100 * |E|`` attempted swaps. At the real scale that
    is 31M attempts, so proposals are generated and validated in vectorised
    batches rather than one at a time.
    """
    rng = np.random.default_rng(seed)
    n, m = graph.n, graph.n_edges
    pre = graph.pre.copy()
    post = graph.post.copy()

    attempts_left = swaps_per_edge * m
    while attempts_left > 0:
        # Sample disjoint edge slots so no slot takes part in two swaps at once.
        # The budget is decremented by the attempts actually proposed, not by
        # the nominal batch size, because 100 * |E| is a pre-registered number.
        pairs = min(batch, attempts_left, m // 2)
        if pairs < 1:
            break
        attempts_left -= pairs

        picks = rng.choice(m, size=2 * pairs, replace=False)
        left, right = picks[:pairs], picks[pairs:]

        new_left = _keys(pre[left], post[right], n)
        new_right = _keys(pre[right], post[left], n)

        sorted_keys = np.sort(_keys(pre, post, n))
        ok = (
            (pre[left] != post[right])
            & (pre[right] != post[left])
            & ~_exists(sorted_keys, new_left)
            & ~_exists(sorted_keys, new_right)
            & (new_left != new_right)
        )

        # Two accepted swaps in one batch must not create the same edge.
        half = int(ok.sum())
        if half == 0:
            continue
        proposed = np.concatenate([new_left[ok], new_right[ok]])
        _, first = np.unique(proposed, return_index=True)
        keep = np.zeros(proposed.size, dtype=bool)
        keep[first] = True
        accepted = ok.copy()
        accepted[ok] = keep[:half] & keep[half:]

        a, b = left[accepted], right[accepted]
        post[a], post[b] = post[b].copy(), post[a].copy()

    return replace(graph, pre=pre, post=post)


def random_control(graph: Graph, seed: int = 0) -> Graph:
    """Matched-size random digraph with the degree sequence deliberately destroyed.

    Node count, edge count, the weight multiset and the per-neuron sign vector
    are all preserved exactly -- stricter than P4's stated tolerances of 1e-3,
    1 percentage point and KS 0.01 respectively. Signs stay attached to their
    own neuron rather than being permuted, because P3 fixes the input and output
    node *indices* across arms: permuting signs would change the excitatory
    composition of the interface and leak a second difference into the
    comparison.
    """
    rng = np.random.default_rng(seed)
    n, m = graph.n, graph.n_edges

    chosen: set[int] = set()
    while len(chosen) < m:
        need = m - len(chosen)
        a = rng.integers(0, n, size=need * 2 + 16)
        b = rng.integers(0, n, size=need * 2 + 16)
        valid = a != b
        chosen.update(_keys(a[valid], b[valid], n).tolist())

    keys = np.array(sorted(chosen)[:m], dtype=np.int64)
    rng.shuffle(keys)

    return replace(
        graph,
        pre=keys // n,
        post=keys % n,
        weight=rng.permutation(graph.weight),
    )


def spectral_radius(graph: Graph) -> float:
    matrix = graph.signed_matrix()
    if graph.n < 12:
        return float(np.max(np.abs(np.linalg.eigvals(matrix.toarray()))))
    values = spla.eigs(
        matrix, k=1, which="LM", return_eigenvectors=False, maxiter=10_000
    )
    return float(abs(values[0]))


def normalise_spectral_radius(graph: Graph, target: float = 0.95) -> Graph:
    """P4's structural gain match: rescale so every arm shares a spectral radius.

    Without it, a real graph with radius 1091.8 (result R1) and a control near 1
    would differ in saturation rather than in structure, and "topology matters"
    would be an artefact of gain.
    """
    current = spectral_radius(graph)
    if current == 0:
        raise ValueError("cannot normalise a graph with spectral radius 0")
    return replace(graph, weight=(graph.weight * (target / current)).astype(np.float32))


def coupling_from_graph(
    graph: Graph, rank: int = 8, mode: CouplingMode = CouplingMode.SPARSE, device="cpu"
) -> LowRankCoupling:
    """Wrap a graph as a frozen recurrent coupling, applying Dale signs."""
    signed = graph.sign[graph.pre].astype(np.float32) * graph.weight
    return LowRankCoupling(
        graph.n,
        torch.from_numpy(np.ascontiguousarray(graph.pre)),
        torch.from_numpy(np.ascontiguousarray(graph.post)),
        torch.from_numpy(signed),
        rank=rank,
        mode=mode,
    ).to(device)


@dataclass(frozen=True)
class ActivityProfile:
    """What one arm does under a fixed tonic input, at a given ``w_syn``.

    These are the quantities the calibration decides on, and they are the ones
    the pilot never measured together: it targeted a dynamical constant and
    never checked that the readout saw anything.
    """

    w_syn_mv: float
    drive_mv: float
    rate_hz: float
    active_fraction: float
    readout_live: int
    readout_rate_hz: float


def activity_profile(
    graph: Graph,
    drive_mv: float,
    input_idx: np.ndarray | None = None,
    readout_idx: np.ndarray | None = None,
    params: ShiuParams | None = None,
    steps: int = 200,
    batch: int = 4,
    seed: int = 0,
    device: str = "cpu",
) -> ActivityProfile:
    """Run the arm under a tonic drive on ``input_idx`` and measure what happens.

    ``drive_mv`` is millivolts per step added in the state-update slot, so it is
    dt-bound: a constant drive settles at ``drive_mv / (1 - exp(-dt/t_mbr))``
    above rest. At the default ``dt = 0.1 ms`` that is about 200x the per-step
    value, against a 7 mV gap.
    """
    p = params or ShiuParams()
    torch.manual_seed(seed)
    substrate = LIFSubstrate(coupling_from_graph(graph, device=device), params=p)

    drive = torch.zeros(steps, batch, graph.n, device=device)
    if drive_mv != 0.0:
        columns = (
            torch.arange(graph.n, device=device)
            if input_idx is None
            else torch.as_tensor(np.asarray(input_idx), device=device).long()
        )
        drive[:, :, columns] = drive_mv

    with torch.no_grad():
        raster, _ = substrate(drive)

    seconds = steps * p.dt_ms / 1000.0
    ever_spiked = raster.sum(dim=0) > 0  # (batch, n)

    if readout_idx is None:
        readout = torch.arange(graph.n, device=device)
    else:
        readout = torch.as_tensor(np.asarray(readout_idx), device=device).long()
    readout_spikes = raster.index_select(2, readout)

    return ActivityProfile(
        w_syn_mv=p.w_syn_mv,
        drive_mv=drive_mv,
        rate_hz=float(raster.sum()) / (graph.n * batch * seconds),
        active_fraction=float(ever_spiked.float().mean()),
        readout_live=int((readout_spikes.sum(dim=0) > 0).any(dim=0).sum()),
        readout_rate_hz=float(readout_spikes.sum())
        / (max(readout.numel(), 1) * batch * seconds),
    )


#: Log-spaced candidates starting from the value Shiu et al. fitted on the whole
#: brain. A subset keeps only a fraction of each neuron's real input, so the
#: scale that works here is expected to sit at or above the published 0.275 mV.
#:
#: The ratio is 1.5, not 2. The window between "the readout never fires" and
#: "the network saturates" is narrow -- on a matched-size synthetic graph it ran
#: from about 3.0 to 4.0 mV, with 2.2 leaving the readout dead and 4.4 already at
#: 25.5% of neurons active -- and a doubling grid steps straight over it.
W_SYN_CANDIDATES = tuple(round(0.275 * 1.5**k, 3) for k in range(11))


def calibrate_w_syn(
    graph: Graph,
    input_idx: np.ndarray,
    readout_idx: np.ndarray,
    drive_mv: float,
    candidates: tuple[float, ...] = W_SYN_CANDIDATES,
    min_readout_live: int = 1,
    max_active_fraction: float = 0.25,
    params: ShiuParams | None = None,
    **kwargs,
) -> tuple[float, list[ActivityProfile]]:
    """Pick the per-synapse scale by what the substrate must actually do.

    This replaces both of the pilot's calibrations. ``rho = 0.95`` and a
    branching ratio of 1 are criteria for a *linear rate* network; on a
    hard-threshold LIF the first left one spike moving the membrane by 6.6e-4 of
    the threshold gap, and the second made the firing rate a property of the
    network rather than of the input, so the 5 Hz input calibration stopped
    converging. Neither ever asked whether the readout saw a spike, and on the
    pilot it did not.

    The criterion here is functional, after ``liuzihe02/fly-craftax``:

    1. at least ``min_readout_live`` readout neurons fire, so the readout
       carries signal rather than a constant;
    2. no more than ``max_active_fraction`` of the network fires, so the arm is
       transmitting rather than running away;
    3. the driven network is strictly more active than the same network with a
       blank input, so activity is caused by the input and not by the substrate.

    Returns the chosen scale and the full sweep, which belongs in the run's
    provenance. If nothing qualifies this raises rather than returning a
    best-effort value: a substrate that fails all three is not a calibration
    result and must not be run past silently.
    """
    base = params or ShiuParams()
    profiles: list[ActivityProfile] = []

    for w in candidates:
        p = replace(base, w_syn_mv=w)
        driven = activity_profile(
            graph, drive_mv, input_idx, readout_idx, params=p, **kwargs
        )
        profiles.append(driven)

        if driven.readout_live < min_readout_live:
            continue
        if driven.active_fraction > max_active_fraction:
            continue

        blank = activity_profile(graph, 0.0, input_idx, readout_idx, params=p, **kwargs)
        if driven.rate_hz > blank.rate_hz:
            return w, profiles

    table = "\n".join(
        f"  w_syn={q.w_syn_mv:6.3f}  rate={q.rate_hz:8.2f} Hz  "
        f"active={q.active_fraction:.3f}  readout_live={q.readout_live}"
        for q in profiles
    )
    raise ValueError(
        "no candidate w_syn satisfied the calibration criterion "
        f"(readout_live >= {min_readout_live}, active_fraction <= "
        f"{max_active_fraction}, input-dependent activity) at drive "
        f"{drive_mv} mV/step:\n{table}"
    )


def graph_stats(graph: Graph) -> dict:
    """Every quantity P4 requires to be reported for each arm."""
    n, m = graph.n, graph.n_edges
    out_degree = np.bincount(graph.pre, minlength=n)
    in_degree = np.bincount(graph.post, minlength=n)

    keys = _keys(graph.pre, graph.post, n)
    reverse = _keys(graph.post, graph.pre, n)
    reciprocal = int(_exists(np.sort(keys), reverse).sum())

    adjacency = sp.csr_matrix(
        (np.ones(m), (graph.pre, graph.post)), shape=(n, n), dtype=np.float64
    )
    n_components, labels = sp.csgraph.connected_components(
        adjacency, directed=True, connection="strong"
    )

    undirected = ((adjacency + adjacency.T) > 0).astype(np.float64)
    undirected.setdiag(0)
    undirected.eliminate_zeros()
    degree = np.asarray(undirected.sum(axis=1)).ravel()
    triangles = np.asarray((undirected @ undirected @ undirected).diagonal()).ravel()
    pairs = degree * (degree - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        local = np.where(pairs > 0, triangles / pairs, 0.0)

    return {
        "n": n,
        "edges": m,
        "density": m / (n * n),
        "in_degree_mean": float(in_degree.mean()),
        "in_degree_max": int(in_degree.max()),
        "out_degree_mean": float(out_degree.mean()),
        "out_degree_max": int(out_degree.max()),
        "reciprocity": reciprocal / m if m else 0.0,
        "clustering": float(local.mean()),
        "n_strongly_connected": int(n_components),
        "largest_scc": int(np.bincount(labels).max()),
        "spectral_radius": spectral_radius(graph),
        "excitatory": int((graph.sign > 0).sum()),
        "inhibitory": int((graph.sign < 0).sum()),
        "modulatory": int((graph.sign == 0).sum()),
    }
