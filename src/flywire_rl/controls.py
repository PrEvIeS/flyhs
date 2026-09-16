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

from flywire_rl.spiking import CouplingMode, LIFSubstrate, LowRankCoupling


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


def population_rate_hz(
    graph: Graph,
    input_gain: float,
    steps: int = 200,
    batch: int = 4,
    dt_ms: float = 1.0,
    seed: int = 0,
    device: str = "cpu",
) -> float:
    """Mean population firing rate under a fixed reference input."""
    torch.manual_seed(seed)
    substrate = LIFSubstrate(coupling_from_graph(graph, device=device), dt_ms=dt_ms)

    generator = torch.Generator().manual_seed(seed)
    drive = torch.randn(steps, batch, graph.n, generator=generator) * input_gain

    with torch.no_grad():
        raster, _ = substrate(drive.to(device))

    seconds = steps * dt_ms / 1000.0
    return float(raster.sum()) / (graph.n * batch * seconds)


def calibrate_input_gain(
    graph: Graph,
    target_hz: float = 5.0,
    tolerance_hz: float = 0.5,
    lo: float = 1e-3,
    hi: float = 1e4,
    max_iter: int = 40,
    **kwargs,
) -> float:
    """P4's dynamical gain match: drive every arm to the same population rate.

    The structural match already fixes ``rho(W) = 0.95``, so the free parameter
    here is the input scale rather than the recurrent weights -- rescaling the
    latter would undo the spectral match.

    This is not cosmetic. The recurrent coupling multiplies presynaptic spikes,
    so ``d(output)/d(w_eff) = s_pre``: an arm that does not spike hands its
    adapter *exactly* zero gradient and would look like a clean topology result
    while measuring nothing at all.
    """
    if population_rate_hz(graph, hi, **kwargs) < target_hz:
        raise ValueError(
            f"target {target_hz} Hz is unreachable: gain {hi} yields only "
            f"{population_rate_hz(graph, hi, **kwargs):.2f} Hz"
        )

    for _ in range(max_iter):
        mid = (lo * hi) ** 0.5  # bisect in log space; gain spans decades
        rate = population_rate_hz(graph, mid, **kwargs)
        if abs(rate - target_hz) <= tolerance_hz:
            return mid
        if rate < target_hz:
            lo = mid
        else:
            hi = mid

    raise ValueError(f"calibration did not converge within {max_iter} iterations")


def branching_ratio(
    graph: Graph,
    scale: float = 1.0,
    kick_fraction: float = 0.02,
    kick_steps: int = 5,
    free_steps: int = 35,
    kick_current: float = 60.0,
    seed: int = 0,
    device: str = "cpu",
) -> float:
    """Spikes produced per spike, measured while the network runs unaided.

    The network is kicked for a few steps and then left with **no external
    drive at all**, so what is measured is transmission through the graph rather
    than a response to injected current. A ratio near 1 is the spiking analogue
    of a unit spectral radius: activity neither dies out nor explodes.
    """
    scaled = replace(graph, weight=(graph.weight * scale).astype(np.float32))
    signed = scaled.sign[scaled.pre].astype(np.float32) * scaled.weight
    coupling = LowRankCoupling(
        scaled.n,
        torch.from_numpy(np.ascontiguousarray(scaled.pre)),
        torch.from_numpy(np.ascontiguousarray(scaled.post)),
        torch.from_numpy(signed),
        rank=1,
        mode=CouplingMode.SPARSE,
    ).to(device)

    torch.manual_seed(seed)
    drive = torch.zeros(kick_steps + free_steps, 1, scaled.n, device=device)
    chosen = torch.randperm(scaled.n, device=device)[: int(kick_fraction * scaled.n)]
    drive[:kick_steps, 0, chosen] = kick_current

    with torch.no_grad():
        raster, _ = LIFSubstrate(coupling)(drive)

    counts = raster[kick_steps:].sum(dim=(1, 2)).cpu().numpy()
    ratios = [
        counts[i + 1] / counts[i] for i in range(len(counts) - 1) if counts[i] > 0
    ]
    return float(np.mean(ratios)) if ratios else 0.0


def calibrate_branching(
    graph: Graph,
    target: float = 1.0,
    tolerance: float = 0.15,
    lo: float = 1e-2,
    hi: float = 1e5,
    max_iter: int = 40,
    **kwargs,
) -> float:
    """Find the weight scale at which the network transmits at ``target``.

    Amendment A3 replaced P4's ``rho = 0.95`` with this. A spectral radius near
    1 is the right criterion for a linear rate network, where the question is
    whether activity decays; it is the wrong scale for leaky integrate-and-fire
    units with a hard threshold. At rho = 0.95 on the v783 subset the mean
    synaptic weight moves a membrane by 6.6e-4 against a threshold of 1, so
    roughly 1,500 coincident inputs would be needed to fire a neuron whose mean
    in-degree is 20: the graph transmits nothing and topology cannot matter.
    """
    if branching_ratio(graph, hi, **kwargs) < target:
        raise ValueError(
            f"target branching {target} unreachable: scale {hi} gives only "
            f"{branching_ratio(graph, hi, **kwargs):.3f}"
        )

    for _ in range(max_iter):
        mid = (lo * hi) ** 0.5
        ratio = branching_ratio(graph, mid, **kwargs)
        if abs(ratio - target) <= tolerance:
            return mid
        if ratio < target:
            lo = mid
        else:
            hi = mid

    raise ValueError(f"branching calibration did not converge in {max_iter} steps")


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
