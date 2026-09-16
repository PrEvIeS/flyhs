"""LIF substrate with a low-rank adapter masked to the connectome's own support.

Implements P3. The adapter is ``Delta_W = (B @ A) * M_arm``, but it is never
computed that way: at N=15,400 a dense product is 0.95 GB and 763x more
arithmetic than the graph needs. Per clarification R2 it is evaluated as a
**per-edge gather**, ``Delta_ij = B[i] . A[:, j]`` over the 310,867 edges only.

Three couplings compute the identical quantity and exist to be checked against
each other, which is the dense/sparse equivalence gate that amendment A1
requires before any confirmatory run:

``gather``
    ``index_select`` + ``index_add``. Runs everywhere. Materialises a
    ``(batch, E)`` intermediate per timestep, so autograd memory grows with
    ``batch * E * steps``.
``sparse``
    ``torch.sparse.mm``. No ``(batch, E)`` intermediate. Backend support is
    uneven, which is precisely what the throughput probe measures.
``dense``
    Baseline only. Materialises the full ``N x N`` matrix.
"""

from __future__ import annotations

import math

from dataclasses import dataclass
from enum import Enum

import torch
from torch import Tensor, nn


class CouplingMode(str, Enum):
    GATHER = "gather"
    SPARSE = "sparse"
    DENSE = "dense"


class SurrogateSpike(torch.autograd.Function):
    """Heaviside forward, fast-sigmoid surrogate backward (P3: beta = 10)."""

    @staticmethod
    def forward(ctx, v: Tensor, beta: float) -> Tensor:
        ctx.save_for_backward(v)
        ctx.beta = beta
        return (v > 0).to(v.dtype)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (v,) = ctx.saved_tensors
        scale = 1.0 / (1.0 + ctx.beta * v.abs()) ** 2
        return grad_output * scale, None


def spike(v: Tensor, beta: float = 10.0) -> Tensor:
    return SurrogateSpike.apply(v, beta)


class LowRankCoupling(nn.Module):
    """Frozen sparse recurrent weights plus a trainable rank-``r`` adapter.

    The frozen weights and the edge list are buffers; only ``B`` and ``A`` are
    trainable, giving ``2 * N * r`` parameters regardless of which arm's graph
    is loaded. The delta an arm can express is confined to that arm's own edges,
    which is what keeps the topology comparison meaningful.
    """

    def __init__(
        self,
        n_neurons: int,
        pre_idx: Tensor,
        post_idx: Tensor,
        weights: Tensor,
        rank: int = 8,
        mode: CouplingMode | str = CouplingMode.GATHER,
    ) -> None:
        super().__init__()
        if pre_idx.shape != post_idx.shape or pre_idx.shape != weights.shape:
            raise ValueError("pre_idx, post_idx and weights must be the same length")

        self.n_neurons = n_neurons
        self.rank = rank
        self.mode = CouplingMode(mode)

        self.register_buffer("pre_idx", pre_idx.long())
        self.register_buffer("post_idx", post_idx.long())
        self.register_buffer("weights", weights.float())

        # P3 initialisation: B = 0 so Delta_W = 0 at t=0 and the substrate
        # starts as the pure connectome.
        self.B = nn.Parameter(torch.zeros(n_neurons, rank))
        self.A = nn.Parameter(torch.randn(rank, n_neurons) / (n_neurons**0.5))

    @property
    def n_edges(self) -> int:
        return int(self.weights.numel())

    def effective_weights(self) -> Tensor:
        """Per-edge ``w_ij + B[i] . A[:, j]``, evaluated only on real edges."""
        delta = (self.B[self.pre_idx] * self.A[:, self.post_idx].t()).sum(-1)
        return self.weights + delta

    def operator(self):
        """Build the coupling once, then reuse it across a rollout's timesteps.

        The adapter is constant within a rollout, so the effective weights are
        materialised once rather than per timestep.
        """
        w_eff = self.effective_weights()

        if self.mode is CouplingMode.GATHER:
            pre, post, n = self.pre_idx, self.post_idx, self.n_neurons

            def apply(s: Tensor) -> Tensor:
                contrib = s.index_select(1, pre) * w_eff
                out = torch.zeros(s.shape[0], n, dtype=s.dtype, device=s.device)
                return out.index_add(1, post, contrib)

        elif self.mode is CouplingMode.SPARSE:
            indices = torch.stack([self.post_idx, self.pre_idx])
            w_sp = torch.sparse_coo_tensor(
                indices, w_eff, (self.n_neurons, self.n_neurons)
            ).coalesce()

            def apply(s: Tensor) -> Tensor:
                return torch.sparse.mm(w_sp, s.t()).t()

        else:
            dense = torch.zeros(
                self.n_neurons, self.n_neurons, dtype=w_eff.dtype, device=w_eff.device
            )
            dense = dense.index_put(
                (self.pre_idx, self.post_idx), w_eff, accumulate=True
            )

            def apply(s: Tensor) -> Tensor:
                return s @ dense

        return apply


@dataclass(frozen=True)
class ShiuParams:
    """Biophysical constants of the Shiu et al. (2024) whole-brain LIF model.

    Millivolts and milliseconds throughout. Every value except ``w_syn_mv`` and
    ``dt_ms`` is taken from the literature via the published implementation
    (``philshiu/Drosophila_brain_model``, ``model.py``): the potentials and
    ``t_mbr``/``t_refractory`` from Kakaria and de Bivort 2017, ``tau_syn`` from
    Juergensen et al. 2021, and ``t_delay`` from Paul et al. 2015.

    ``w_syn_mv`` is the model's **single free parameter**, fitted once at
    0.275 mV per anatomical synapse. It is what replaces the pilot's per-arm
    rescaling: a spectral radius or a branching ratio is a criterion for a
    linear rate network, and on a hard-threshold LIF both left the graph
    transmitting nothing. Here the anatomy keeps its own units -- the coupling
    holds signed synapse counts -- and this one absolute scale converts them to
    millivolts, identically for every arm.
    """

    dt_ms: float = 0.1
    v_rest_mv: float = -52.0
    v_reset_mv: float = -52.0
    v_threshold_mv: float = -45.0
    t_mbr_ms: float = 20.0
    tau_syn_ms: float = 5.0
    t_refractory_ms: float = 2.2
    t_delay_ms: float = 1.8
    w_syn_mv: float = 0.275
    beta: float = 10.0

    @property
    def threshold_gap_mv(self) -> float:
        """Distance from rest to threshold: the yardstick for any weight scale."""
        return self.v_threshold_mv - self.v_rest_mv

    @property
    def n_delay_steps(self) -> int:
        return max(1, round(self.t_delay_ms / self.dt_ms))

    @property
    def n_refractory_steps(self) -> int:
        return round(self.t_refractory_ms / self.dt_ms)

    def decay_coefficients(self) -> tuple[float, float, float]:
        """Exact per-step solution of the two linear ODEs.

        For ``dv/dt = (g - (v - v_rest)) / t_mbr`` and ``dg/dt = -g / tau``::

            g <- a * g
            v <- v_rest + b * (v - v_rest) + c * g

        Explicit Euler is not an option here: the pilot's substrate put synaptic
        input through a ``dt/tau`` gain, which attenuated it by a factor of 20
        on its own.
        """
        a = math.exp(-self.dt_ms / self.tau_syn_ms)
        b = math.exp(-self.dt_ms / self.t_mbr_ms)
        c = (b - a) * self.tau_syn_ms / (self.t_mbr_ms - self.tau_syn_ms)
        return a, b, c


@dataclass
class Rollout:
    """What one pass over the substrate produced. Potentials are in millivolts.

    ``membrane`` is the mean **pre-reset** potential per neuron and ``peak_v``
    its maximum. Both remain useful as graded observables, but with the absolute
    scale in place they are no longer a workaround for a silent spike raster:
    amendment A4 reached for the membrane because branching-calibrated weights
    left the readout neurons at exactly zero spikes.

    ``g_trace`` is the full synaptic-drive history and is recorded only on
    request -- it is ``(steps, batch, n_neurons)``, the same size as the raster.
    """

    spikes: Tensor
    membrane: Tensor
    final_v: Tensor
    peak_v: Tensor
    g_trace: Tensor | None = None


class LIFSubstrate(nn.Module):
    """Shiu et al. (2024) leaky integrate-and-fire population on a connectome.

    Two state variables per neuron, both in millivolts: the membrane ``v`` and
    the synaptic drive ``g``, which receives an instantaneous jump of
    ``sign(j) * n_synapses(j, i) * w_syn`` on a presynaptic spike and decays with
    ``tau_syn``. Despite the name, ``g`` is not a conductance: it enters
    ``dv/dt`` additively in volts, so excitation and inhibition have identical
    driving force and there is no reversal potential.

    The discrete-time semantics follow Brian2's slot order, as probed and
    documented by ``liuzihe02/fly-craftax``:

    * ``(unless refractory)`` on both ODEs means Brian2 skips *every* write to
      ``v`` and ``g`` while a neuron is refractory. A synaptic jump landing on a
      refractory target is dropped outright -- it does not accumulate and it
      does not fire the neuron once refractoriness ends.
    * The refractory counter is decremented *before* the active mask is read.
    * The threshold is evaluated after the state update but before this step's
      synaptic input, so the reset wipes anything that arrived on a spiking step.

    ``drive`` is our own addition, not Brian2's: a tonic millivolt-per-step
    input applied in the state-update slot and, like every other write to ``v``,
    dropped while refractory. It is dt-bound -- a constant drive settles at
    ``drive / (1 - b)`` above rest -- so a calibrated value must be rescaled if
    ``dt_ms`` changes.
    """

    def __init__(
        self,
        coupling: LowRankCoupling,
        params: ShiuParams | None = None,
    ) -> None:
        super().__init__()
        self.coupling = coupling
        self.params = params or ShiuParams()

    def forward(self, drive: Tensor) -> tuple[Tensor, Tensor]:
        """Spike raster and final membrane potential; see :meth:`rollout`."""
        result = self.rollout(drive)
        return result.spikes, result.final_v

    def rollout(self, drive: Tensor, record_g: bool = False) -> Rollout:
        """Roll out over ``drive`` of shape ``(steps, batch, n_neurons)``, in mV."""
        steps, batch, n = drive.shape
        if n != self.coupling.n_neurons:
            raise ValueError(
                f"drive has {n} channels but the coupling has "
                f"{self.coupling.n_neurons} neurons"
            )

        p = self.params
        a, b, c = p.decay_coefficients()
        n_delay = p.n_delay_steps
        apply = self.coupling.operator()

        v = torch.full(
            (batch, n), p.v_rest_mv, dtype=drive.dtype, device=drive.device
        )
        g = torch.zeros_like(v)
        # Integer countdown, deliberately outside the autograd graph: gating is
        # not a quantity the adapter can learn through.
        refractory = torch.zeros((batch, n), dtype=torch.long, device=drive.device)

        raster: list[Tensor] = []
        g_history: list[Tensor] = []
        membrane = torch.zeros_like(v)
        peak_v = v.clone()

        for t in range(steps):
            refractory = torch.clamp(refractory - 1, min=0)
            active = (refractory == 0).to(v.dtype)

            # State update, then the tonic drive: both skipped while refractory.
            stepped = p.v_rest_mv + b * (v - p.v_rest_mv) + c * g + drive[t]
            v = torch.lerp(v, stepped, active)
            g = torch.lerp(g, a * g, active)

            membrane = membrane + v  # accumulated before the reset
            peak_v = torch.maximum(peak_v, v)

            # Brian2 checks the threshold here, before this step's synaptic input.
            s = active * spike(v - p.v_threshold_mv, p.beta)

            delayed = raster[t - n_delay] if t >= n_delay else torch.zeros_like(v)
            g = g + active * (p.w_syn_mv * apply(delayed))

            # Reset runs after the synapse slot and wipes whatever just landed.
            v = torch.lerp(v, torch.full_like(v, p.v_reset_mv), s)
            g = g * (1.0 - s)
            refractory = torch.where(s.detach() > 0, p.n_refractory_steps, refractory)

            raster.append(s)
            if record_g:
                g_history.append(g)

        return Rollout(
            spikes=torch.stack(raster),
            membrane=membrane / steps,
            final_v=v,
            peak_v=peak_v,
            g_trace=torch.stack(g_history) if record_g else None,
        )
