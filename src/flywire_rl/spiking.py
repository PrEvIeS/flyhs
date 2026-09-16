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


@dataclass
class Rollout:
    """What one pass over the substrate produced.

    ``membrane`` is the mean **pre-reset** potential per neuron. Amendment A4
    made it the readout: spike counts are identically zero for most neurons
    unless the network sits exactly at criticality, whereas the membrane is
    graded and always carries the state.
    """

    spikes: Tensor
    membrane: Tensor
    final_v: Tensor


class LIFSubstrate(nn.Module):
    """Leaky integrate-and-fire population driven by a recurrent coupling.

    Membrane update is the explicit Euler form
    ``v <- v * (1 - dt/tau) + (dt/tau) * I`` with a hard reset to zero, matching
    the Brian2 reference that P6 reserves for the spike-train cross-check.
    """

    def __init__(
        self,
        coupling: LowRankCoupling,
        dt_ms: float = 1.0,
        tau_ms: float = 20.0,
        v_threshold: float = 1.0,
        beta: float = 10.0,
    ) -> None:
        super().__init__()
        self.coupling = coupling
        self.decay = 1.0 - dt_ms / tau_ms
        self.gain = dt_ms / tau_ms
        self.v_threshold = v_threshold
        self.beta = beta

    def forward(self, drive: Tensor) -> tuple[Tensor, Tensor]:
        """Spike raster and final membrane potential; see :meth:`rollout`."""
        result = self.rollout(drive)
        return result.spikes, result.final_v

    def rollout(self, drive: Tensor) -> Rollout:
        """Roll out over ``drive`` of shape ``(steps, batch, n_neurons)``."""
        steps, batch, n = drive.shape
        if n != self.coupling.n_neurons:
            raise ValueError(
                f"drive has {n} channels but the coupling has "
                f"{self.coupling.n_neurons} neurons"
            )

        apply = self.coupling.operator()
        v = torch.zeros(batch, n, dtype=drive.dtype, device=drive.device)
        s = torch.zeros_like(v)
        raster = []
        membrane = torch.zeros_like(v)

        for t in range(steps):
            current = drive[t] + apply(s)
            v = v * self.decay + self.gain * current
            membrane = membrane + v  # accumulated before the reset
            s = spike(v - self.v_threshold, self.beta)
            v = v * (1.0 - s)  # hard reset, differentiable through s
            raster.append(s)

        return Rollout(torch.stack(raster), membrane / steps, v)
