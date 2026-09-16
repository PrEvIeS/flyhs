"""Tests for the masked low-rank coupling and the LIF substrate.

The equivalence tests here are not incidental: amendment A1 requires the dense
and sparse code paths to be proven numerically identical before any confirmatory
run, because development happens on MPS (which lacks efficient sparse matmul)
while the confirmatory series runs on CUDA. Without this, "topology matters"
could reduce to "the two hosts computed different things".
"""

import pytest
import torch

from flywire_rl.spiking import CouplingMode, LIFSubstrate, LowRankCoupling, spike


def _coupling(mode, rank=4, seed=0):
    pre = torch.tensor([0, 1, 2, 3, 0, 4])
    post = torch.tensor([1, 2, 3, 4, 4, 0])
    weights = torch.tensor([0.5, -0.25, 1.0, 0.75, -0.5, 0.25])
    torch.manual_seed(seed)
    coupling = LowRankCoupling(5, pre, post, weights, rank=rank, mode=mode)
    # Non-zero B, or the adapter contributes nothing and the test is vacuous.
    torch.manual_seed(seed)
    with torch.no_grad():
        coupling.B.copy_(torch.randn_like(coupling.B) * 0.1)
    return coupling


@pytest.mark.parametrize("mode", [CouplingMode.SPARSE, CouplingMode.DENSE])
def test_couplings_agree_with_the_gather_reference(mode):
    s = torch.randn(3, 5)
    reference = _coupling(CouplingMode.GATHER).operator()(s)
    other = _coupling(mode).operator()(s)

    torch.testing.assert_close(other, reference, rtol=1e-5, atol=1e-6)


def test_adapter_starts_as_a_no_op():
    pre = torch.tensor([0, 1])
    post = torch.tensor([1, 0])
    weights = torch.tensor([0.5, -0.5])
    torch.manual_seed(0)
    coupling = LowRankCoupling(2, pre, post, weights, rank=4)

    # B is initialised to zero, so the substrate must start as the pure connectome.
    torch.testing.assert_close(coupling.effective_weights(), weights)


def test_delta_is_confined_to_existing_edges():
    # A dense B @ A would couple every pair. Only the two real edges may move.
    pre = torch.tensor([0, 2])
    post = torch.tensor([1, 3])
    weights = torch.zeros(2)
    torch.manual_seed(0)
    coupling = LowRankCoupling(4, pre, post, weights, rank=2, mode=CouplingMode.DENSE)
    with torch.no_grad():
        coupling.B.copy_(torch.randn_like(coupling.B))

    s = torch.eye(4)
    dense = coupling.operator()(s)  # row i holds the outgoing weights of neuron i

    nonzero = (dense.abs() > 0).nonzero().tolist()
    assert sorted(nonzero) == [[0, 1], [2, 3]]


def test_trainable_parameter_count_is_two_n_r():
    coupling = _coupling(CouplingMode.GATHER, rank=8)
    trainable = sum(p.numel() for p in coupling.parameters() if p.requires_grad)

    assert trainable == 2 * coupling.n_neurons * 8


def test_frozen_weights_are_buffers_not_parameters():
    coupling = _coupling(CouplingMode.GATHER)
    names = {name for name, _ in coupling.named_parameters()}

    assert names == {"B", "A"}


def test_surrogate_gradient_is_nonzero_away_from_threshold():
    v = torch.tensor([-2.0, 0.0, 2.0], requires_grad=True)
    spike(v).sum().backward()

    # A true Heaviside derivative is zero everywhere; the surrogate must not be.
    assert torch.all(v.grad > 0)


def test_spike_forward_is_a_hard_threshold():
    v = torch.tensor([-0.1, 0.0, 0.1])
    assert torch.equal(spike(v), torch.tensor([0.0, 0.0, 1.0]))


def test_gradient_reaches_the_adapter_through_the_rollout():
    coupling = _coupling(CouplingMode.GATHER)
    substrate = LIFSubstrate(coupling)
    drive = torch.ones(6, 2, 5) * 50.0

    raster, _ = substrate(drive)
    raster.sum().backward()

    assert coupling.A.grad is not None
    assert coupling.A.grad.abs().sum() > 0


def test_a_silent_substrate_gives_the_adapter_no_gradient():
    """The failure mode that P4's firing-rate calibration exists to prevent.

    The coupling multiplies presynaptic spikes, so d(out)/d(w_eff) = s_pre. If
    an arm never spikes, its adapter receives exactly zero learning signal --
    not a small one. An arm silenced by unnormalised gain would therefore look
    like a topology result while actually measuring nothing at all.
    """
    coupling = _coupling(CouplingMode.GATHER)
    substrate = LIFSubstrate(coupling)
    drive = torch.ones(6, 2, 5) * 3.0  # v saturates near 0.77, threshold is 1.0

    raster, _ = substrate(drive)
    raster.sum().backward()

    assert raster.sum() == 0
    assert coupling.A.grad.abs().sum() == 0


def test_substrate_spikes_under_supra_threshold_drive():
    coupling = _coupling(CouplingMode.GATHER)
    substrate = LIFSubstrate(coupling)

    silent, _ = substrate(torch.zeros(10, 2, 5))
    driven, _ = substrate(torch.ones(10, 2, 5) * 50.0)

    assert silent.sum() == 0
    assert driven.sum() > 0


def test_rollout_rejects_mismatched_channel_count():
    substrate = LIFSubstrate(_coupling(CouplingMode.GATHER))

    with pytest.raises(ValueError, match="channels"):
        substrate(torch.zeros(4, 2, 7))
