"""Environment acceptance checks for protocol amendment A1.

A1 requires a single lockfile to work on both hosts: the M1 Pro development
machine (MPS) and the CUDA desktop that runs the confirmatory series. These
tests assert what must hold on whichever host they run on, and report which
accelerator that host actually offers.
"""

import importlib

import pytest

REQUIRED_PACKAGES = [
    "torch",
    "numpy",
    "scipy",
    "pandas",
    "networkx",
    "pyarrow",
    "brian2",
]


@pytest.mark.parametrize("name", REQUIRED_PACKAGES)
def test_required_package_imports(name):
    importlib.import_module(name)


def test_an_accelerator_is_available():
    """P6 learning runs need MPS (dev host) or CUDA (confirmatory host)."""
    import torch

    mps = torch.backends.mps.is_available()
    cuda = torch.cuda.is_available()
    assert mps or cuda, "neither MPS nor CUDA is available on this host"


def test_brian2_runs_a_two_neuron_simulation():
    """P6 reserves Brian2 CPU as the spike-train cross-check reference."""
    import brian2 as b2

    b2.start_scope()
    tau = 10 * b2.ms  # noqa: F841 - resolved from scope by brian2's equations
    neurons = b2.NeuronGroup(
        2,
        "dv/dt = (1.1 - v) / tau : 1",
        threshold="v > 1",
        reset="v = 0",
        method="exact",
    )
    neurons.v = 0
    spikes = b2.SpikeMonitor(neurons)
    b2.run(100 * b2.ms)

    assert spikes.num_spikes > 0, "constant drive above threshold produced no spikes"
