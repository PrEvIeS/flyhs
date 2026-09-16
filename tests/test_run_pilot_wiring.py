"""Wiring tests for the CUDA entry point.

``run_pilot`` had no tests at all, which mattered more than the line count
suggests: it is the only module that touches the real FlyWire tables, and the
rest of the suite runs on synthetic fixtures, so a green run said nothing about
it. The two properties pinned here are the ones that fail silently.

The manifest check must happen *before* anything is read, or a different
release reaches a results file stamped with the pinned provenance. And labels
must arrive at ``select_io_indices`` in neuron order: that function takes a
positional array, so handing it a root_id-indexed Series returns wrong indices
and raises nothing -- the interface would simply be the wrong neurons.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from flywire_rl import run_pilot
from flywire_rl.connectome import ManifestMismatch, Subset

# Deliberately not sorted: build_real_graph sorts them, and every downstream
# index is a position in that sorted order. A fixture already in order could
# not catch a mapping that quietly used input order instead.
IDS = [500, 100, 300, 200, 400]
SORTED_IDS = sorted(IDS)


@pytest.fixture
def fake_tables(monkeypatch, tmp_path):
    """Replace every read of the real tables, recording the order of calls."""
    calls: list[str] = []

    edges = pd.DataFrame(
        {
            "pre_root_id": [100, 300, 500],
            "post_root_id": [200, 400, 100],
            "syn_count": [7.0, 9.0, 11.0],
        }
    )

    def verify_manifest(raw, manifest):
        calls.append("verify_manifest")

    def load_connections(path):
        calls.append("load_connections")
        return edges

    def select_subset(connections):
        calls.append("select_subset")
        return Subset(seed_ids=set(IDS), interface_ids=set(), edges=edges)

    def dale_signs(neuron_ids, nt_by_neuron, **kwargs):
        calls.append("dale_signs")
        # One sign per neuron, in the order the caller passed the ids.
        return np.array([1 if int(r) % 200 else -1 for r in neuron_ids], dtype=np.int8)

    seen_labels: dict[str, np.ndarray] = {}

    def select_io_indices(labels):
        calls.append("select_io_indices")
        seen_labels["labels"] = labels
        return np.array([0, 1], dtype=np.int64), np.array([3, 4], dtype=np.int64)

    # classification.csv.gz is indexed by root_id and is *not* in neuron order;
    # that mismatch is the whole point of the reindex under test.
    classes = pd.DataFrame(
        {
            "root_id": [400, 100, 500, 200, 300],
            "super_class": ["output", "input", "output", "input", "central"],
        }
    )
    neurons = pd.DataFrame({"root_id": IDS, "nt_type": ["ACH"] * len(IDS)})

    def read_csv(path, usecols=None, **kwargs):
        calls.append(f"read_csv:{'classification' if 'classification' in str(path) else 'neurons'}")
        frame = classes if "classification" in str(path) else neurons
        return frame[list(usecols)] if usecols else frame

    monkeypatch.setattr(run_pilot.C, "verify_manifest", verify_manifest)
    monkeypatch.setattr(run_pilot.C, "load_connections", load_connections)
    monkeypatch.setattr(run_pilot.C, "select_subset", select_subset)
    monkeypatch.setattr(run_pilot.C, "dale_signs", dale_signs)
    monkeypatch.setattr(run_pilot, "select_io_indices", select_io_indices)
    monkeypatch.setattr(run_pilot.pd, "read_csv", read_csv)
    monkeypatch.setattr(run_pilot, "DATA_RAW", tmp_path)
    monkeypatch.setattr(run_pilot, "MANIFEST", tmp_path / "manifest.json")

    return calls, seen_labels


def test_the_manifest_is_verified_before_anything_is_read(fake_tables):
    calls, _ = fake_tables
    run_pilot.build_real_graph()

    assert calls[0] == "verify_manifest"


def test_a_manifest_mismatch_stops_the_run_before_any_table_is_loaded(
    fake_tables, monkeypatch
):
    """A hard failure, not a warning: the alternative is mislabelled provenance."""
    calls, _ = fake_tables

    def refuse(raw, manifest):
        calls.append("verify_manifest")
        raise ManifestMismatch("digest does not match")

    monkeypatch.setattr(run_pilot.C, "verify_manifest", refuse)

    with pytest.raises(ManifestMismatch):
        run_pilot.build_real_graph()

    assert calls == ["verify_manifest"]


def test_labels_reach_select_io_indices_in_neuron_order(fake_tables):
    """The silent one: select_io_indices takes positions, not a root_id index.

    classification.csv.gz is indexed by root_id and arrives in its own order.
    Passing it through unreindexed returns wrong interface indices without
    raising, so the experiment would drive and read the wrong neurons.
    """
    _, seen = fake_tables
    run_pilot.build_real_graph()

    labels = seen["labels"]
    assert isinstance(labels, np.ndarray)
    assert list(labels) == ["input", "input", "central", "output", "output"]


def test_edges_are_remapped_to_positions_in_the_sorted_id_order(fake_tables):
    real, _, _ = run_pilot.build_real_graph()

    assert real.n == len(SORTED_IDS)
    # (100 -> 200), (300 -> 400), (500 -> 100) under [100, 200, 300, 400, 500]
    assert list(real.pre) == [0, 2, 4]
    assert list(real.post) == [1, 3, 0]
    assert list(real.weight) == [7.0, 9.0, 11.0]


def test_dale_signs_are_requested_in_the_same_sorted_order(fake_tables):
    real, _, _ = run_pilot.build_real_graph()

    expected = np.array([1 if r % 200 else -1 for r in SORTED_IDS], dtype=np.int8)
    assert list(real.sign) == list(expected)


def test_interface_indices_come_back_as_tensors(fake_tables):
    _, inputs, outputs = run_pilot.build_real_graph()

    assert isinstance(inputs, torch.Tensor)
    assert isinstance(outputs, torch.Tensor)
    assert list(inputs) == [0, 1]
    assert list(outputs) == [3, 4]


def test_pick_device_honours_an_explicit_request():
    """A run pinned to cpu must not quietly move to the card, or vice versa."""
    assert run_pilot.pick_device("cpu") == "cpu"
    assert run_pilot.pick_device("cuda") == "cuda"
    assert run_pilot.pick_device("auto") in {"cpu", "cuda", "mps"}


# --- an unsigned neuron must not pass for a modulatory one ----------------


def _nt(pairs) -> pd.Series:
    return pd.Series(
        [v for _, v in pairs], index=[k for k, _ in pairs], name="nt_type"
    )


def test_known_transmitters_keep_their_signs():
    from flywire_rl.connectome import dale_signs

    signs = dale_signs(
        [1, 2, 3, 4], _nt([(1, "ACH"), (2, "GABA"), (3, "GLUT"), (4, "DA")])
    )

    # DA is genuinely 0: modulatory, not missing.
    assert list(signs) == [1, -1, -1, 0]


def test_the_check_is_off_unless_a_caller_asks_for_it():
    """The mapping's documented behaviour is unchanged: unrecognised is 0.

    Strictness belongs on the path that produces evidence, which passes the
    tolerance explicitly, not on every caller inspecting a fragment.
    """
    from flywire_rl.connectome import dale_signs

    signs = dale_signs([1, 2], _nt([(1, "ACH"), (2, "UNKNOWN")]))

    assert list(signs) == [1, 0]


def test_a_few_unsigned_neurons_are_tolerated():
    """A release can leave a handful unpredicted; refusing on one is useless."""
    from flywire_rl.connectome import MAX_UNSIGNED_FRACTION, dale_signs

    ids = list(range(200))
    table = _nt([(i, "ACH") for i in ids if i != 0])  # one absent, i.e. 0.5%

    signs = dale_signs(ids, table, max_unsigned_fraction=MAX_UNSIGNED_FRACTION)
    assert signs[0] == 0
    assert set(signs[1:]) == {1}


def test_a_join_gap_above_tolerance_raises():
    """The failure it guards is invisible: sign 0 mutes every outgoing edge."""
    from flywire_rl.connectome import MAX_UNSIGNED_FRACTION, ManifestMismatch, dale_signs

    ids = list(range(100))
    table = _nt([(i, "ACH") for i in ids[:90]])  # ten percent absent

    with pytest.raises(ManifestMismatch, match="no usable transmitter"):
        dale_signs(ids, table, max_unsigned_fraction=MAX_UNSIGNED_FRACTION)


def test_an_unknown_transmitter_category_is_reported_by_name():
    from flywire_rl.connectome import MAX_UNSIGNED_FRACTION, ManifestMismatch, dale_signs

    ids = list(range(100))
    table = _nt([(i, "ACH" if i < 90 else "HISTAMINE") for i in ids])

    with pytest.raises(ManifestMismatch) as caught:
        dale_signs(ids, table, max_unsigned_fraction=MAX_UNSIGNED_FRACTION)
    assert "HISTAMINE" in str(caught.value)


def test_the_message_separates_absent_neurons_from_unknown_categories():
    """They need different fixes: a join gap versus a transmitter table gap."""
    from flywire_rl.connectome import MAX_UNSIGNED_FRACTION, ManifestMismatch, dale_signs

    ids = list(range(100))
    table = _nt([(i, "ACH" if i < 90 else "HISTAMINE") for i in ids[:95]])

    with pytest.raises(ManifestMismatch) as caught:
        dale_signs(ids, table, max_unsigned_fraction=MAX_UNSIGNED_FRACTION)
    message = str(caught.value)
    assert "5 absent from the transmitter table" in message
    assert "HISTAMINE" in message


def test_the_pilot_path_asks_for_the_strict_check():
    """The library default is permissive; the evidence path must not be."""
    source = (
        Path(__file__).resolve().parents[1] / "src" / "flywire_rl" / "run_pilot.py"
    ).read_text(encoding="utf-8")

    assert "max_unsigned_fraction=C.MAX_UNSIGNED_FRACTION" in source
