"""Tests for the P1 subset selection rule.

The rule is pre-registered in ``.omc/specs/deep-interview-flywire-hearthstone.md``
section P1. These tests pin the two places where the probe found the prose
underdetermined:

* ``syn_count >= 5`` applies to the **aggregated neuron-to-neuron pair**, not to
  an individual per-neuropil row. The Codex bulk CSV stores one row per
  (pre, post, neuropil) triple and those rows routinely fall below 5 while the
  pair total does not.
* "majority-synapse neuropil" counts a neuron's presynaptic **and**
  postsynaptic mass.
"""

import json

import pandas as pd
import pytest

from flywire_rl import connectome as C


def _connections(rows):
    return pd.DataFrame(
        rows,
        columns=["pre_root_id", "post_root_id", "neuropil", "syn_count", "nt_type"],
    )


def test_majority_neuropil_counts_pre_and_post_mass():
    # Neuron 1 emits 3 synapses in FB but receives 10 in AVLP_R. Counting only
    # its outputs would call it a central-complex neuron; counting both does not.
    conn = _connections(
        [
            (1, 2, "FB", 3, "ACH"),
            (3, 1, "AVLP_R", 10, "ACH"),
        ]
    )
    assert C.majority_neuropil(conn)[1] == "AVLP_R"


def test_majority_neuropil_breaks_toward_the_larger_mass():
    conn = _connections(
        [
            (1, 2, "FB", 7, "ACH"),
            (1, 3, "AVLP_R", 4, "ACH"),
            (4, 1, "FB", 1, "ACH"),
        ]
    )
    assert C.majority_neuropil(conn)[1] == "FB"


def test_edge_threshold_applies_to_the_aggregated_pair_not_the_row():
    # 3 + 3 synapses across two neuropils is a 6-synapse connection and survives
    # a threshold of 5, even though neither row does.
    conn = _connections(
        [
            (1, 2, "FB", 3, "ACH"),
            (1, 2, "EB", 3, "ACH"),
        ]
    )
    edges = C.aggregate_edges(conn, min_syn=5)
    assert len(edges) == 1
    assert edges.iloc[0].syn_count == 6


def test_edge_below_threshold_after_aggregation_is_dropped():
    conn = _connections([(1, 2, "FB", 2, "ACH"), (1, 2, "EB", 2, "ACH")])
    assert len(C.aggregate_edges(conn, min_syn=5)) == 0


def test_select_subset_separates_seed_from_interface():
    # Neuron 1 lives in FB (seed). Neuron 2 lives in AVLP_R but connects to 1
    # with 6 synapses, so it joins as interface. Neuron 9 is unconnected noise.
    conn = _connections(
        [
            (1, 2, "FB", 6, "ACH"),
            (2, 2, "AVLP_R", 40, "GABA"),
            (9, 9, "AVLP_R", 40, "ACH"),
        ]
    )
    result = C.select_subset(conn, min_syn=5)

    assert result.seed_ids == {1}
    assert result.interface_ids == {2}
    assert 9 not in result.neuron_ids


def test_select_subset_keeps_only_edges_internal_to_the_subset():
    conn = _connections(
        [
            (1, 2, "FB", 6, "ACH"),
            (2, 9, "AVLP_R", 40, "ACH"),  # interface -> outsider, must be dropped
            (9, 9, "AVLP_R", 40, "ACH"),
        ]
    )
    result = C.select_subset(conn, min_syn=5)
    endpoints = set(result.edges.pre_root_id) | set(result.edges.post_root_id)

    assert endpoints <= result.neuron_ids
    assert 9 not in endpoints


def test_interface_requires_an_edge_meeting_the_threshold():
    # Neuron 2 must sit outside MB/CX or it would qualify as seed on its own
    # merits, so its mass is anchored in AVLP_R. Its only link to the seed is a
    # 4-synapse edge, one short of the threshold.
    conn = _connections(
        [
            (1, 1, "FB", 40, "ACH"),
            (1, 2, "FB", 4, "ACH"),
            (3, 2, "AVLP_R", 40, "ACH"),
        ]
    )
    result = C.select_subset(conn, min_syn=5)

    assert result.seed_ids == {1}
    assert result.interface_ids == set()


def test_dale_signs_follow_the_presynaptic_neuron():
    nt = pd.Series({1: "ACH", 2: "GABA", 3: "GLUT", 4: "DA", 5: "UNKNOWN"})
    signs = C.dale_signs([1, 2, 3, 4, 5], nt)

    assert list(signs) == [1, -1, -1, 0, 0]


def test_manifest_round_trip_detects_tampering(tmp_path):
    source = tmp_path / "connections.csv.gz"
    source.write_bytes(b"original")
    manifest = tmp_path / "manifest.json"

    C.write_manifest(tmp_path, manifest, filenames=["connections.csv.gz"])
    C.verify_manifest(tmp_path, manifest)  # must not raise

    source.write_bytes(b"tampered")
    with pytest.raises(C.ManifestMismatch):
        C.verify_manifest(tmp_path, manifest)


def test_manifest_records_a_digest_per_file(tmp_path):
    (tmp_path / "a.csv.gz").write_bytes(b"a")
    manifest = tmp_path / "manifest.json"

    C.write_manifest(tmp_path, manifest, filenames=["a.csv.gz"])
    payload = json.loads(manifest.read_text())

    assert payload["release"] == C.RELEASE
    assert len(payload["files"]["a.csv.gz"]["sha256"]) == 64
