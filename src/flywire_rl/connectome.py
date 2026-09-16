"""FlyWire FAFB v783 loading and the P1 subset selection rule.

Pre-registered in ``.omc/specs/deep-interview-flywire-hearthstone.md`` section
P1. Two points the prose left underdetermined, resolved here and pinned by
tests:

1. The Codex bulk ``connections.csv.gz`` holds one row per
   (pre, post, neuropil) triple, so individual rows fall below 5 synapses. The
   ``syn_count >= 5`` threshold is therefore applied to the **aggregated
   neuron-to-neuron pair**, which is the FlyWire convention. On the v783 file
   every aggregated pair already clears 5, so the filter is a no-op there and
   exists to keep the rule explicit and portable.
2. "Majority-synapse neuropil" counts a neuron's presynaptic **and**
   postsynaptic mass.

Excitatory/inhibitory sign follows Dale's law: it is a property of the
*presynaptic neuron*, taken from the per-neuron ``nt_type`` in
``neurons.csv.gz`` rather than the per-edge prediction, so one neuron cannot be
half-excitatory.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RELEASE = "fafb-v783-codex"

_BUCKET = "https://storage.googleapis.com/flywire-data/codex/data/fafb/783"

SOURCE_URLS: dict[str, str] = {
    # The bucket ships nine `connections*` variants (princeton / buhmann /
    # no_threshold / ol_min_2 ...). P1 must name exactly one or its threshold
    # is meaningless; this is the canonical Codex export for v783.
    "connections.csv.gz": f"{_BUCKET}/connections.csv.gz",
    "classification.csv.gz": f"{_BUCKET}/classification.csv.gz",
    "neurons.csv.gz": f"{_BUCKET}/neurons.csv.gz",
    "cell_stats.csv.gz": f"{_BUCKET}/cell_stats.csv.gz",
}

#: Mushroom body compartments (both hemispheres) plus the unilateral central
#: complex neuropils, exactly as spelled in the v783 connectivity table.
SEED_NEUROPILS = frozenset(
    {
        f"MB_{compartment}_{side}"
        for compartment in ("CA", "ML", "PED", "VL")
        for side in ("L", "R")
    }
    | {"EB", "FB", "NO", "PB"}
)

MIN_SYN = 5

#: Acetylcholine excites; GABA and glutamate inhibit in Drosophila (GluCl).
#: The aminergic transmitters are modulatory and carry no sign of their own.
_DALE_SIGN = {"ACH": 1, "GABA": -1, "GLUT": -1, "DA": 0, "OCT": 0, "SER": 0}

_CONNECTION_DTYPES = {
    "pre_root_id": "int64",
    "post_root_id": "int64",
    "neuropil": "category",
    "syn_count": "int32",
    "nt_type": "category",
}


class ManifestMismatch(RuntimeError):
    """A source file no longer matches the digest recorded in the manifest."""


@dataclass(frozen=True)
class Subset:
    """The outcome of applying rule P1 to a connectivity table."""

    seed_ids: set[int]
    interface_ids: set[int]
    edges: pd.DataFrame

    @property
    def neuron_ids(self) -> set[int]:
        return self.seed_ids | self.interface_ids


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_manifest(
    root: Path, manifest_path: Path, filenames: list[str] | None = None
) -> dict:
    """Hash every source file so a later run can prove it read the same bytes."""
    names = filenames if filenames is not None else list(SOURCE_URLS)
    payload = {
        "release": RELEASE,
        "generated": _dt.date.today().isoformat(),
        "files": {
            name: {
                "sha256": _sha256(Path(root) / name),
                "bytes": (Path(root) / name).stat().st_size,
                "url": SOURCE_URLS.get(name, ""),
            }
            for name in names
        },
    }
    Path(manifest_path).write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def verify_manifest(root: Path, manifest_path: Path) -> None:
    """Raise :class:`ManifestMismatch` unless every file still hashes as recorded."""
    payload = json.loads(Path(manifest_path).read_text())
    for name, entry in payload["files"].items():
        path = Path(root) / name
        if not path.exists():
            raise ManifestMismatch(f"{name}: missing from {root}")
        actual = _sha256(path)
        if actual != entry["sha256"]:
            raise ManifestMismatch(
                f"{name}: expected {entry['sha256'][:12]}..., found {actual[:12]}..."
            )


def load_connections(path: Path) -> pd.DataFrame:
    """Read the per-(pre, post, neuropil) connectivity table."""
    return pd.read_csv(path, dtype=_CONNECTION_DTYPES)


def majority_neuropil(connections: pd.DataFrame) -> pd.Series:
    """Map each neuron to the neuropil holding most of its synaptic mass.

    Presynaptic and postsynaptic contributions both count, so a neuron that
    merely projects into a neuropil is not assigned to it.
    """
    pre = (
        connections.groupby(["pre_root_id", "neuropil"], observed=True)
        .syn_count.sum()
        .rename("mass")
        .reset_index()
        .rename(columns={"pre_root_id": "root_id"})
    )
    post = (
        connections.groupby(["post_root_id", "neuropil"], observed=True)
        .syn_count.sum()
        .rename("mass")
        .reset_index()
        .rename(columns={"post_root_id": "root_id"})
    )
    mass = (
        pd.concat([pre, post])
        .groupby(["root_id", "neuropil"], observed=True)
        .mass.sum()
        .reset_index()
    )
    return mass.loc[mass.groupby("root_id").mass.idxmax()].set_index("root_id").neuropil


def aggregate_edges(connections: pd.DataFrame, min_syn: int = MIN_SYN) -> pd.DataFrame:
    """Collapse per-neuropil rows into neuron-to-neuron edges, then threshold."""
    edges = (
        connections.groupby(["pre_root_id", "post_root_id"], observed=True)
        .syn_count.sum()
        .reset_index()
    )
    return edges[edges.syn_count >= min_syn].reset_index(drop=True)


def select_subset(connections: pd.DataFrame, min_syn: int = MIN_SYN) -> Subset:
    """Apply rule P1: MB+CX by majority neuropil, plus a 1-hop interface layer."""
    major = majority_neuropil(connections)
    seed = set(major.index[major.isin(SEED_NEUROPILS)])

    edges = aggregate_edges(connections, min_syn=min_syn)
    touching = edges[edges.pre_root_id.isin(seed) | edges.post_root_id.isin(seed)]
    interface = (set(touching.pre_root_id) | set(touching.post_root_id)) - seed

    members = seed | interface
    internal = edges[
        edges.pre_root_id.isin(members) & edges.post_root_id.isin(members)
    ].reset_index(drop=True)

    return Subset(seed_ids=seed, interface_ids=interface, edges=internal)


def dale_signs(neuron_ids, nt_by_neuron: pd.Series) -> np.ndarray:
    """Sign each neuron +1/-1/0 from its own transmitter, per Dale's law."""
    lookup = nt_by_neuron.to_dict()
    return np.array(
        [_DALE_SIGN.get(lookup.get(rid), 0) for rid in neuron_ids], dtype=np.int8
    )
