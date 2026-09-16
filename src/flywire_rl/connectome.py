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

#: The version component of :data:`RELEASE`, for naming result files.
#:
#: Derived rather than retyped. ``verify_manifest`` checks the four file
#: digests, not this string, and provenance records only those digests -- so a
#: result filename carrying a literal ``v783`` would keep claiming v783 after
#: the manifest was repinned to another release, with nothing to catch it.
RELEASE_TAG = RELEASE.split("-")[1]

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


#: How many neurons may be missing from the transmitter table before a run is
#: refused, as a fraction.
#:
#: This guards a join gap, not a gap in knowledge. A neuron absent from
#: ``neurons.csv.gz`` while present in the connectivity table means the two
#: files disagree about which neurons exist, which is a data-assembly error and
#: should be small or zero.
MAX_ABSENT_FRACTION = 0.001


def dale_signs(
    neuron_ids,
    nt_by_neuron: pd.Series,
    max_absent_fraction: float | None = None,
    refuse_unknown_categories: bool = False,
    report: dict | None = None,
) -> np.ndarray:
    """Sign each neuron +1/-1/0 from its own transmitter, per Dale's law.

    Three different things used to collapse into the same silent 0, and they
    mean different things:

    * **absent** from the transmitter table entirely. The connectivity table and
      ``neurons.csv.gz`` disagree about which neurons exist -- a join gap, and
      an error. Refused when ``max_absent_fraction`` is given.
    * **present but null**. The release has no transmitter prediction for that
      neuron. That is the data telling the truth about what it does not know,
      not a mistake, so it is counted and reported rather than refused. On v783
      this is 705 of the 15,400 neurons in the P1 subset, 4.6 percent.
    * **present with a category Dale's table does not know**. The release knows
      a transmitter that :data:`_DALE_SIGN` does not, so the table here is
      stale. Refused when ``refuse_unknown_categories`` is set, because guessing
      a sign is worse than stopping.

    Aminergic neurons are 0 as well, but that is a real sign rather than a gap,
    and they are not counted in any of the three.

    Every refusal is opt-in. A plain call keeps the documented mapping, where
    anything unrecognised is 0, because a caller inspecting a fragment of a
    release is not making a claim about it. ``run_pilot`` opts in, because its
    output is a result file.

    ``report``, if given, receives the counts, so a run can record how much of
    its substrate was unsigned instead of leaving it to be rediscovered.
    """
    lookup = nt_by_neuron.to_dict()
    ids = list(neuron_ids)

    absent = [rid for rid in ids if rid not in lookup]
    null = [rid for rid in ids if rid in lookup and pd.isna(lookup[rid])]
    unknown_categories = sorted(
        {
            str(lookup[rid])
            for rid in ids
            if rid in lookup
            and not pd.isna(lookup[rid])
            and lookup[rid] not in _DALE_SIGN
        }
    )

    if report is not None:
        report.update(
            n_neurons=len(ids),
            absent_from_table=len(absent),
            null_transmitter=len(null),
            null_fraction=round(len(null) / len(ids), 6) if ids else 0.0,
            unknown_categories=unknown_categories,
        )

    if unknown_categories and refuse_unknown_categories:
        raise ManifestMismatch(
            f"transmitter categories not in Dale's table: {unknown_categories}. "
            "The release knows a transmitter this code does not, so _DALE_SIGN "
            "is stale; signing them 0 would silence their outgoing edges in "
            "every arm while looking like modulation."
        )

    if max_absent_fraction is not None and ids:
        if len(absent) / len(ids) > max_absent_fraction:
            raise ManifestMismatch(
                f"{len(absent)} of {len(ids)} neurons "
                f"({len(absent) / len(ids):.2%}) are absent from the transmitter "
                f"table, above the {max_absent_fraction:.2%} tolerance. The "
                "connectivity table and the transmitter table disagree about "
                "which neurons exist."
            )

    return np.array(
        [_DALE_SIGN.get(lookup.get(rid), 0) for rid in ids], dtype=np.int8
    )
