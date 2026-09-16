"""Guards on the ways a run can lie about itself without raising.

Every defect here shares a shape: the code completes, writes a well-formed
file, and the wrongness is visible only to someone who goes looking. That is
worse than a crash for an experiment whose output becomes evidence, so each is
pinned to a test that fails loudly if the guard is ever removed.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from flywire_rl import connectome as C
from flywire_rl.controls import Graph
from flywire_rl.experiment import (
    DECISION_WINDOW_MS,
    DECISION_WINDOW_STEPS,
    ExperimentResult,
    decision_window_steps,
    evaluate_policy,
    load_result,
    make_policy,
    save_result,
)
from flywire_rl.spiking import ShiuParams


def _result(label: str = "pilot") -> ExperimentResult:
    return ExperimentResult(
        label=label,
        scores={"real": np.array([[1.0, -1.0], [0.5, 0.25]])},
        provenance={"device": "cpu"},
        config={"n_seeds": 2},
    )


# --- the decision window must not drift silently -------------------------


def test_published_dt_still_gives_the_pre_registered_window():
    params = ShiuParams()
    assert decision_window_steps(params) == 300
    assert DECISION_WINDOW_STEPS == 300
    assert 300 * params.dt_ms == pytest.approx(DECISION_WINDOW_MS)


def test_a_dt_that_does_not_divide_the_window_raises():
    """The mirror image of the bug that deriving steps from dt was meant to fix.

    ``round`` absorbs a remainder without complaint, so dt = 0.13 would hand
    back 231 steps -- a 30.03 ms window reported as the pre-registered 30 ms.
    Silent protocol drift is the failure mode this whole module exists to
    prevent, so it must raise.
    """
    with pytest.raises(ValueError, match="does not divide"):
        decision_window_steps(replace(ShiuParams(), dt_ms=0.13))


def test_the_error_names_the_window_it_would_have_produced():
    with pytest.raises(ValueError) as caught:
        decision_window_steps(replace(ShiuParams(), dt_ms=0.13))
    # Without the realised figure the message cannot be acted on.
    assert "30.03" in str(caught.value)


@pytest.mark.parametrize("dt_ms", [0.1, 0.2, 0.3, 0.5, 1.0])
def test_dividing_dts_are_accepted_and_preserve_the_window(dt_ms):
    steps = decision_window_steps(replace(ShiuParams(), dt_ms=dt_ms))
    assert steps * dt_ms == pytest.approx(DECISION_WINDOW_MS)


# --- a result file must survive a failed rewrite --------------------------


def test_save_result_round_trips_non_trivial_values(tmp_path):
    """Non-zero, fractional and negative values, so truncation would show."""
    path = tmp_path / "pilot.json"
    save_result(_result(), path)

    reloaded = load_result(path)
    assert reloaded.label == "pilot"
    np.testing.assert_allclose(
        reloaded.scores["real"], np.array([[1.0, -1.0], [0.5, 0.25]])
    )


def test_a_failed_write_leaves_the_previous_result_intact(tmp_path, monkeypatch):
    """The expensive case: a rerun aimed at a path that holds a finished run.

    The filename is deterministic, so the file a crashing rerun destroys is
    exactly the one that cost the most to produce.
    """
    path = tmp_path / "pilot.json"
    save_result(_result(label="pilot"), path)
    before = path.read_text(encoding="utf-8")

    def explode(src, dst):
        raise OSError("simulated interruption")

    monkeypatch.setattr(os, "replace", explode)
    with pytest.raises(OSError):
        save_result(_result(label="confirmatory"), path)

    assert path.read_text(encoding="utf-8") == before
    assert json.loads(before)["label"] == "pilot"


def test_a_failed_write_leaves_no_partial_file_behind(tmp_path, monkeypatch):
    path = tmp_path / "pilot.json"
    save_result(_result(), path)

    def explode(src, dst):
        raise OSError("simulated interruption")

    monkeypatch.setattr(os, "replace", explode)
    with pytest.raises(OSError):
        save_result(_result(), path)

    assert sorted(p.name for p in tmp_path.iterdir()) == ["pilot.json"]


def test_a_successful_overwrite_leaves_no_partial_file_behind(tmp_path):
    path = tmp_path / "pilot.json"
    save_result(_result(label="pilot"), path)
    save_result(_result(label="confirmatory"), path)

    assert sorted(p.name for p in tmp_path.iterdir()) == ["pilot.json"]
    assert load_result(path).label == "confirmatory"


# --- the release tag in a filename must track the manifest ----------------


def test_release_tag_is_derived_from_the_release_string():
    """Retyped, it would keep claiming v783 after the manifest was repinned."""
    assert C.RELEASE_TAG == "v783"
    assert C.RELEASE_TAG in C.RELEASE


def test_result_filenames_use_the_derived_tag():
    source = (
        Path(__file__).resolve().parents[1] / "src" / "flywire_rl" / "run_pilot.py"
    ).read_text(encoding="utf-8")
    assert "C.RELEASE_TAG" in source
    assert '_v783_' not in source


# --- a truncated episode must not pass for a draw -------------------------


def _tiny_policy():
    rng = np.random.default_rng(0)
    n, m = 60, 600
    seen: set[tuple[int, int]] = set()
    while len(seen) < m:
        a, b = int(rng.integers(n)), int(rng.integers(n))
        if a != b:
            seen.add((a, b))
    pre, post = map(np.array, zip(*sorted(seen)))
    graph = Graph(
        n=n,
        pre=pre.astype(np.int64),
        post=post.astype(np.int64),
        weight=rng.gamma(2.0, 3.0, size=m).astype(np.float32),
        sign=rng.choice([1, -1], size=n, p=[0.78, 0.22]).astype(np.int8),
    )
    return make_policy(
        graph, torch.arange(0, 6), torch.arange(n - 4, n), seed=0, rank=2, steps=4
    )


def test_episodes_cut_off_by_the_action_cap_are_counted():
    """They score 0.0, exactly like a draw. Only the count tells them apart."""
    sink: list[int] = []
    scores = evaluate_policy(
        _tiny_policy(), n_episodes=3, seed=0, max_actions=1, truncation_sink=sink
    )

    # One action cannot finish a game, so every episode is a truncation.
    assert sink == [3]
    assert np.all(scores == 0.0)


def test_completed_episodes_are_not_counted_as_truncations():
    sink: list[int] = []
    evaluate_policy(
        _tiny_policy(), n_episodes=2, seed=0, max_actions=600, truncation_sink=sink
    )

    assert sink == [0]


def test_evaluate_policy_still_works_without_a_sink():
    """The sink is optional; existing callers must be unaffected."""
    scores = evaluate_policy(_tiny_policy(), n_episodes=2, seed=0)
    assert scores.shape == (2,)
