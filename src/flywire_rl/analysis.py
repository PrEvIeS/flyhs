"""The P5 statistical protocol: IQM, stratified bootstrap, Holm.

P5 excludes bare t-tests on means. Reinforcement-learning outcomes are
heavy-tailed and dominated by seed variance, so a mean over ten runs is moved
more by which seeds were drawn than by which arm was tested. The primary
statistic is therefore the **interquartile mean** with a **stratified
bootstrap** over seeds, following Agarwal et al. (2021), and multiplicity over
the two primary comparisons is controlled with **Holm** at alpha = 0.05.

Scores arrive as one array per arm, shaped ``(n_seeds, n_episodes)``. The
distinction matters: seeds are the unit of independent replication, episodes are
repeated measurements within a seed, and the bootstrap resamples both so the
interval reflects the uncertainty that actually applies.
"""

from __future__ import annotations

import numpy as np

REQUIRED_ARMS = ("real", "shuffled", "random")
PRIMARY_COMPARISONS = (("real", "shuffled"), ("real", "random"))
DEFAULT_ALPHA = 0.05
DEFAULT_RESAMPLES = 10_000


def iqm(values: np.ndarray) -> float:
    """Mean of the middle 50% of the sample.

    More robust than the mean, far less wasteful than the median: it keeps half
    the data instead of one point.
    """
    flat = np.sort(np.asarray(values, dtype=np.float64).ravel())
    if flat.size == 0:
        return float("nan")
    low = int(np.floor(flat.size * 0.25))
    high = int(np.ceil(flat.size * 0.75))
    return float(flat[low:high].mean())


def _resample(scores: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw seeds with replacement, then episodes within each drawn seed."""
    n_seeds, n_episodes = scores.shape
    seeds = rng.integers(0, n_seeds, size=n_seeds)
    episodes = rng.integers(0, n_episodes, size=(n_seeds, n_episodes))
    return scores[seeds[:, None], episodes]


def stratified_bootstrap(
    scores: np.ndarray,
    seed: int = 0,
    n_resamples: int = DEFAULT_RESAMPLES,
) -> np.ndarray:
    """Bootstrap distribution of the IQM for one arm."""
    scores = np.asarray(scores, dtype=np.float64)
    rng = np.random.default_rng(seed)
    return np.array(
        [iqm(_resample(scores, rng)) for _ in range(n_resamples)], dtype=np.float64
    )


def stratified_bootstrap_ci(
    scores: np.ndarray,
    seed: int = 0,
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Percentile confidence interval for an arm's IQM."""
    draws = stratified_bootstrap(scores, seed=seed, n_resamples=n_resamples)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(draws, [tail, 1.0 - tail])
    return float(low), float(high)


def probability_of_improvement(better: np.ndarray, worse: np.ndarray) -> float:
    """P(a random episode from ``better`` beats one from ``worse``), ties as half."""
    a = np.asarray(better, dtype=np.float64).ravel()
    b = np.sort(np.asarray(worse, dtype=np.float64).ravel())
    wins = np.searchsorted(b, a, side="left")
    ties = np.searchsorted(b, a, side="right") - wins
    return float((wins + 0.5 * ties).sum() / (a.size * b.size))


def holm_adjust(p_values) -> list[float]:
    """Holm step-down adjustment, monotone and capped at 1."""
    raw = np.asarray(list(p_values), dtype=np.float64)
    n = raw.size
    order = np.argsort(raw)

    adjusted = np.empty(n, dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (n - rank) * raw[index])
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def _difference_distribution(
    better: np.ndarray,
    worse: np.ndarray,
    seed: int,
    n_resamples: int,
) -> np.ndarray:
    """Bootstrap distribution of ``IQM(better) - IQM(worse)``.

    The arms are independent training runs, so they are resampled with
    independent streams rather than paired.
    """
    rng_a = np.random.default_rng(seed)
    rng_b = np.random.default_rng(seed + 1_000_003)
    a = np.asarray(better, dtype=np.float64)
    b = np.asarray(worse, dtype=np.float64)
    return np.array(
        [
            iqm(_resample(a, rng_a)) - iqm(_resample(b, rng_b))
            for _ in range(n_resamples)
        ]
    )


def compare_arms(
    arms: dict[str, np.ndarray],
    seed: int = 0,
    n_resamples: int = DEFAULT_RESAMPLES,
    alpha: float = DEFAULT_ALPHA,
    confidence: float = 0.95,
) -> dict:
    """Run the pre-registered P5 analysis.

    H0 for each comparison is ``IQM(real) - IQM(other) <= 0``, tested one-sided
    against the bootstrap distribution and rejected only if the Holm-adjusted
    p-value clears ``alpha``. The interval is reported alongside, since P5 also
    states the decision in terms of a 95% interval excluding zero.
    """
    for name in REQUIRED_ARMS:
        if name not in arms:
            raise KeyError(f"missing required arm: {name!r}")

    point = {name: iqm(scores) for name, scores in arms.items()}
    intervals = {
        name: stratified_bootstrap_ci(scores, seed, n_resamples, confidence)
        for name, scores in arms.items()
    }

    raw: list[float] = []
    entries: list[tuple[str, dict]] = []
    tail = (1.0 - confidence) / 2.0

    for index, (better, worse) in enumerate(PRIMARY_COMPARISONS):
        draws = _difference_distribution(
            arms[better], arms[worse], seed + index * 17, n_resamples
        )
        low, high = np.quantile(draws, [tail, 1.0 - tail])
        # One-sided bootstrap p-value: how often the resampled difference fails
        # to favour `better`.
        p_value = float((draws <= 0.0).mean())
        entries.append(
            (
                f"{better}_vs_{worse}",
                {
                    "difference": point[better] - point[worse],
                    "ci": (float(low), float(high)),
                    "p_value": p_value,
                    "probability_of_improvement": probability_of_improvement(
                        arms[better], arms[worse]
                    ),
                },
            )
        )
        raw.append(p_value)

    for (_, entry), adjusted in zip(entries, holm_adjust(raw)):
        entry["p_value_holm"] = adjusted
        entry["reject"] = bool(adjusted < alpha and entry["ci"][0] > 0.0)

    return {
        "iqm": point,
        "ci": intervals,
        "comparisons": dict(entries),
        "alpha": alpha,
        "confidence": confidence,
        "n_resamples": n_resamples,
        "n_seeds": {name: int(np.asarray(s).shape[0]) for name, s in arms.items()},
    }
