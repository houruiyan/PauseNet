"""Evaluation metrics for PauseNet."""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr

EPSILON = 1e-8
PROFILE_RESOLUTIONS = (1, 5, 10, 20)
PROFILE_COUNT_THRESHOLDS = (0, 100, 200, 500)


def safe_pearson(observed, predicted) -> float:
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    mask = np.isfinite(observed) & np.isfinite(predicted)
    if mask.sum() < 2:
        return float("nan")
    if np.std(observed[mask]) == 0 or np.std(predicted[mask]) == 0:
        return float("nan")
    return float(pearsonr(observed[mask], predicted[mask]).statistic)


def bin_profiles(values: np.ndarray, resolution: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if resolution == 1:
        return values
    usable = values.shape[1] // resolution * resolution
    return values[:, :usable].reshape(values.shape[0], usable // resolution, resolution).sum(axis=-1)


def js_distance_rows(observed: np.ndarray, predicted: np.ndarray, resolution: int = 1) -> np.ndarray:
    observed = bin_profiles(observed, resolution)
    predicted = bin_profiles(predicted, resolution)
    observed = observed / np.maximum(observed.sum(axis=1, keepdims=True), EPSILON)
    predicted = predicted / np.maximum(predicted.sum(axis=1, keepdims=True), EPSILON)
    midpoint = 0.5 * (observed + predicted)
    left = np.sum(
        np.where(
            observed > 0,
            observed * np.log2((observed + EPSILON) / (midpoint + EPSILON)),
            0.0,
        ),
        axis=1,
    )
    right = np.sum(
        np.where(
            predicted > 0,
            predicted * np.log2((predicted + EPSILON) / (midpoint + EPSILON)),
            0.0,
        ),
        axis=1,
    )
    return np.sqrt(np.maximum(0.5 * (left + right), 0.0))


def profile_similarity_summary(
    observed_profiles: np.ndarray,
    predicted_profiles: np.ndarray,
    profile_masks: np.ndarray,
    resolutions: tuple[int, ...] = PROFILE_RESOLUTIONS,
    seed: int = 2025,
) -> list[dict[str, float | int | str]]:
    """Summarize profile similarity against reproducible reference baselines.

    Pseudoreplicates are produced by an independent binomial split of each
    observed profile. The random baseline permutes positions within each
    observed profile, retaining its total signal and value distribution.
    """
    rows = profile_similarity_by_count_threshold(
        observed_profiles,
        predicted_profiles,
        profile_masks,
        resolutions=resolutions,
        count_thresholds=(0,),
        seed=seed,
    )
    return [
        {key: value for key, value in row.items() if key != "count_threshold"}
        for row in rows
    ]


def profile_similarity_by_count_threshold(
    observed_profiles: np.ndarray,
    predicted_profiles: np.ndarray,
    profile_masks: np.ndarray,
    observed_counts: np.ndarray | None = None,
    resolutions: tuple[int, ...] = PROFILE_RESOLUTIONS,
    count_thresholds: tuple[int, ...] = PROFILE_COUNT_THRESHOLDS,
    seed: int = 2025,
) -> list[dict[str, float | int | str]]:
    """Summarize profile similarity after observed-count filtering.

    Pseudoreplicates and random profiles are generated once before applying
    the count thresholds. Consequently, a window that is present at multiple
    thresholds keeps the same reference profiles in every panel.
    """
    observed = np.asarray(observed_profiles, dtype=np.float64)
    predicted = np.asarray(predicted_profiles, dtype=np.float64)
    masks = np.asarray(profile_masks, dtype=bool)
    if observed.ndim != 2 or predicted.shape != observed.shape:
        raise ValueError("Observed and predicted profiles must have the same 2D shape.")
    if len(masks) != len(observed):
        raise ValueError("Profile masks must contain one value per profile.")
    if observed_counts is None:
        counts = observed.sum(axis=1)
    else:
        counts = np.asarray(observed_counts, dtype=np.float64)
        if counts.ndim != 1 or len(counts) != len(observed):
            raise ValueError("Observed counts must contain one value per profile.")

    thresholds = tuple(dict.fromkeys(int(value) for value in count_thresholds))
    if not thresholds:
        raise ValueError("At least one count threshold is required.")
    if any(value < 0 for value in thresholds):
        raise ValueError("Count thresholds must be non-negative integers.")

    valid = masks & np.isfinite(observed).all(axis=1) & np.isfinite(predicted).all(axis=1)
    valid &= np.isfinite(counts) & (counts >= 0)
    valid &= observed.sum(axis=1) > 0
    observed = np.maximum(observed[valid], 0.0)
    predicted = np.maximum(predicted[valid], 0.0)
    observed_counts_valid = counts[valid]
    if len(observed) == 0:
        return [
            {
                "count_threshold": threshold,
                "comparison": comparison,
                "resolution_bp": resolution,
                "similarity_1_minus_jsd": float("nan"),
                "n": 0,
            }
            for threshold in thresholds
            for comparison in ("Pseudoreplicates", "PauseNet", "Random profile")
            for resolution in resolutions
        ]

    rng = np.random.default_rng(seed)
    observed_integer = np.rint(observed).astype(np.int64)
    pseudoreplicate_a = rng.binomial(observed_integer, 0.5).astype(np.float64)
    pseudoreplicate_b = observed_integer.astype(np.float64) - pseudoreplicate_a

    random_profiles = np.empty_like(observed)
    for index, profile in enumerate(observed):
        random_profiles[index] = rng.permutation(profile)

    comparisons = {
        "Pseudoreplicates": (pseudoreplicate_a, pseudoreplicate_b),
        "PauseNet": (observed, predicted),
        "Random profile": (observed, random_profiles),
    }
    rows = []
    for threshold in thresholds:
        threshold_valid = observed_counts_valid >= threshold
        for comparison, (left, right) in comparisons.items():
            comparison_valid = threshold_valid & (left.sum(axis=1) > 0) & (right.sum(axis=1) > 0)
            for resolution in resolutions:
                distances = js_distance_rows(
                    left[comparison_valid], right[comparison_valid], resolution
                )
                rows.append(
                    {
                        "count_threshold": threshold,
                        "comparison": comparison,
                        "resolution_bp": resolution,
                        "similarity_1_minus_jsd": float(1.0 - np.mean(distances))
                        if len(distances)
                        else float("nan"),
                        "n": int(comparison_valid.sum()),
                    }
                )
    return rows
