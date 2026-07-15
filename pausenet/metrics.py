"""Evaluation metrics for PauseNet."""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr

EPSILON = 1e-8
PROFILE_RESOLUTIONS = (1, 5, 10, 20)


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
    observed = np.asarray(observed_profiles, dtype=np.float64)
    predicted = np.asarray(predicted_profiles, dtype=np.float64)
    masks = np.asarray(profile_masks, dtype=bool)
    if observed.ndim != 2 or predicted.shape != observed.shape:
        raise ValueError("Observed and predicted profiles must have the same 2D shape.")
    if len(masks) != len(observed):
        raise ValueError("Profile masks must contain one value per profile.")

    valid = masks & np.isfinite(observed).all(axis=1) & np.isfinite(predicted).all(axis=1)
    valid &= observed.sum(axis=1) > 0
    observed = np.maximum(observed[valid], 0.0)
    predicted = np.maximum(predicted[valid], 0.0)
    if len(observed) == 0:
        return [
            {
                "comparison": comparison,
                "resolution_bp": resolution,
                "similarity_1_minus_jsd": float("nan"),
                "n": 0,
            }
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
    for comparison, (left, right) in comparisons.items():
        comparison_valid = (left.sum(axis=1) > 0) & (right.sum(axis=1) > 0)
        for resolution in resolutions:
            distances = js_distance_rows(left[comparison_valid], right[comparison_valid], resolution)
            rows.append(
                {
                    "comparison": comparison,
                    "resolution_bp": resolution,
                    "similarity_1_minus_jsd": float(1.0 - np.mean(distances))
                    if len(distances)
                    else float("nan"),
                    "n": int(comparison_valid.sum()),
                }
            )
    return rows
