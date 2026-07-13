"""Evaluation metrics for PauseNet."""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr

EPSILON = 1e-8


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
