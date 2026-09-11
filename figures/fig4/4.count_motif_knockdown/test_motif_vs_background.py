#!/usr/bin/env python3
"""Paired statistical tests for motif knockdown versus matched background.

Primary inference uses a two-sided gene-cluster label-swap permutation test.
Within every matched pair, the effect is motif minus its own background.  All
pairs from the same positive-sequence gene are sign-flipped together to avoid
treating nearby windows from one gene as independent observations.  Holm
correction controls the family-wise error rate across the four TF comparisons.

Pair-level label-swap permutation, Wilcoxon signed-rank and sign tests are
reported as sensitivity analyses.  Predicted knockdown delta is the primary
outcome.  Matched observed log1p count is reported as a secondary association,
not as a causal perturbation result.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path("/mnt/HDD8TB/houruiyan/pausing_site")
HERE = ROOT / "4_plot_figure/fig3/4.count_motif_knockdown"
DEFAULT_INPUT = HERE / "count_motif_knockdown_seqlets.tsv.gz"
DEFAULT_MANIFEST = (
    ROOT / "3_model_explanation/DeepSHAP/hek293t_netseq/selected_manifest.tsv"
)
DEFAULT_OUTPUT = HERE / "count_motif_vs_background_statistics.tsv"
TF_ORDER = ("PATZ1", "ZNF610", "SP1", "SP2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--selected-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--permutations", type=int, default=200_000)
    parser.add_argument("--bootstraps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260904)
    return parser.parse_args()


def adjust_holm(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted_sorted = np.maximum.accumulate(
        (len(p_values) - np.arange(len(p_values))) * p_values[order]
    )
    adjusted = np.empty_like(p_values)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


def adjust_bh(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted_sorted = np.minimum.accumulate(
        (ranked * len(p_values) / np.arange(1, len(p_values) + 1))[::-1]
    )[::-1]
    adjusted = np.empty_like(p_values)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


def two_sided_sign_flip_pvalue(
    contributions: np.ndarray,
    denominator: int,
    observed: float,
    n_resamples: int,
    rng: np.random.Generator,
    batch_size: int = 5000,
) -> tuple[float, int]:
    """Monte Carlo two-sided label-swap P value with plus-one correction."""
    contributions = np.asarray(contributions, dtype=float)
    threshold = abs(float(observed)) - 1e-15
    extreme = 0
    completed = 0
    while completed < n_resamples:
        size = min(batch_size, n_resamples - completed)
        signs = rng.integers(0, 2, size=(size, len(contributions)), dtype=np.int8)
        signs = signs.astype(np.float64) * 2.0 - 1.0
        permuted = signs @ contributions / float(denominator)
        extreme += int(np.sum(np.abs(permuted) >= threshold))
        completed += size
    return (extreme + 1.0) / (n_resamples + 1.0), extreme


def bootstrap_pair_median_ci(
    differences: np.ndarray,
    n_resamples: int,
    rng: np.random.Generator,
    batch_size: int = 2000,
) -> tuple[float, float]:
    differences = np.asarray(differences, dtype=float)
    estimates = np.empty(n_resamples, dtype=float)
    completed = 0
    while completed < n_resamples:
        size = min(batch_size, n_resamples - completed)
        indices = rng.integers(0, len(differences), size=(size, len(differences)))
        estimates[completed : completed + size] = np.median(
            differences[indices], axis=1
        )
        completed += size
    return tuple(np.quantile(estimates, [0.025, 0.975]).tolist())


def bootstrap_cluster_mean_ci(
    cluster_sum: np.ndarray,
    cluster_size: np.ndarray,
    n_resamples: int,
    rng: np.random.Generator,
    batch_size: int = 5000,
) -> tuple[float, float]:
    cluster_sum = np.asarray(cluster_sum, dtype=float)
    cluster_size = np.asarray(cluster_size, dtype=float)
    estimates = np.empty(n_resamples, dtype=float)
    completed = 0
    while completed < n_resamples:
        size = min(batch_size, n_resamples - completed)
        indices = rng.integers(0, len(cluster_sum), size=(size, len(cluster_sum)))
        numerator = cluster_sum[indices].sum(axis=1)
        denominator = cluster_size[indices].sum(axis=1)
        estimates[completed : completed + size] = numerator / denominator
        completed += size
    return tuple(np.quantile(estimates, [0.025, 0.975]).tolist())


def matched_rank_biserial(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=float)
    nonzero = differences[differences != 0]
    if len(nonzero) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero), method="average")
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    return (positive - negative) / (positive + negative)


def significance_label(p_value: float) -> str:
    if p_value < 1e-4:
        return "****"
    if p_value < 1e-3:
        return "***"
    if p_value < 1e-2:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def build_paired_table(frame: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    required = {
        "match_id",
        "matched_to_tf",
        "class",
        "dataset_idx",
        "observed_count",
        "delta_predicted_log1p_count",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Input is missing columns: {sorted(missing)}")
    counts_per_pair = frame.groupby("match_id").size()
    if not (counts_per_pair == 2).all():
        raise ValueError("Every match_id must contain exactly two rows")

    source_columns = [
        "match_id",
        "matched_to_tf",
        "dataset_idx",
        "observed_count",
        "delta_predicted_log1p_count",
    ]
    positive = frame.loc[
        frame["class"] == "positive", source_columns
    ].copy()
    background = frame.loc[
        frame["class"] == "background", source_columns
    ].copy()
    if positive["match_id"].duplicated().any() or background["match_id"].duplicated().any():
        raise ValueError("Positive or background match_id is duplicated")
    positive = positive.rename(
        columns={
            "dataset_idx": "positive_dataset_idx",
            "observed_count": "positive_observed_count",
            "delta_predicted_log1p_count": "positive_predicted_delta",
        }
    )
    background = background.rename(
        columns={
            "dataset_idx": "background_dataset_idx_check",
            "observed_count": "background_observed_count",
            "delta_predicted_log1p_count": "background_predicted_delta",
        }
    )
    paired = positive[
        [
            "match_id",
            "matched_to_tf",
            "positive_dataset_idx",
            "positive_observed_count",
            "positive_predicted_delta",
        ]
    ].merge(
        background[
            [
                "match_id",
                "matched_to_tf",
                "background_dataset_idx_check",
                "background_observed_count",
                "background_predicted_delta",
            ]
        ],
        on=["match_id", "matched_to_tf"],
        validate="one_to_one",
    )

    manifest_index = manifest.set_index("array_index")
    missing_indices = set(paired["positive_dataset_idx"]) - set(manifest_index.index)
    if missing_indices:
        raise KeyError(
            "Positive dataset indices absent from manifest: "
            f"{sorted(missing_indices)[:10]}"
        )
    sample_ids = manifest_index.loc[
        paired["positive_dataset_idx"], "sample_id"
    ].astype(str).to_numpy()
    paired["positive_sample_id"] = sample_ids
    paired["gene_cluster"] = [sample_id.split("|")[0] for sample_id in sample_ids]
    if any(not value for value in paired["gene_cluster"]):
        raise ValueError("At least one gene cluster could not be parsed")

    paired["predicted_delta_difference"] = (
        paired["positive_predicted_delta"] - paired["background_predicted_delta"]
    )
    paired["positive_observed_log1p"] = np.log1p(
        paired["positive_observed_count"].to_numpy(dtype=float)
    )
    paired["background_observed_log1p"] = np.log1p(
        paired["background_observed_count"].to_numpy(dtype=float)
    )
    paired["observed_log1p_difference"] = (
        paired["positive_observed_log1p"]
        - paired["background_observed_log1p"]
    )
    return paired


def analyze_one_comparison(
    data: pd.DataFrame,
    tf_name: str,
    outcome: str,
    positive_column: str,
    background_column: str,
    difference_column: str,
    n_permutations: int,
    n_bootstraps: int,
    seed_sequence: np.random.SeedSequence,
) -> dict[str, object]:
    subset = data.loc[data["matched_to_tf"] == tf_name].copy()
    differences = subset[difference_column].to_numpy(dtype=float)
    if len(differences) == 0 or not np.all(np.isfinite(differences)):
        raise ValueError(f"{tf_name}/{outcome}: invalid or empty differences")

    cluster = (
        subset.assign(_difference=differences)
        .groupby("gene_cluster", sort=False)["_difference"]
        .agg(["sum", "size"])
    )
    mean_difference = float(np.mean(differences))
    median_difference = float(np.median(differences))
    children = seed_sequence.spawn(4)
    cluster_p, cluster_extreme = two_sided_sign_flip_pvalue(
        cluster["sum"].to_numpy(),
        len(differences),
        mean_difference,
        n_permutations,
        np.random.default_rng(children[0]),
    )
    pair_p, pair_extreme = two_sided_sign_flip_pvalue(
        differences,
        len(differences),
        mean_difference,
        n_permutations,
        np.random.default_rng(children[1]),
    )
    mean_ci_low, mean_ci_high = bootstrap_cluster_mean_ci(
        cluster["sum"].to_numpy(),
        cluster["size"].to_numpy(),
        n_bootstraps,
        np.random.default_rng(children[2]),
    )
    median_ci_low, median_ci_high = bootstrap_pair_median_ci(
        differences,
        n_bootstraps,
        np.random.default_rng(children[3]),
    )

    wilcoxon = stats.wilcoxon(
        differences,
        zero_method="wilcox",
        correction=False,
        alternative="two-sided",
        method="approx",
    )
    positive_n = int(np.sum(differences > 0))
    negative_n = int(np.sum(differences < 0))
    sign_n = positive_n + negative_n
    sign_p = (
        float(stats.binomtest(positive_n, sign_n, 0.5, alternative="two-sided").pvalue)
        if sign_n
        else 1.0
    )

    return {
        "outcome": outcome,
        "role": "primary" if outcome == "predicted_delta" else "secondary",
        "tf_name": tf_name,
        "n_pairs": len(differences),
        "n_gene_clusters": len(cluster),
        "positive_median": float(np.median(subset[positive_column])),
        "background_median": float(np.median(subset[background_column])),
        "paired_mean_difference": mean_difference,
        "cluster_bootstrap_mean_ci_low": mean_ci_low,
        "cluster_bootstrap_mean_ci_high": mean_ci_high,
        "paired_median_difference": median_difference,
        "pair_bootstrap_median_ci_low": median_ci_low,
        "pair_bootstrap_median_ci_high": median_ci_high,
        "fraction_pair_difference_positive": float(np.mean(differences > 0)),
        "matched_rank_biserial": matched_rank_biserial(differences),
        "difference_skewness": float(stats.skew(differences, bias=False)),
        "cluster_permutation_p_raw": cluster_p,
        "cluster_permutation_extreme": cluster_extreme,
        "cluster_permutation_resamples": n_permutations,
        "pair_permutation_p_raw": pair_p,
        "pair_permutation_extreme": pair_extreme,
        "wilcoxon_statistic": float(wilcoxon.statistic),
        "wilcoxon_p_raw": float(wilcoxon.pvalue),
        "sign_test_positive_n": positive_n,
        "sign_test_negative_n": negative_n,
        "sign_test_p_raw": sign_p,
    }


def main() -> int:
    args = parse_args()
    if args.permutations < 1000:
        raise ValueError("--permutations must be at least 1000")
    if args.bootstraps < 1000:
        raise ValueError("--bootstraps must be at least 1000")
    frame = pd.read_csv(args.input, sep="\t")
    manifest = pd.read_csv(args.selected_manifest, sep="\t")
    paired = build_paired_table(frame, manifest)

    analyses = [
        (
            "predicted_delta",
            "positive_predicted_delta",
            "background_predicted_delta",
            "predicted_delta_difference",
        ),
        (
            "observed_log1p_count",
            "positive_observed_log1p",
            "background_observed_log1p",
            "observed_log1p_difference",
        ),
    ]
    seeds = np.random.SeedSequence(args.seed).spawn(len(analyses) * len(TF_ORDER))
    records: list[dict[str, object]] = []
    seed_index = 0
    for outcome, positive_column, background_column, difference_column in analyses:
        for tf_name in TF_ORDER:
            records.append(
                analyze_one_comparison(
                    paired,
                    tf_name,
                    outcome,
                    positive_column,
                    background_column,
                    difference_column,
                    args.permutations,
                    args.bootstraps,
                    seeds[seed_index],
                )
            )
            seed_index += 1

    results = pd.DataFrame.from_records(records)
    for outcome in results["outcome"].unique():
        mask = results["outcome"] == outcome
        for p_column in (
            "cluster_permutation_p_raw",
            "pair_permutation_p_raw",
            "wilcoxon_p_raw",
            "sign_test_p_raw",
        ):
            p_values = results.loc[mask, p_column].to_numpy(dtype=float)
            results.loc[mask, p_column.replace("_raw", "_holm")] = adjust_holm(
                p_values
            )
            results.loc[mask, p_column.replace("_raw", "_bh")] = adjust_bh(
                p_values
            )
    results["primary_significance"] = results["cluster_permutation_p_holm"].map(
        significance_label
    )
    results["approx_multiplicative_effect_percent"] = 100.0 * np.expm1(
        results["paired_median_difference"]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, sep="\t", index=False, float_format="%.10g")
    display_columns = [
        "outcome",
        "tf_name",
        "n_pairs",
        "n_gene_clusters",
        "paired_mean_difference",
        "cluster_bootstrap_mean_ci_low",
        "cluster_bootstrap_mean_ci_high",
        "paired_median_difference",
        "cluster_permutation_p_raw",
        "cluster_permutation_p_holm",
        "primary_significance",
        "wilcoxon_p_holm",
    ]
    print(results[display_columns].to_string(index=False), flush=True)
    print(f"[out] {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
