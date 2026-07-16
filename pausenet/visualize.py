"""Create publication-ready plots from saved PauseNet evaluation outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import safe_pearson


COMPARISON_ORDER = ("Pseudoreplicates", "PauseNet", "Random profile")
COMPARISON_COLORS = {
    "Pseudoreplicates": "#2AB7B0",
    "PauseNet": "#E81E38",
    "Random profile": "#6646B7",
}


def _load_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "Visualization requires matplotlib. Install it with "
            "`python -m pip install -e '.[figures]'`."
        ) from error
    return plt


def _require_columns(table: pd.DataFrame, columns: set[str], path: Path) -> None:
    missing = columns.difference(table.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"{path} is missing required columns: {missing_text}")


def _log10_window_density(
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    bins: int = 190,
) -> np.ndarray:
    """Return the log10 count of the 2D histogram bin containing each point."""
    if bins < 1:
        raise ValueError("bins must be at least 1")
    if x_values.shape != y_values.shape:
        raise ValueError("x_values and y_values must have the same shape")

    counts, x_edges, y_edges = np.histogram2d(x_values, y_values, bins=bins)
    x_indices = np.clip(
        np.searchsorted(x_edges, x_values, side="right") - 1,
        0,
        counts.shape[0] - 1,
    )
    y_indices = np.clip(
        np.searchsorted(y_edges, y_values, side="right") - 1,
        0,
        counts.shape[1] - 1,
    )
    return np.log10(np.maximum(counts[x_indices, y_indices], 1.0))


def plot_count_scatter(predictions: pd.DataFrame, output_path: Path) -> None:
    """Plot observed versus predicted log1p counts as density-colored points."""
    _require_columns(predictions, {"observed_counts", "predicted_log1p_counts"}, output_path)
    observed = pd.to_numeric(predictions["observed_counts"], errors="coerce").to_numpy()
    predicted = pd.to_numeric(predictions["predicted_log1p_counts"], errors="coerce").to_numpy()
    finite = np.isfinite(observed) & np.isfinite(predicted) & (observed >= 0)
    observed_log1p = np.log1p(observed[finite])
    predicted_log1p = predicted[finite]
    if len(observed_log1p) < 2:
        raise ValueError("At least two finite observed/predicted count pairs are required.")

    plt = _load_pyplot()
    figure, axis = plt.subplots(figsize=(5.0, 4.0), dpi=300)
    maximum = float(max(observed_log1p.max(), predicted_log1p.max()))
    maximum = max(1.0, np.ceil(maximum))
    padding = maximum * 0.025
    log_density = _log10_window_density(observed_log1p, predicted_log1p)
    draw_order = np.argsort(log_density, kind="stable")
    points = axis.scatter(
        observed_log1p[draw_order],
        predicted_log1p[draw_order],
        c=log_density[draw_order],
        cmap="plasma",
        s=2.5,
        alpha=0.9,
        edgecolors="none",
        rasterized=True,
    )
    axis.plot(
        (0, maximum),
        (0, maximum),
        color="#666666",
        linestyle=(0, (4, 3)),
        linewidth=1.0,
        zorder=0,
    )
    axis.set_xlim(-padding, maximum + padding)
    axis.set_ylim(-padding, maximum + padding)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Observed log1p(counts)", fontsize=13)
    axis.set_ylabel("Predicted log1p(counts)", fontsize=13)
    axis.tick_params(axis="both", labelsize=10)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.text(
        0.04,
        0.96,
        f"R = {safe_pearson(observed_log1p, predicted_log1p):.3f}\nn = {len(observed_log1p):,}",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=12,
    )
    colorbar_axis = axis.inset_axes([0.50, 0.87, 0.38, 0.055])
    colorbar = figure.colorbar(points, cax=colorbar_axis, orientation="horizontal")
    colorbar.ax.set_title("log10(window density)", fontsize=9, pad=2)
    colorbar.ax.tick_params(axis="x", labelsize=8, length=2.5, pad=1)
    maximum_density = int(np.floor(float(log_density.max())))
    colorbar.set_ticks(np.arange(maximum_density + 1, dtype=float))
    figure.subplots_adjust(left=0.16, right=0.98, bottom=0.16, top=0.98)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_profile_similarity(similarity: pd.DataFrame, output_path: Path) -> None:
    """Plot mean 1-JSD over the supported profile resolutions."""
    required = {"comparison", "resolution_bp", "similarity_1_minus_jsd"}
    _require_columns(similarity, required, output_path)
    if "count_threshold" in similarity.columns:
        plot_profile_similarity_by_count_threshold(similarity, output_path)
        return

    plt = _load_pyplot()
    figure, axis = plt.subplots(figsize=(4.8, 3.8), dpi=300)
    for comparison in COMPARISON_ORDER:
        rows = similarity.loc[similarity["comparison"] == comparison].copy()
        if rows.empty:
            continue
        rows["resolution_bp"] = pd.to_numeric(rows["resolution_bp"], errors="coerce")
        rows["similarity_1_minus_jsd"] = pd.to_numeric(
            rows["similarity_1_minus_jsd"], errors="coerce"
        )
        rows = rows.dropna(subset=["resolution_bp", "similarity_1_minus_jsd"]).sort_values("resolution_bp")
        if rows.empty:
            continue
        axis.plot(
            rows["resolution_bp"],
            rows["similarity_1_minus_jsd"],
            color=COMPARISON_COLORS[comparison],
            marker="o",
            markersize=5,
            linewidth=2.0,
            label=comparison,
        )
    axis.set_xticks((1, 5, 10, 20))
    axis.set_xlim(0.5, 20.5)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Resolution (bp)")
    axis.set_ylabel("Profile similarity (1 - JSD)")
    axis.legend(frameon=False, loc="upper left", handlelength=1.8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_profile_similarity_by_count_threshold(
    similarity: pd.DataFrame, output_path: Path
) -> None:
    """Plot one profile-similarity panel per observed count threshold."""
    required = {
        "count_threshold",
        "comparison",
        "resolution_bp",
        "similarity_1_minus_jsd",
        "n",
    }
    _require_columns(similarity, required, output_path)
    table = similarity.copy()
    for column in ("count_threshold", "resolution_bp", "similarity_1_minus_jsd", "n"):
        table[column] = pd.to_numeric(table[column], errors="coerce")
    table = table.dropna(subset=["count_threshold", "resolution_bp"])
    thresholds = sorted(int(value) for value in table["count_threshold"].unique())
    if not thresholds:
        raise ValueError("At least one finite count threshold is required.")

    plt = _load_pyplot()
    figure, axes = plt.subplots(
        1,
        len(thresholds),
        figsize=(2.55 * len(thresholds), 2.8),
        dpi=300,
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes.ravel()
    legend_handles = []
    legend_labels = []
    for axis, threshold in zip(axes, thresholds):
        threshold_rows = table.loc[table["count_threshold"] == threshold]
        panel_n = threshold_rows.loc[
            threshold_rows["comparison"] == "PauseNet", "n"
        ].dropna()
        if panel_n.empty:
            panel_n = threshold_rows["n"].dropna()
        n_text = int(panel_n.iloc[0]) if not panel_n.empty else 0
        for comparison in COMPARISON_ORDER:
            rows = threshold_rows.loc[
                threshold_rows["comparison"] == comparison
            ].dropna(subset=["resolution_bp", "similarity_1_minus_jsd"])
            rows = rows.sort_values("resolution_bp")
            if rows.empty:
                continue
            (line,) = axis.plot(
                rows["resolution_bp"],
                rows["similarity_1_minus_jsd"],
                color=COMPARISON_COLORS[comparison],
                marker="o",
                markersize=4.5,
                linewidth=1.8,
                label=comparison,
            )
            if comparison not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(comparison)
        axis.set_title(f"count >= {threshold:,}\nn={n_text:,}", fontsize=9, pad=5)
        axis.set_xticks((1, 5, 10, 20))
        axis.set_xlim(0.5, 20.5)
        axis.set_ylim(0, 1)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    axes[0].set_ylabel("Profile similarity (1 - JSD)")
    figure.supxlabel("Resolution (bp)", y=0.14)
    if legend_handles:
        figure.legend(
            legend_handles,
            legend_labels,
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=len(legend_labels),
            handlelength=1.8,
            columnspacing=1.2,
            handletextpad=0.4,
        )
    figure.subplots_adjust(left=0.075, right=0.99, top=0.78, bottom=0.28, wspace=0.34)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def visualize_evaluation(
    evaluation_dir: str | Path,
    split: str = "test",
    output_dir: str | Path | None = None,
    image_format: str = "png",
) -> tuple[Path, Path]:
    """Create both standard PauseNet evaluation figures from saved files."""
    evaluation_dir = Path(evaluation_dir)
    output_dir = Path(output_dir) if output_dir is not None else evaluation_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = evaluation_dir / f"{split}_predictions.tsv"
    threshold_similarity_path = (
        evaluation_dir / f"{split}_profile_similarity_by_count_threshold.tsv"
    )
    similarity_path = evaluation_dir / f"{split}_profile_similarity.tsv"
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Missing {predictions_path}. Re-run `pausenet evaluate` without `--no-save-predictions`."
        )
    if not threshold_similarity_path.exists() and not similarity_path.exists():
        raise FileNotFoundError(
            "Missing profile-similarity results. Re-run `pausenet evaluate` "
            "to create them."
        )

    count_path = output_dir / f"{split}_count_scatter.{image_format}"
    profile_path = output_dir / f"{split}_profile_similarity.{image_format}"
    plot_count_scatter(pd.read_csv(predictions_path, sep="\t"), count_path)
    selected_similarity_path = (
        threshold_similarity_path if threshold_similarity_path.exists() else similarity_path
    )
    plot_profile_similarity(pd.read_csv(selected_similarity_path, sep="\t"), profile_path)
    return count_path, profile_path


def add_visualize_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--evaluation-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir")
    parser.add_argument("--format", choices=("png", "pdf", "svg"), default="png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize saved PauseNet evaluation outputs.")
    add_visualize_args(parser)
    args = parser.parse_args()
    count_path, profile_path = visualize_evaluation(
        evaluation_dir=args.evaluation_dir,
        split=args.split,
        output_dir=args.output_dir,
        image_format=args.format,
    )
    print(f"Count scatter: {count_path}")
    print(f"Profile similarity: {profile_path}")


if __name__ == "__main__":
    main()
