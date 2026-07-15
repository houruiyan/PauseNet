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


def plot_count_scatter(predictions: pd.DataFrame, output_path: Path) -> None:
    """Plot observed versus predicted log1p counts as a density hexbin."""
    _require_columns(predictions, {"observed_counts", "predicted_log1p_counts"}, output_path)
    observed = pd.to_numeric(predictions["observed_counts"], errors="coerce").to_numpy()
    predicted = pd.to_numeric(predictions["predicted_log1p_counts"], errors="coerce").to_numpy()
    finite = np.isfinite(observed) & np.isfinite(predicted) & (observed >= 0)
    observed_log1p = np.log1p(observed[finite])
    predicted_log1p = predicted[finite]
    if len(observed_log1p) < 2:
        raise ValueError("At least two finite observed/predicted count pairs are required.")

    plt = _load_pyplot()
    figure, axis = plt.subplots(figsize=(4.1, 3.8), dpi=300)
    maximum = float(max(observed_log1p.max(), predicted_log1p.max()))
    maximum = max(1.0, np.ceil(maximum))
    density = axis.hexbin(
        observed_log1p,
        predicted_log1p,
        gridsize=65,
        mincnt=1,
        bins="log",
        cmap="plasma",
        linewidths=0,
        extent=(0, maximum, 0, maximum),
    )
    axis.plot((0, maximum), (0, maximum), color="#4A4A4A", linestyle="--", linewidth=1.0, zorder=0)
    axis.set_xlim(0, maximum)
    axis.set_ylim(0, maximum)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Observed log1p(counts)")
    axis.set_ylabel("Predicted log1p(counts)")
    axis.text(
        0.04,
        0.96,
        f"R = {safe_pearson(observed_log1p, predicted_log1p):.3f}\nn = {len(observed_log1p):,}",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )
    colorbar = figure.colorbar(density, ax=axis, pad=0.02, fraction=0.05)
    colorbar.set_label("log10(bin count)", fontsize=8)
    colorbar.ax.tick_params(labelsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_profile_similarity(similarity: pd.DataFrame, output_path: Path) -> None:
    """Plot mean 1-JSD over the supported profile resolutions."""
    required = {"comparison", "resolution_bp", "similarity_1_minus_jsd"}
    _require_columns(similarity, required, output_path)
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
    similarity_path = evaluation_dir / f"{split}_profile_similarity.tsv"
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Missing {predictions_path}. Re-run `pausenet evaluate` without `--no-save-predictions`."
        )
    if not similarity_path.exists():
        raise FileNotFoundError(
            f"Missing {similarity_path}. Re-run `pausenet evaluate` to create profile similarity results."
        )

    count_path = output_dir / f"{split}_count_scatter.{image_format}"
    profile_path = output_dir / f"{split}_profile_similarity.{image_format}"
    plot_count_scatter(pd.read_csv(predictions_path, sep="\t"), count_path)
    plot_profile_similarity(pd.read_csv(similarity_path, sep="\t"), profile_path)
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
