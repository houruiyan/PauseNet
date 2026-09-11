#!/usr/bin/env python3
"""PNG edition of 1.plot_count_correlation.py: identical style, PNG output.

Original docstring:
Plot observed-versus-predicted count correlations for PauseNet outputs.

Each input ``*_profiles.npz`` must contain one-dimensional
``observed_counts`` and ``predicted_counts`` arrays. The script draws one
density-colored scatter plot per input and writes a PNG image. Every scatter
point remains a vector object, so it stays sharp when the PDF is enlarged.

Examples
--------
python plot_count_correlation.py --npz /path/to/test_profiles.npz

python plot_count_correlation.py \
    --npz sample_a/test_profiles.npz sample_b/test_profiles.npz \
    --outdir ./figs \
    --shared-axis-limit
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402


FIGSIZE = (5, 4)
TICK_FONTSIZE = 12
LABEL_FONTSIZE = 14
LEGEND_FONTSIZE = 12

DISPLAY_TITLES = {
    "hek293t_groseq": "HEK293T GRO-seq",
    "hek293t_netseq": "HEK293T NET-seq",
    "hek293t_proseq": "HEK293T PRO-seq",
    "helas3_netseq": "HeLa-S3 NET-seq",
    "k562_mnetseq": "K562 mNET-seq",
    "molt4_netseq": "MOLT-4 NET-seq",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot density-colored Pearson correlations between observed and "
            "predicted log1p counts."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--npz",
        nargs="+",
        required=True,
        type=Path,
        help="One or more PauseNet *_profiles.npz files.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory; defaults to each NPZ's parent directory.",
    )
    parser.add_argument(
        "--shared-axis-limit",
        action="store_true",
        help="Use one common x/y upper limit across all supplied datasets.",
    )
    parser.add_argument(
        "--axis-max",
        type=float,
        default=None,
        help="Explicit x/y upper limit in log1p-count units.",
    )
    parser.add_argument("--density-bins", type=int, default=300)
    parser.add_argument("--point-size", type=float, default=1.5)
    return parser.parse_args()


def configure_style() -> None:
    """Apply the final-figure-small PDF style."""

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica",
                "DejaVu Sans",
            ],
            "font.size": TICK_FONTSIZE,
            "axes.labelsize": LABEL_FONTSIZE,
            "axes.titlesize": LABEL_FONTSIZE,
            "xtick.labelsize": TICK_FONTSIZE,
            "ytick.labelsize": TICK_FONTSIZE,
            "legend.fontsize": LEGEND_FONTSIZE,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("Pearson inputs must be aligned one-dimensional arrays.")
    if len(x) < 2:
        return float("nan")
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    denominator = np.sqrt(
        np.dot(x_centered, x_centered)
        * np.dot(y_centered, y_centered)
    )
    if not np.isfinite(denominator) or denominator <= 0:
        return float("nan")
    return float(np.dot(x_centered, y_centered) / denominator)


def load_counts(npz_path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    if not npz_path.is_file():
        raise FileNotFoundError(f"NPZ file does not exist: {npz_path}")

    with np.load(npz_path, allow_pickle=False) as data:
        required = {"observed_counts", "predicted_counts"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise KeyError(
                f"{npz_path} is missing required key(s): {', '.join(missing)}"
            )
        observed = np.asarray(data["observed_counts"], dtype=np.float64)
        predicted = np.asarray(data["predicted_counts"], dtype=np.float64)

    if observed.ndim != 1 or predicted.ndim != 1:
        raise ValueError(
            f"Count arrays must be one-dimensional: "
            f"{observed.shape}, {predicted.shape}"
        )
    if observed.shape != predicted.shape:
        raise ValueError(
            f"Observed/predicted count shapes differ: "
            f"{observed.shape}, {predicted.shape}"
        )

    valid = (
        np.isfinite(observed)
        & np.isfinite(predicted)
        & (observed >= 0)
        & (predicted >= 0)
    )
    discarded = int(len(observed) - valid.sum())
    observed = observed[valid]
    predicted = predicted[valid]
    if not len(observed):
        raise ValueError(f"No finite non-negative count pairs remain: {npz_path}")
    return observed, predicted, discarded


def point_density(
    x: np.ndarray,
    y: np.ndarray,
    bins: int,
) -> np.ndarray:
    if bins <= 1:
        raise ValueError("--density-bins must be greater than one.")
    histogram, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
    x_index = np.clip(np.searchsorted(x_edges, x, side="right") - 1, 0, bins - 1)
    y_index = np.clip(np.searchsorted(y_edges, y, side="right") - 1, 0, bins - 1)
    return np.log10(histogram[x_index, y_index] + 1.0)


def automatic_axis_limit(x: np.ndarray, y: np.ndarray) -> float:
    maximum = float(max(np.max(x), np.max(y)))
    return max(1.0, maximum * 1.02)


def plot_correlation(
    observed: np.ndarray,
    predicted: np.ndarray,
    output_prefix: Path,
    title: str,
    axis_limit: float,
    density_bins: int,
    point_size: float,
) -> tuple[float, int, Path]:
    x = np.log1p(observed)
    y = np.log1p(predicted)
    r_value = pearson_r(x, y)
    n = int(len(x))

    density = point_density(x, y, density_bins)
    draw_order = np.argsort(density)

    configure_style()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    scatter = ax.scatter(
        x[draw_order],
        y[draw_order],
        c=density[draw_order],
        cmap="magma",
        norm=Normalize(vmin=0.0, vmax=max(2.0, float(np.max(density)))),
        s=point_size,
        alpha=0.85,
        linewidths=0,
        rasterized=False,
        zorder=2,
    )
    ax.plot(
        (0.0, axis_limit),
        (0.0, axis_limit),
        linestyle=(0, (3, 2)),
        linewidth=0.8,
        color="#777777",
        zorder=3,
    )
    ax.set_xlim(0.0, axis_limit)
    ax.set_ylim(0.0, axis_limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Observed log1p(count)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Predicted log1p(count)", fontsize=LABEL_FONTSIZE)
    ax.set_title(title, fontsize=LABEL_FONTSIZE, pad=6)
    ax.text(
        0.04,
        0.96,
        f"Pearson $R$ = {r_value:.3f}\n$n$ = {n:,}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=LEGEND_FONTSIZE,
    )

    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.035)
    colorbar.set_label(
        r"$\log_{10}$(window density)",
        fontsize=LEGEND_FONTSIZE,
    )
    colorbar.ax.tick_params(
        labelsize=TICK_FONTSIZE,
        width=0.8,
        length=3,
    )
    colorbar.outline.set_linewidth(0.7)
    ax.tick_params(
        axis="x",
        direction="out",
        width=0.8,
        length=3,
        labelsize=TICK_FONTSIZE,
    )
    ax.tick_params(
        axis="y",
        direction="out",
        width=0.8,
        length=3,
        labelsize=TICK_FONTSIZE,
    )
    fig.tight_layout()

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix(".png")
    fig.savefig(
        png_path,
        format="png",
        dpi=600,
    )
    plt.close(fig)
    return r_value, n, png_path


def main() -> None:
    args = parse_args()
    if args.axis_max is not None and args.axis_max <= 0:
        raise ValueError("--axis-max must be positive.")
    if args.point_size <= 0:
        raise ValueError("--point-size must be positive.")

    datasets = []
    for npz_path in args.npz:
        observed, predicted, discarded = load_counts(npz_path)
        x = np.log1p(observed)
        y = np.log1p(predicted)
        datasets.append(
            {
                "path": npz_path,
                "name": npz_path.parent.name,
                "observed": observed,
                "predicted": predicted,
                "discarded": discarded,
                "auto_limit": automatic_axis_limit(x, y),
            }
        )

    shared_limit = None
    if args.axis_max is not None:
        shared_limit = float(args.axis_max)
    elif args.shared_axis_limit:
        shared_limit = max(float(dataset["auto_limit"]) for dataset in datasets)

    for dataset in datasets:
        npz_path = Path(dataset["path"])
        name = str(dataset["name"])
        outdir = args.outdir if args.outdir is not None else npz_path.parent
        axis_limit = (
            shared_limit
            if shared_limit is not None
            else float(dataset["auto_limit"])
        )
        title = DISPLAY_TITLES.get(name, name.replace("_", " "))
        output_prefix = outdir / f"{name}_count_correlation"
        r_value, n, png_path = plot_correlation(
            observed=np.asarray(dataset["observed"]),
            predicted=np.asarray(dataset["predicted"]),
            output_prefix=output_prefix,
            title=title,
            axis_limit=axis_limit,
            density_bins=args.density_bins,
            point_size=args.point_size,
        )
        print(
            f"[{name}] Pearson R(log1p)={r_value:.4f}, "
            f"n={n:,}, discarded={int(dataset['discarded']):,}"
        )
        print(f"  wrote {png_path}")

if __name__ == "__main__":
    main()
