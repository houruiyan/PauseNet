#!/usr/bin/env python3
"""Plot pairwise count correlations between nascent-RNA technologies.

Compares per-window counts of NET-seq / PRO-seq / GRO-seq across the same
26,040 windows, using the same window order in each PauseNet
``*_profiles.npz`` file. Plotting style is copied from
fig1/1.plot_count_correlation.py (density-colored magma scatter, vector
points, equal aspect ratio, dashed diagonal, Pearson R + n annotation).

Examples
--------
python plot_tech_correlation.py \
    --netseq hek293t_netseq/test_profiles.npz \
    --proseq hek293t_proseq/test_profiles.npz \
    --groseq hek293t_groseq/test_profiles.npz \
    --cell HEK293T --outdir ./figs --combined

Use --source predicted to compare model predictions instead of measured
counts (the ProCapNet Fig. 6C-style analysis).
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

TECH_ORDER = ("netseq", "proseq", "groseq")
TECH_LABELS = {
    "netseq": "NET-seq",
    "proseq": "PRO-seq",
    "groseq": "GRO-seq",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot pairwise density-colored Pearson correlations of "
            "log1p counts between NET-seq / PRO-seq / GRO-seq."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--netseq",
        required=True,
        type=Path,
        help="PauseNet *_profiles.npz for NET-seq.",
    )
    parser.add_argument(
        "--proseq",
        required=True,
        type=Path,
        help="PauseNet *_profiles.npz for PRO-seq.",
    )
    parser.add_argument(
        "--groseq",
        required=True,
        type=Path,
        help="PauseNet *_profiles.npz for GRO-seq.",
    )
    parser.add_argument(
        "--source",
        choices=("observed", "predicted"),
        default="observed",
        help="Which count array to compare across technologies.",
    )
    parser.add_argument("--cell", default="HEK293T")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("./tech_correlation_out"),
        help="Output directory for PDFs and summary.",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Also write a 3-panel combined figure.",
    )
    parser.add_argument(
        "--shared-axis-limit",
        action="store_true",
        help="Use one common axis limit across all three pair plots.",
    )
    parser.add_argument("--density-bins", type=int, default=300)
    parser.add_argument("--point-size", type=float, default=1.5)
    return parser.parse_args()


def configure_style() -> None:
    """Apply the final-figure-small PDF style (same as fig1 script)."""

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
        np.dot(x_centered, x_centered) * np.dot(y_centered, y_centered)
    )
    if not np.isfinite(denominator) or denominator <= 0:
        return float("nan")
    return float(np.dot(x_centered, y_centered) / denominator)


def load_counts(npz_path: Path, source: str) -> tuple[np.ndarray, int]:
    """Load one count array from a PauseNet *_profiles.npz file."""
    if not npz_path.is_file():
        raise FileNotFoundError(f"NPZ file does not exist: {npz_path}")

    key = f"{source}_counts"
    with np.load(npz_path, allow_pickle=False) as data:
        if key not in data.files:
            raise KeyError(f"{npz_path} is missing required key: {key}")
        counts = np.asarray(data[key], dtype=np.float64)

    if counts.ndim != 1:
        raise ValueError(f"Count array must be one-dimensional: {counts.shape}")

    valid = np.isfinite(counts) & (counts >= 0)
    discarded = int(len(counts) - valid.sum())
    counts = counts[valid]
    if not len(counts):
        raise ValueError(f"No finite non-negative counts remain: {npz_path}")
    return counts, discarded


def point_density(x: np.ndarray, y: np.ndarray, bins: int) -> np.ndarray:
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
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    axis_limit: float,
    density_bins: int,
    point_size: float,
    show_colorbar: bool = True,
) -> tuple[float, int]:
    """Draw one density-colored scatter on `ax`; x/y are log1p counts."""

    r_value = pearson_r(x, y)
    n = int(len(x))

    density = point_density(x, y, density_bins)
    draw_order = np.argsort(density)

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
    ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
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

    if show_colorbar:
        colorbar = ax.figure.colorbar(scatter, ax=ax, fraction=0.046, pad=0.035)
        colorbar.set_label(r"$\log_{10}$(window density)", fontsize=LEGEND_FONTSIZE)
        colorbar.ax.tick_params(labelsize=TICK_FONTSIZE, width=0.8, length=3)
        colorbar.outline.set_linewidth(0.7)

    ax.tick_params(axis="x", direction="out", width=0.8, length=3,
                   labelsize=TICK_FONTSIZE)
    ax.tick_params(axis="y", direction="out", width=0.8, length=3,
                   labelsize=TICK_FONTSIZE)
    return r_value, n


def main() -> None:
    args = parse_args()
    if args.density_bins <= 1:
        raise ValueError("--density-bins must be greater than one.")
    if args.point_size <= 0:
        raise ValueError("--point-size must be positive.")

    paths = {
        "netseq": args.netseq,
        "proseq": args.proseq,
        "groseq": args.groseq,
    }

    # load and align counts across technologies
    counts: dict[str, np.ndarray] = {}
    for tech in TECH_ORDER:
        arr, discarded = load_counts(paths[tech], args.source)
        counts[tech] = arr
        print(
            f"[{tech}] loaded {len(arr):,} windows "
            f"(discarded {discarded:,}) from {paths[tech]}"
        )

    lengths = {tech: len(arr) for tech, arr in counts.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(
            "Window counts differ between technologies "
            f"({lengths}); inputs must share the same window set/order."
        )

    log_counts = {tech: np.log1p(arr) for tech, arr in counts.items()}

    pairs = [("netseq", "proseq"), ("netseq", "groseq"), ("proseq", "groseq")]

    auto_limits = {
        pair: automatic_axis_limit(log_counts[pair[0]], log_counts[pair[1]])
        for pair in pairs
    }
    shared_limit = max(auto_limits.values()) if args.shared_axis_limit else None

    configure_style()
    args.outdir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for pair in pairs:
        x_tech, y_tech = pair
        x_label, y_label = TECH_LABELS[x_tech], TECH_LABELS[y_tech]
        axis_limit = (
            shared_limit if shared_limit is not None else auto_limits[pair]
        )

        fig, ax = plt.subplots(figsize=FIGSIZE)
        r_value, n = plot_correlation(
            ax=ax,
            x=log_counts[x_tech],
            y=log_counts[y_tech],
            title=f"{args.cell} {x_label} vs {y_label}",
            xlabel=f"{x_label} log1p(count)",
            ylabel=f"{y_label} log1p(count)",
            axis_limit=axis_limit,
            density_bins=args.density_bins,
            point_size=args.point_size,
        )
        fig.tight_layout()
        tag = f"{x_tech}_vs_{y_tech}"
        pdf_path = args.outdir / f"{args.cell.lower()}_{tag}_count_correlation.pdf"
        fig.savefig(pdf_path, format="pdf", dpi=600)
        plt.close(fig)
        summary_rows.append((x_label, y_label, r_value, n))
        print(f"[{tag}] Pearson R(log1p)={r_value:.4f}, n={n:,}")
        print(f"  wrote {pdf_path}")

    if args.combined:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, pair in zip(axes, pairs):
            x_tech, y_tech = pair
            axis_limit = (
                shared_limit if shared_limit is not None else auto_limits[pair]
            )
            plot_correlation(
                ax=ax,
                x=log_counts[x_tech],
                y=log_counts[y_tech],
                title=f"{args.cell} {TECH_LABELS[x_tech]} vs {TECH_LABELS[y_tech]}",
                xlabel=f"{TECH_LABELS[x_tech]} log1p(count)",
                ylabel=f"{TECH_LABELS[y_tech]} log1p(count)",
                axis_limit=axis_limit,
                density_bins=args.density_bins,
                point_size=args.point_size,
            )
        fig.tight_layout()
        pdf_path = (
            args.outdir / f"{args.cell.lower()}_tech_pairwise_count_correlation.pdf"
        )
        fig.savefig(pdf_path, format="pdf", dpi=600)
        plt.close(fig)
        print(f"  wrote {pdf_path}")

    summary_path = args.outdir / f"{args.cell.lower()}_tech_correlation_summary.tsv"
    with summary_path.open("w") as fh:
        fh.write("tech_x\ttech_y\tpearson_r_log1p\tn\n")
        for x_label, y_label, r_value, n in summary_rows:
            fh.write(f"{x_label}\t{y_label}\t{r_value:.4f}\t{n}\n")
    print(f"  wrote {summary_path}")


if __name__ == "__main__":
    main()
