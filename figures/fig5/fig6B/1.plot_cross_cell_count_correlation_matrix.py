#!/usr/bin/env python3
"""Plot a cross-cell-line observed/predicted count correlation matrix.

Matrix definition
-----------------
* Lower triangle: observed counts in column cell line vs. observed counts
  in row cell line.
* Upper triangle: predicted counts in column cell line vs. predicted counts
  in row cell line.
* Diagonal: observed vs. predicted Pearson correlation within each cell line.

All correlations and scatter coordinates use log1p(counts). Windows are matched
by their manifest identifiers, not by NPZ row number.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
import numpy as np


DEFAULT_NPZ = [
    "/mnt/HDD8TB/houruiyan/pausing_site/2_train_model/evaluate/"
    "hek293t_netseq/test_profiles.npz",
    "/mnt/HDD8TB/houruiyan/pausing_site/2_train_model/evaluate/"
    "k562_mnetseq/test_profiles.npz",
    "/mnt/HDD8TB/houruiyan/pausing_site/2_train_model/evaluate/"
    "helas3_netseq/test_profiles.npz",
    "/mnt/HDD8TB/houruiyan/pausing_site/2_train_model/evaluate/"
    "molt4_netseq/test_profiles.npz",
]
DEFAULT_LABELS = ["HEK293T", "K562", "HeLa-S3", "MOLT-4"]
DEFAULT_OUTPUT = (
    "/mnt/HDD8TB/houruiyan/pausing_site/4_plot_figure/fig6/fig6B/"
    "cross_cell_count_correlation_matrix.pdf"
)

WINDOW_FIELDS = (
    "manifest_sample_id",
    "manifest_chrom",
    "manifest_output_start",
    "manifest_output_end",
    "manifest_strand",
)


def set_figure_style() -> None:
    """Apply the project's final-figure-small typography."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 12,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def as_text(values: np.ndarray) -> np.ndarray:
    """Convert manifest text fields to Unicode safely."""
    values = np.asarray(values)
    if values.dtype.kind == "S":
        return np.char.decode(values, "utf-8")
    return values.astype(str)


def make_window_keys(data: np.lib.npyio.NpzFile) -> List[Tuple[str, str, int, int, str]]:
    sample_id = as_text(data["manifest_sample_id"])
    chrom = as_text(data["manifest_chrom"])
    start = np.asarray(data["manifest_output_start"], dtype=np.int64)
    end = np.asarray(data["manifest_output_end"], dtype=np.int64)
    strand = as_text(data["manifest_strand"])
    return [
        (str(sid), str(ch), int(st), int(en), str(sd))
        for sid, ch, st, en, sd in zip(sample_id, chrom, start, end, strand)
    ]


def load_dataset(path: Path) -> Dict[str, object]:
    required = {"observed_counts", "predicted_counts", *WINDOW_FIELDS}
    with np.load(path, allow_pickle=False) as data:
        missing = sorted(required.difference(data.files))
        if missing:
            raise KeyError(f"{path} is missing required NPZ keys: {', '.join(missing)}")

        observed = np.asarray(data["observed_counts"], dtype=np.float64).reshape(-1)
        predicted = np.asarray(data["predicted_counts"], dtype=np.float64).reshape(-1)
        keys = make_window_keys(data)
        if "profile_masks" in data.files:
            profile_mask = np.asarray(data["profile_masks"], dtype=bool).reshape(-1)
        else:
            profile_mask = np.ones(observed.size, dtype=bool)

    n = observed.size
    if predicted.size != n or profile_mask.size != n or len(keys) != n:
        raise ValueError(f"Inconsistent array lengths in {path}")
    if len(set(keys)) != n:
        raise ValueError(f"Window identifiers are not unique in {path}")
    return {
        "observed": observed,
        "predicted": predicted,
        "profile_mask": profile_mask,
        "keys": keys,
    }


def align_datasets(
    datasets: Sequence[Dict[str, object]],
) -> Tuple[List[Dict[str, np.ndarray]], List[Tuple[str, str, int, int, str]]]:
    """Align every dataset to the ordered intersection of manifest windows."""
    first_keys = datasets[0]["keys"]
    common = set(first_keys)
    for dataset in datasets[1:]:
        common.intersection_update(dataset["keys"])
    ordered_common = [key for key in first_keys if key in common]
    if not ordered_common:
        raise ValueError("The supplied NPZ files do not share any manifest windows")

    aligned: List[Dict[str, np.ndarray]] = []
    for dataset in datasets:
        index = {key: i for i, key in enumerate(dataset["keys"])}
        take = np.fromiter(
            (index[key] for key in ordered_common),
            dtype=np.int64,
            count=len(ordered_common),
        )
        aligned.append(
            {
                "observed": np.asarray(dataset["observed"])[take],
                "predicted": np.asarray(dataset["predicted"])[take],
                "profile_mask": np.asarray(dataset["profile_mask"])[take],
            }
        )
    return aligned, ordered_common


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[valid], dtype=np.float64)
    y = np.asarray(y[valid], dtype=np.float64)
    if x.size < 2:
        return float("nan")
    x -= x.mean()
    y -= y.mean()
    denominator = np.sqrt(np.dot(x, x) * np.dot(y, y))
    if denominator == 0:
        return float("nan")
    return float(np.dot(x, y) / denominator)


def point_density(
    x: np.ndarray, y: np.ndarray, axis_limit: float, bins: int
) -> np.ndarray:
    histogram, x_edges, y_edges = np.histogram2d(
        x,
        y,
        bins=bins,
        range=((0.0, axis_limit), (0.0, axis_limit)),
    )
    x_index = np.clip(
        np.searchsorted(x_edges, x, side="right") - 1, 0, bins - 1
    )
    y_index = np.clip(
        np.searchsorted(y_edges, y, side="right") - 1, 0, bins - 1
    )
    return np.log10(histogram[x_index, y_index] + 1.0)


def draw_scatter(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    axis_limit: float,
    density_bins: int,
    point_size: float,
) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    density = point_density(x, y, axis_limit, density_bins)
    order = np.argsort(density)
    ax.scatter(
        x[order],
        y[order],
        c=density[order],
        cmap="magma",
        s=point_size,
        linewidths=0,
        rasterized=True,
    )
    # Ordinary least-squares fit of y on x.  Draw only the part that falls
    # inside the visible axes instead of using a y = x reference line.
    x_centered = x - x.mean()
    denominator = np.dot(x_centered, x_centered)
    if denominator > 0:
        slope = float(np.dot(x_centered, y - y.mean()) / denominator)
        intercept = float(y.mean() - slope * x.mean())
        x_line = np.linspace(float(x.min()), float(x.max()), 400)
        y_line = intercept + slope * x_line
        visible = (
            (x_line >= 0.0)
            & (x_line <= axis_limit)
            & (y_line >= 0.0)
            & (y_line <= axis_limit)
        )
        ax.plot(
            x_line[visible],
            y_line[visible],
            color="0.45",
            linewidth=0.9,
            linestyle=(0, (3, 3)),
            zorder=3,
        )
    ax.set_xlim(0.0, axis_limit)
    ax.set_ylim(0.0, axis_limit)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return pearson_r(x, y)


def correlation_matrix(
    observed: Sequence[np.ndarray], predicted: Sequence[np.ndarray]
) -> np.ndarray:
    n = len(observed)
    result = np.full((n, n), np.nan, dtype=np.float64)
    for row in range(n):
        for col in range(n):
            if row > col:
                result[row, col] = pearson_r(observed[col], observed[row])
            elif row < col:
                result[row, col] = pearson_r(predicted[col], predicted[row])
            else:
                result[row, col] = pearson_r(observed[row], predicted[row])
    return result


def plot_matrix(
    observed: Sequence[np.ndarray],
    predicted: Sequence[np.ndarray],
    labels: Sequence[str],
    output_pdf: Path,
    density_bins: int,
    point_size: float,
    figure_width: float,
    figure_height: float,
) -> np.ndarray:
    n = len(labels)
    all_values = np.concatenate([*observed, *predicted])
    finite = all_values[np.isfinite(all_values)]
    if finite.size == 0:
        raise ValueError("No finite count values remain after filtering")
    axis_limit = max(1.0, float(np.ceil(finite.max())))
    r_values = correlation_matrix(observed, predicted)
    diagonal_r = np.diag(r_values)
    finite_diag = diagonal_r[np.isfinite(diagonal_r)]
    if finite_diag.size == 0:
        color_norm = Normalize(vmin=0.0, vmax=1.0)
    else:
        vmin = max(0.0, np.floor(finite_diag.min() * 20.0) / 20.0)
        vmax = min(1.0, np.ceil(finite_diag.max() * 20.0) / 20.0)
        if vmax <= vmin:
            vmax = min(1.0, vmin + 0.05)
        color_norm = Normalize(vmin=vmin, vmax=vmax)
    block_cmap = plt.get_cmap("Blues")

    fig, axes = plt.subplots(
        n,
        n,
        figsize=(figure_width, figure_height),
        gridspec_kw={"wspace": 0.10, "hspace": 0.10},
    )
    axes = np.atleast_2d(axes)
    for row in range(n):
        for col in range(n):
            ax = axes[row, col]
            if row == col:
                r = r_values[row, col]
                ax.set_facecolor(block_cmap(color_norm(r)))
                text_color = "white" if color_norm(r) >= 0.55 else "black"
                ax.text(
                    0.5,
                    0.5,
                    f"R = {r:.3f}",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    color=text_color,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
            else:
                if row > col:
                    x, y = observed[col], observed[row]
                else:
                    x, y = predicted[col], predicted[row]
                r = draw_scatter(
                    ax,
                    x,
                    y,
                    axis_limit=axis_limit,
                    density_bins=density_bins,
                    point_size=point_size,
                )
                ax.text(
                    0.04,
                    0.96,
                    f"R = {r:.3f}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=10,
                    fontweight="bold",
                )
                ax.tick_params(
                    labelbottom=(row == n - 1),
                    labelleft=(col == 0),
                )

    # Wider-than-tall panels and generous outer margins prevent row/column
    # labels from touching or being clipped by the PDF page boundaries.
    fig.subplots_adjust(left=0.15, right=0.87, bottom=0.17, top=0.97)
    for row, label in enumerate(labels):
        bbox = axes[row, 0].get_position()
        fig.text(
            bbox.x0 - 0.030,
            0.5 * (bbox.y0 + bbox.y1),
            label,
            ha="right",
            va="center",
            fontsize=14,
        )
    for col, label in enumerate(labels):
        bbox = axes[-1, col].get_position()
        fig.text(
            0.5 * (bbox.x0 + bbox.x1),
            bbox.y0 - 0.055,
            label,
            ha="right",
            va="top",
            rotation=38,
            rotation_mode="anchor",
            fontsize=14,
        )

    matrix_bbox_top = axes[0, -1].get_position()
    matrix_bbox_bottom = axes[-1, -1].get_position()
    cbar_ax = fig.add_axes(
        [
            0.905,
            matrix_bbox_bottom.y0,
            0.020,
            matrix_bbox_top.y1 - matrix_bbox_bottom.y0,
        ]
    )
    scalar_mappable = plt.cm.ScalarMappable(norm=color_norm, cmap=block_cmap)
    scalar_mappable.set_array([])
    cbar = fig.colorbar(scalar_mappable, cax=cbar_ax)
    cbar.set_label("Pearson R of log1p(counts)", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, format="pdf", dpi=600)
    plt.close(fig)
    return r_values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot lower-triangle observed/observed, upper-triangle "
            "predicted/predicted, and diagonal observed/predicted correlations."
        )
    )
    parser.add_argument(
        "--npz",
        nargs="+",
        default=DEFAULT_NPZ,
        help="One test_profiles.npz file per cell line.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=DEFAULT_LABELS,
        help="Cell-line labels in the same order as --npz.",
    )
    parser.add_argument(
        "--output-pdf",
        default=DEFAULT_OUTPUT,
        help="Output PDF path.",
    )
    parser.add_argument(
        "--min-observed-count",
        type=float,
        default=0.0,
        help=(
            "Keep a matched window only when every cell line has an observed "
            "count greater than or equal to this value (default: 0)."
        ),
    )
    parser.add_argument(
        "--use-common-profile-mask",
        action="store_true",
        help="Also require profile_masks=True in every cell line.",
    )
    parser.add_argument(
        "--density-bins",
        type=int,
        default=160,
        help="Number of bins per axis for density coloring.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=1.0,
        help="Scatter point area in points squared.",
    )
    parser.add_argument(
        "--figure-width",
        type=float,
        default=10.0,
        help="Figure width in inches (default: 10).",
    )
    parser.add_argument(
        "--figure-height",
        type=float,
        default=7.2,
        help="Figure height in inches (default: 7.2).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.npz) != len(args.labels):
        raise ValueError("--npz and --labels must contain the same number of values")
    if len(args.npz) < 2:
        raise ValueError("At least two NPZ files are required")
    if args.min_observed_count < 0:
        raise ValueError("--min-observed-count must be non-negative")
    if args.density_bins < 10:
        raise ValueError("--density-bins must be at least 10")
    if args.point_size <= 0 or args.figure_width <= 0 or args.figure_height <= 0:
        raise ValueError(
            "--point-size, --figure-width, and --figure-height must be positive"
        )

    set_figure_style()
    raw = [load_dataset(Path(path)) for path in args.npz]
    datasets, shared_keys = align_datasets(raw)

    common_valid = np.ones(len(shared_keys), dtype=bool)
    for dataset in datasets:
        obs = dataset["observed"]
        pred = dataset["predicted"]
        common_valid &= np.isfinite(obs) & np.isfinite(pred)
        common_valid &= (obs >= args.min_observed_count) & (pred >= 0)
        if args.use_common_profile_mask:
            common_valid &= dataset["profile_mask"]

    kept = int(common_valid.sum())
    if kept < 2:
        raise ValueError(
            f"Only {kept} matched windows remain after filtering; at least 2 are needed"
        )
    observed = [
        np.log1p(dataset["observed"][common_valid]) for dataset in datasets
    ]
    predicted = [
        np.log1p(dataset["predicted"][common_valid]) for dataset in datasets
    ]

    output_pdf = Path(args.output_pdf)
    r_values = plot_matrix(
        observed=observed,
        predicted=predicted,
        labels=args.labels,
        output_pdf=output_pdf,
        density_bins=args.density_bins,
        point_size=args.point_size,
        figure_width=args.figure_width,
        figure_height=args.figure_height,
    )

    print(f"Matched windows shared by all datasets: {len(shared_keys):,}")
    print(f"Windows plotted after filtering: {kept:,}")
    print("Correlation matrix:")
    print("\t" + "\t".join(args.labels))
    for label, row in zip(args.labels, r_values):
        print(label + "\t" + "\t".join(f"{value:.3f}" for value in row))
    print(f"Saved: {output_pdf}")


if __name__ == "__main__":
    main()
