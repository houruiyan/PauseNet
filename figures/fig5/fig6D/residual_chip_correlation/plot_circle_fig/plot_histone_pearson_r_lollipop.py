#!/usr/bin/env python3
"""Plot histone-mark Pearson correlations as a lollipop chart.

For each histone mark, Pearson R is recalculated from the per-window table
written by ``plot_residual_vs_chip_difference.py`` using these two columns:

    delta_chip_z_a_minus_b
    delta_residual_a_minus_b

The category order is fixed to the order used in the histone bar figures.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402


HISTONE_ORDER = (
    ("H3K27ac", "h3k27ac"),
    ("H3K27me3", "h3k27me3"),
    ("H3K36me3", "h3k36me3"),
    ("H3K4me3", "h3k4me3"),
    ("H3K79me2", "h3k79me2"),
    ("H3K9ac", "h3k9ac"),
    ("H3K9me3", "h3k9me3"),
    ("H4K20me1", "h4k20me1"),
)

X_COLUMN = "delta_chip_z_a_minus_b"
Y_COLUMN = "delta_residual_a_minus_b"
POSITIVE_COLOR = "#2C7FB8"
NEGATIVE_COLOR = "#E66101"
STEM_COLOR = "#686868"


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Calculate Pearson R from histone per-window TSV files and draw "
            "a lollipop chart in a fixed biological order."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=script_dir.parent,
        help="Directory containing *_residual_vs_chip_values.tsv files.",
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=script_dir / "histone_pearson_r_lollipop.pdf",
    )
    parser.add_argument(
        "--output-tsv",
        type=Path,
        default=script_dir / "histone_pearson_r_lollipop_values.tsv",
    )
    parser.add_argument(
        "--title",
        default="Histone–pausing correlation",
        help="Figure title; pass an empty string to omit it.",
    )
    return parser.parse_args()


def apply_figure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.linewidth": 1.0,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.major.size": 4.0,
            "ytick.major.size": 4.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_correlation_columns(path: Path) -> tuple[np.ndarray, np.ndarray]:
    x_values: list[float] = []
    y_values: list[float] = []

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"Missing header in {path}")
        missing = [
            column
            for column in (X_COLUMN, Y_COLUMN)
            if column not in reader.fieldnames
        ]
        if missing:
            raise KeyError(
                f"Missing column(s) {', '.join(missing)} in {path}; "
                f"available: {', '.join(reader.fieldnames)}"
            )

        for line_number, row in enumerate(reader, start=2):
            try:
                x_value = float(row[X_COLUMN])
                y_value = float(row[Y_COLUMN])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid numeric value in {path} at line {line_number}."
                ) from error
            if math.isfinite(x_value) and math.isfinite(y_value):
                x_values.append(x_value)
                y_values.append(y_value)

    if len(x_values) < 3:
        raise ValueError(f"Fewer than three finite paired rows in {path}")
    return (
        np.asarray(x_values, dtype=np.float64),
        np.asarray(y_values, dtype=np.float64),
    )


def calculate_correlations(
    input_dir: Path,
) -> tuple[list[str], np.ndarray, np.ndarray, list[Path]]:
    labels: list[str] = []
    correlations: list[float] = []
    sample_sizes: list[int] = []
    source_paths: list[Path] = []

    for label, file_prefix in HISTONE_ORDER:
        source_path = (
            input_dir
            / f"{file_prefix}_k562_minus_helas3_residual_vs_chip_values.tsv"
        )
        if not source_path.is_file():
            raise FileNotFoundError(f"Input table not found: {source_path}")
        x_values, y_values = read_correlation_columns(source_path)
        result = pearsonr(x_values, y_values)
        correlation = float(result.statistic)
        if not math.isfinite(correlation):
            raise ValueError(f"Pearson R is undefined for {label}: {source_path}")

        labels.append(label)
        correlations.append(correlation)
        sample_sizes.append(len(x_values))
        source_paths.append(source_path)
        print(f"{label}: Pearson R={correlation:.6f}, n={len(x_values):,}")

    return (
        labels,
        np.asarray(correlations, dtype=np.float64),
        np.asarray(sample_sizes, dtype=np.int64),
        source_paths,
    )


def write_summary(
    output_path: Path,
    labels: list[str],
    correlations: np.ndarray,
    sample_sizes: np.ndarray,
    source_paths: list[Path],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("histone", "pearson_r", "n_windows", "source_tsv"))
        for row in zip(
            labels, correlations, sample_sizes, source_paths, strict=True
        ):
            label, correlation, sample_size, source_path = row
            writer.writerow(
                (
                    label,
                    f"{float(correlation):.10g}",
                    int(sample_size),
                    source_path,
                )
            )
    print(f"Wrote values: {output_path}")


def plot_lollipop(
    output_pdf: Path,
    labels: list[str],
    correlations: np.ndarray,
    title: str,
) -> None:
    apply_figure_style()
    positions = np.arange(len(labels), dtype=np.float64)
    colors = np.where(
        correlations >= 0,
        POSITIVE_COLOR,
        NEGATIVE_COLOR,
    )

    fig, axis = plt.subplots(figsize=(5.0, 4.0))
    axis.axhline(
        0,
        color="#666666",
        linewidth=1.1,
        linestyle=(0, (5, 4)),
        zorder=0,
    )
    axis.vlines(
        positions,
        0,
        correlations,
        color=STEM_COLOR,
        linewidth=1.6,
        zorder=1,
    )
    axis.scatter(
        positions,
        correlations,
        s=145,
        c=colors,
        edgecolors="white",
        linewidths=0.9,
        zorder=2,
    )

    value_offset = 0.018
    for position, correlation in zip(positions, correlations, strict=True):
        if correlation >= 0:
            y_text = correlation + value_offset
            vertical_alignment = "bottom"
        else:
            y_text = correlation - value_offset
            vertical_alignment = "top"
        axis.text(
            position,
            y_text,
            f"{correlation:.3f}",
            ha="center",
            va=vertical_alignment,
            fontsize=10,
            color="#222222",
        )

    lower = min(-0.05, float(correlations.min()) - 0.07)
    upper = max(0.05, float(correlations.max()) + 0.07)
    axis.set_ylim(lower, upper)
    axis.set_xlim(-0.55, len(labels) - 0.45)
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")
    axis.set_ylabel("Pearson R")
    if title:
        axis.set_title(title, pad=8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(False)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.31, top=0.90)
    fig.savefig(output_pdf, format="pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"Wrote figure: {output_pdf}")


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_pdf = args.output_pdf.expanduser().resolve()
    output_tsv = args.output_tsv.expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)
    if output_pdf.suffix.lower() != ".pdf":
        raise ValueError("--output-pdf must end in .pdf")

    labels, correlations, sample_sizes, source_paths = calculate_correlations(
        input_dir
    )
    write_summary(
        output_tsv,
        labels,
        correlations,
        sample_sizes,
        source_paths,
    )
    plot_lollipop(output_pdf, labels, correlations, args.title)


if __name__ == "__main__":
    main()
