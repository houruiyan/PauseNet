#!/usr/bin/env python3
"""Calculate four TF Pearson correlations and draw a lollipop chart."""

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

from plot_histone_pearson_r_lollipop import (  # noqa: E402
    NEGATIVE_COLOR,
    POSITIVE_COLOR,
    STEM_COLOR,
    apply_figure_style,
    read_correlation_columns,
)


TF_ORDER = (
    ("CTCF", "ctcf"),
    ("CHD2", "chd2"),
    ("SWI", "swi"),
    ("TFIIF", "tfiif"),
)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Calculate Pearson R from four TF per-window TSV files and draw "
            "a lollipop chart in the order CTCF, CHD2, SWI, TFIIF."
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
        default=script_dir / "tf_pearson_r_lollipop.pdf",
    )
    parser.add_argument(
        "--output-tsv",
        type=Path,
        default=script_dir / "tf_pearson_r_lollipop_values.tsv",
    )
    parser.add_argument(
        "--title",
        default="TF–pausing correlation",
        help="Figure title; pass an empty string to omit it.",
    )
    return parser.parse_args()


def calculate_correlations(
    input_dir: Path,
) -> tuple[list[str], np.ndarray, np.ndarray, list[Path]]:
    labels: list[str] = []
    correlations: list[float] = []
    sample_sizes: list[int] = []
    source_paths: list[Path] = []

    for label, file_prefix in TF_ORDER:
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
        writer.writerow(("tf", "pearson_r", "n_windows", "source_tsv"))
        for label, correlation, sample_size, source_path in zip(
            labels,
            correlations,
            sample_sizes,
            source_paths,
            strict=True,
        ):
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

    value_offset = max(0.008, float(np.ptp(correlations)) * 0.06)
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

    padding = max(0.05, float(np.ptp(correlations)) * 0.22)
    lower = min(-0.05, float(correlations.min()) - padding)
    upper = max(0.05, float(correlations.max()) + padding)
    axis.set_ylim(lower, upper)
    axis.set_xlim(-0.45, len(labels) - 0.55)
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")
    axis.set_ylabel("Pearson R")
    if title:
        axis.set_title(title, pad=8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(False)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.25, top=0.90)
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
