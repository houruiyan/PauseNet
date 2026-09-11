#!/usr/bin/env python3
"""Plot HeLaS3 histone-marker adjusted-R2 comparisons.

For each histone marker, plot:

1. HeLaS3 histone only
2. PauseNet + HeLaS3 histone (matched cell type)
3. PauseNet + K562 histone (mismatched cell type)

The gray and red dashed lines show PauseNet only and PauseNet plus all
HeLaS3 histone markers, respectively. Reference-line values are read from
the batch-analysis summaries rather than entered manually.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULT_DIR = SCRIPT_DIR / "helas3_histone_r2_results"
DEFAULT_INPUT = DEFAULT_RESULT_DIR / "helas3_histone_r2_summary.tsv"
DEFAULT_ALL_SUMMARY = (
    DEFAULT_RESULT_DIR / "helas3_all_histones_r2_summary.tsv"
)
DEFAULT_OUTPUT = (
    DEFAULT_RESULT_DIR / "helas3_histone_r2_grouped_bars.pdf"
)

HISTONE_ONLY_COLOR = "#f9c00c"
PAUSENET_HELAS3_COLOR = "#00b9f1"
PAUSENET_K562_COLOR = "#f9320c"
PAUSENET_ONLY_LINE_COLOR = "#777777"
PAUSENET_ALL_HISTONES_LINE_COLOR = "#d62728"

MARKER_COLUMN = "histone_marker"
PAUSENET_ONLY_COLUMN = "PauseNet_only_R2_adj"
BAR_COLUMNS = (
    "HeLaS3_histone_only_R2_adj",
    "PauseNet_plus_HeLaS3_histone_R2_adj",
    "PauseNet_plus_K562_histone_R2_adj",
)
BAR_LABELS = (
    "Histone only",
    "PauseNet + HeLaS3",
    "PauseNet + K562",
)
BAR_COLORS = (
    HISTONE_ONLY_COLOR,
    PAUSENET_HELAS3_COLOR,
    PAUSENET_K562_COLOR,
)
ALL_HISTONES_COLUMN = "PauseNet_plus_all_HeLaS3_histones_R2_adj"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-tsv",
        default=str(DEFAULT_INPUT),
        help=f"Per-marker summary TSV (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--all-summary-tsv",
        default=str(DEFAULT_ALL_SUMMARY),
        help=(
            "All-histone summary TSV used for the red reference line "
            f"(default: {DEFAULT_ALL_SUMMARY})."
        ),
    )
    parser.add_argument(
        "--output-pdf",
        default=str(DEFAULT_OUTPUT),
        help=f"Output PDF (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--histone-order",
        action="append",
        help=(
            "Optional marker order/subset. Repeat the option or provide "
            "comma-separated marker names."
        ),
    )
    parser.add_argument(
        "--title",
        default="HeLaS3",
        help="Figure title (default: HeLaS3); use an empty string to hide.",
    )
    parser.add_argument(
        "--show-values",
        action="store_true",
        help="Write adjusted-R2 values above the bars.",
    )
    parser.add_argument(
        "--ymin",
        type=float,
        default=None,
        help="Optional lower y limit; determined automatically by default.",
    )
    parser.add_argument(
        "--ymax",
        type=float,
        default=None,
        help="Optional upper y limit; determined automatically by default.",
    )
    return parser.parse_args()


def parse_marker_order(values: Sequence[str] | None) -> list[str]:
    if values is None:
        return []
    result: list[str] = []
    for value in values:
        result.extend(
            item.strip() for item in value.split(",") if item.strip()
        )
    if len(result) != len(set(result)):
        raise ValueError("--histone-order contains duplicate marker names.")
    return result


def load_marker_summary(
    path_text: str,
    requested_order: Sequence[str],
) -> tuple[list[str], dict[str, np.ndarray], float]:
    path = Path(path_text).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Input TSV not found: {path}")

    required = {MARKER_COLUMN, PAUSENET_ONLY_COLUMN, *BAR_COLUMNS}
    rows: dict[str, dict[str, float]] = {}
    input_order: list[str] = []
    pausenet_values: list[float] = []

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = sorted(required.difference(reader.fieldnames or []))
        if missing:
            raise ValueError(
                f"{path} is missing columns: " + ", ".join(missing)
            )
        for line_number, row in enumerate(reader, start=2):
            marker = row[MARKER_COLUMN].strip()
            if not marker:
                raise ValueError(
                    f"{path}:{line_number}: empty histone marker."
                )
            if marker in rows:
                raise ValueError(
                    f"{path}:{line_number}: duplicate marker {marker}."
                )
            try:
                marker_values = {
                    column: float(row[column]) for column in BAR_COLUMNS
                }
                pausenet_only = float(row[PAUSENET_ONLY_COLUMN])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{path}:{line_number}: nonnumeric adjusted R2."
                ) from exc
            numeric_values = [*marker_values.values(), pausenet_only]
            if not np.all(np.isfinite(numeric_values)):
                raise ValueError(
                    f"{path}:{line_number}: non-finite adjusted R2."
                )
            rows[marker] = marker_values
            input_order.append(marker)
            pausenet_values.append(pausenet_only)

    if not rows:
        raise ValueError(f"No histone-marker rows found in {path}.")
    if not np.allclose(pausenet_values, pausenet_values[0], atol=5e-7):
        raise ValueError(
            f"{PAUSENET_ONLY_COLUMN} is inconsistent between rows."
        )

    if requested_order:
        missing_markers = [
            marker for marker in requested_order if marker not in rows
        ]
        if missing_markers:
            raise ValueError(
                "Requested markers are absent from the table: "
                + ", ".join(missing_markers)
            )
        marker_names = list(requested_order)
    else:
        marker_names = input_order

    values_by_column = {
        column: np.asarray(
            [rows[marker][column] for marker in marker_names],
            dtype=np.float64,
        )
        for column in BAR_COLUMNS
    }
    return marker_names, values_by_column, pausenet_values[0]


def load_all_histones_value(path_text: str) -> float:
    path = Path(path_text).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"All-histone summary not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if ALL_HISTONES_COLUMN not in (reader.fieldnames or []):
            raise ValueError(
                f"{path} is missing column: {ALL_HISTONES_COLUMN}"
            )
        rows = list(reader)
    if len(rows) != 1:
        raise ValueError(
            f"{path} must contain exactly one data row; found {len(rows)}."
        )
    try:
        value = float(rows[0][ALL_HISTONES_COLUMN])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{path}: invalid value in {ALL_HISTONES_COLUMN}."
        ) from exc
    if not np.isfinite(value):
        raise ValueError(f"{path}: non-finite all-histone adjusted R2.")
    return value


def apply_figure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica",
                "DejaVu Sans",
                "sans-serif",
            ],
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 10,
            "axes.linewidth": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def legend_bottom_above_data_line(
    fig: plt.Figure,
    axis: plt.Axes,
    y_value: float,
    gap_cm: float = 0.01,
) -> float:
    """Return the figure y-coordinate placing a legend gap_cm above y_value."""
    fig.canvas.draw()
    display_y = axis.transData.transform((0.0, y_value))[1]
    figure_y = fig.transFigure.inverted().transform((0.0, display_y))[1]
    return figure_y + (gap_cm / 2.54) / fig.get_figheight()


def plot(
    marker_names: Sequence[str],
    values_by_column: dict[str, np.ndarray],
    pausenet_only: float,
    pausenet_all_histones: float,
    output_pdf: Path,
    args: argparse.Namespace,
) -> None:
    apply_figure_style()
    fig, axis = plt.subplots(figsize=(5, 3.2))
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.28, top=0.74)

    x = np.arange(len(marker_names), dtype=float)
    bar_width = 0.24
    offsets = (-bar_width, 0.0, bar_width)
    all_bar_values: list[float] = []

    for column, label, color, offset in zip(
        BAR_COLUMNS,
        BAR_LABELS,
        BAR_COLORS,
        offsets,
        strict=True,
    ):
        heights = values_by_column[column]
        all_bar_values.extend(heights.tolist())
        bars = axis.bar(
            x + offset,
            heights,
            width=bar_width,
            color=color,
            edgecolor="none",
            label=label,
            zorder=2,
        )
        if args.show_values:
            axis.bar_label(
                bars,
                labels=[f"{value:.2f}" for value in heights],
                padding=2,
                fontsize=9,
                rotation=90,
            )

    dash_pattern = (0, (3, 2))
    axis.axhline(
        pausenet_only,
        color=PAUSENET_ONLY_LINE_COLOR,
        linestyle=dash_pattern,
        linewidth=1.2,
        zorder=3,
    )
    axis.axhline(
        pausenet_all_histones,
        color=PAUSENET_ALL_HISTONES_LINE_COLOR,
        linestyle=dash_pattern,
        linewidth=1.3,
        zorder=3,
    )
    axis.axhline(0, color="#222222", linewidth=0.8, zorder=1)

    axis.set_xticks(x)
    axis.set_xticklabels(marker_names, rotation=45, ha="right")
    axis.set_ylabel(r"Test $R_{\mathrm{adj}}^2$")
    axis.set_xlim(-0.65, len(marker_names) - 0.35)

    minimum = min(min(all_bar_values), pausenet_only, pausenet_all_histones)
    maximum = max(max(all_bar_values), pausenet_only, pausenet_all_histones)
    automatic_ymin = min(0.0, minimum - 0.04)
    automatic_ymax = max(0.70, maximum + 0.06)
    ymin = args.ymin if args.ymin is not None else automatic_ymin
    ymax = args.ymax if args.ymax is not None else automatic_ymax
    if ymin >= minimum:
        raise ValueError(
            f"--ymin ({ymin}) must be below the smallest value ({minimum})."
        )
    if ymax <= maximum:
        raise ValueError(
            f"--ymax ({ymax}) must exceed the largest value ({maximum})."
        )
    axis.set_ylim(ymin, ymax)

    if args.title:
        fig.suptitle(args.title, x=0.42, y=0.91, fontsize=14)

    bar_handles = [
        Patch(facecolor=color, edgecolor="none", label=label)
        for color, label in zip(BAR_COLORS, BAR_LABELS, strict=True)
    ]
    legend_bottom_y = legend_bottom_above_data_line(
        fig, axis, pausenet_all_histones
    )
    fig.legend(
        handles=bar_handles,
        loc="lower right",
        bbox_to_anchor=(0.98, legend_bottom_y),
        bbox_transform=fig.transFigure,
        ncol=1,
        handlelength=1.0,
        handletextpad=0.6,
        labelspacing=0.35,
        borderaxespad=0,
        borderpad=0,
        fontsize=10,
    )

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_pdf,
        format="pdf",
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    requested_order = parse_marker_order(args.histone_order)
    marker_names, values_by_column, pausenet_only = load_marker_summary(
        args.input_tsv,
        requested_order,
    )
    pausenet_all_histones = load_all_histones_value(
        args.all_summary_tsv
    )
    output_pdf = Path(args.output_pdf).expanduser().resolve()
    if output_pdf.suffix.lower() != ".pdf":
        output_pdf = output_pdf.with_suffix(".pdf")
    plot(
        marker_names,
        values_by_column,
        pausenet_only,
        pausenet_all_histones,
        output_pdf,
        args,
    )
    print(f"Completed: {output_pdf}")
    print("Histone order: " + ", ".join(marker_names))
    print(f"PauseNet only: {pausenet_only:.6f}")
    print(f"PauseNet + all HeLaS3 histones: {pausenet_all_histones:.6f}")


if __name__ == "__main__":
    main()
