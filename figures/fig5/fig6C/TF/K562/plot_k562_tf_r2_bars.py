#!/usr/bin/env python3
"""Plot grouped TF R2 bars with PauseNet reference lines.

The input is the tab-separated summary produced by
5.run_k562_k562_vs_helas3_tf_r2.sh. Three values are plotted for every TF:

    1. K562 TF only
    2. PauseNet + K562 TF
    3. PauseNet + HeLaS3 TF

The HeLaS3-TF-only column is intentionally not plotted.

Example
-------
python 6.plot_k562_tf_r2_bars.py \
  --input-tsv k562_tf_r2_results/k562_tf_r2_summary.tsv \
  --output-pdf k562_tf_r2_results/k562_tf_r2_grouped_bars.pdf
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


TF_ONLY_COLOR = "#f9c00c"
PAUSENET_K562_TF_COLOR = "#00b9f1"
PAUSENET_HELAS3_TF_COLOR = "#f9320c"
PAUSENET_ONLY_LINE_COLOR = "#777777"
PAUSENET_ALL_TF_LINE_COLOR = "#d62728"

DEFAULT_PAUSENET_ONLY = 0.286494
DEFAULT_PAUSENET_ALL_TF = 0.625955
DEFAULT_TF_ORDER = ("CTCF", "CHD2", "SWI", "TFIIF")

BAR_COLUMNS = (
    "K562_TF_only_R2_adj",
    "PauseNet_plus_K562_TF_R2_adj",
    "PauseNet_plus_HeLaS3_TF_R2_adj",
)
BAR_LABELS = (
    "TF only",
    "PauseNet + K562",
    "PauseNet + HeLaS3",
)
BAR_COLORS = (
    TF_ONLY_COLOR,
    PAUSENET_K562_TF_COLOR,
    PAUSENET_HELAS3_TF_COLOR,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-tsv",
        required=True,
        help="k562_tf_r2_summary.tsv produced by the batch script.",
    )
    parser.add_argument(
        "--output-pdf",
        required=True,
        help="Output PDF path.",
    )
    parser.add_argument(
        "--pausenet-only",
        type=float,
        default=DEFAULT_PAUSENET_ONLY,
        help="Gray dashed reference line (default: 0.286494).",
    )
    parser.add_argument(
        "--pausenet-all-tf",
        type=float,
        default=DEFAULT_PAUSENET_ALL_TF,
        help="Red dashed reference line (default: 0.62599).",
    )
    parser.add_argument(
        "--tf-order",
        action="append",
        help=(
            "Optional TF order/subset. Repeat or provide comma-separated "
            "names. Default: CTCF, CHD2, SWI, TFIIF."
        ),
    )
    parser.add_argument(
        "--title",
        default="K562",
        help="Figure title (default: K562); use an empty string to hide it.",
    )
    parser.add_argument(
        "--show-values",
        action="store_true",
        help="Show numeric values above bars.",
    )
    parser.add_argument(
        "--ymax",
        type=float,
        default=None,
        help="Optional upper y limit; determined automatically by default.",
    )
    return parser.parse_args()


def parse_tf_order(values: Sequence[str] | None) -> list[str]:
    if values is None:
        return list(DEFAULT_TF_ORDER)
    result: list[str] = []
    for value in values:
        result.extend(
            item.strip() for item in value.split(",") if item.strip()
        )
    if len(result) != len(set(result)):
        raise ValueError("--tf-order contains duplicate TF names.")
    return result


def load_table(
    path_text: str,
    requested_order: Sequence[str],
) -> tuple[list[str], dict[str, np.ndarray]]:
    path = Path(path_text).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Input TSV not found: {path}")

    required = {"TF", *BAR_COLUMNS}
    rows: dict[str, dict[str, float]] = {}
    input_order: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = sorted(required.difference(reader.fieldnames or []))
        if missing:
            raise ValueError(
                f"{path} is missing columns: " + ", ".join(missing)
            )
        for line_number, row in enumerate(reader, start=2):
            tf = row["TF"].strip()
            if not tf:
                raise ValueError(f"{path}:{line_number}: empty TF name.")
            if tf in rows:
                raise ValueError(f"{path}:{line_number}: duplicate TF {tf}.")
            try:
                values = {
                    column: float(row[column]) for column in BAR_COLUMNS
                }
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_number}: nonnumeric R2 value."
                ) from exc
            if not all(np.isfinite(list(values.values()))):
                raise ValueError(
                    f"{path}:{line_number}: non-finite R2 value."
                )
            rows[tf] = values
            input_order.append(tf)

    if not rows:
        raise ValueError(f"No TF rows found in {path}.")
    if requested_order:
        missing_tfs = [tf for tf in requested_order if tf not in rows]
        if missing_tfs:
            raise ValueError(
                "Requested TFs are absent from the table: "
                + ", ".join(missing_tfs)
            )
        tf_names = list(requested_order)
    else:
        tf_names = input_order

    values_by_column = {
        column: np.asarray(
            [rows[tf][column] for tf in tf_names],
            dtype=np.float64,
        )
        for column in BAR_COLUMNS
    }
    return tf_names, values_by_column


def apply_publication_style() -> None:
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
            "svg.fonttype": "none",
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
    tf_names: Sequence[str],
    values_by_column: dict[str, np.ndarray],
    output_pdf: Path,
    args: argparse.Namespace,
) -> None:
    apply_publication_style()
    fig, axis = plt.subplots(figsize=(5, 3.2))
    fig.subplots_adjust(
        left=0.15,
        right=0.98,
        bottom=0.28,
        top=0.74,
    )

    x = np.arange(len(tf_names), dtype=float)
    bar_width = 0.23
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
        args.pausenet_only,
        color=PAUSENET_ONLY_LINE_COLOR,
        linestyle=dash_pattern,
        linewidth=1.2,
        zorder=3,
    )
    axis.axhline(
        args.pausenet_all_tf,
        color=PAUSENET_ALL_TF_LINE_COLOR,
        linestyle=dash_pattern,
        linewidth=1.3,
        zorder=3,
    )

    axis.set_xticks(x)
    axis.set_xticklabels(tf_names, rotation=45, ha="right")
    axis.set_ylabel(r"Test $R_{\mathrm{adj}}^2$")
    axis.set_xlim(-0.65, len(tf_names) - 0.35)
    if args.title:
        fig.suptitle(args.title, x=0.42, y=0.91, fontsize=14)

    maximum = max(
        max(all_bar_values),
        args.pausenet_only,
        args.pausenet_all_tf,
    )
    automatic_ymax = max(0.70, maximum + 0.08)
    ymax = args.ymax if args.ymax is not None else automatic_ymax
    if ymax <= maximum:
        raise ValueError(
            f"--ymax ({ymax}) must exceed the largest plotted value "
            f"({maximum})."
        )
    axis.set_ylim(0, ymax)

    bar_legend_handles = [
        Patch(facecolor=color, edgecolor="none", label=label)
        for color, label in zip(BAR_COLORS, BAR_LABELS, strict=True)
    ]
    legend_bottom_y = legend_bottom_above_data_line(
        fig, axis, args.pausenet_all_tf
    )
    fig.legend(
        handles=bar_legend_handles,
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
    if args.pausenet_only < 0 or args.pausenet_all_tf < 0:
        raise ValueError("Reference R2 values cannot be negative.")
    requested_order = parse_tf_order(args.tf_order)
    tf_names, values_by_column = load_table(
        args.input_tsv, requested_order
    )
    output_pdf = Path(args.output_pdf).expanduser().resolve()
    if output_pdf.suffix.lower() != ".pdf":
        output_pdf = output_pdf.with_suffix(".pdf")
    plot(tf_names, values_by_column, output_pdf, args)
    print(f"Completed: {output_pdf}")
    print("TF order: " + ", ".join(tf_names))


if __name__ == "__main__":
    main()
