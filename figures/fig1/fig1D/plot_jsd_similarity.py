#!/usr/bin/env python3
"""Plot profile similarity by HEK293T anchored-window type."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ANCHOR_ORDER = ("TSS", "5SS", "3SS", "TES")
DISPLAY_LABELS = {
    "TSS": "TSS",
    "5SS": "5' splice site",
    "3SS": "3' splice site",
    "TES": "TES",
}
COLORS = {
    "TSS": "#5E5E5E",
    "5SS": "#D95F66",
    "3SS": "#DDAA3B",
    "TES": "#5D9FB9",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--per-window-jsd", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--resolution-bp", type=int, choices=(1, 5, 10, 20), default=1)
    parser.add_argument(
        "--min-count-exclusive",
        type=float,
        default=None,
        help="Keep only windows whose observed count is strictly greater than this value.",
    )
    parser.add_argument(
        "--global-minmax",
        action="store_true",
        help="Rescale similarity using one min-max range shared by all retained windows.",
    )
    parser.add_argument(
        "--figure-stem",
        default="PauseNet_HEK293T_1bp_profile_similarity_by_window_type",
    )
    parser.add_argument(
        "--x-label",
        default=None,
        help="Optional replacement for the x-axis label.",
    )
    parser.add_argument(
        "--hide-count-annotation",
        action="store_true",
        help="Do not print the count-filter threshold inside the panel.",
    )
    return parser.parse_args()


def load_similarity(
    manifest_path: Path,
    jsd_path: Path,
    resolution_bp: int,
    min_count_exclusive: float | None,
    global_minmax: bool,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    manifest = pd.read_csv(manifest_path, sep="\t").set_index(
        "array_index", verify_integrity=True
    )
    values = np.load(jsd_path)
    valid_index = values["valid_array_index"]
    jsd_key = f"pausenet_jsd_{resolution_bp}bp"
    if jsd_key not in values.files:
        raise ValueError(f"The per-window archive has no {jsd_key} array.")
    jsd = values[jsd_key]
    valid_count = values["valid_count"]

    if len(valid_index) != len(jsd) or len(jsd) != len(valid_count):
        raise ValueError("Per-window arrays have inconsistent lengths.")
    if not np.isfinite(jsd).all():
        raise ValueError("The 1 bp JSD array contains non-finite values.")

    frame = manifest.loc[valid_index, ["anchor_type", "total_count"]].copy()
    if not np.array_equal(frame["total_count"].to_numpy(), valid_count):
        raise ValueError("The JSD array does not align with the test manifest counts.")
    frame["array_index"] = frame.index
    frame["raw_similarity_1_minus_jsd"] = 1.0 - jsd
    frame = frame.loc[frame["anchor_type"].isin(ANCHOR_ORDER)].copy()
    if min_count_exclusive is not None:
        frame = frame.loc[frame["total_count"] > min_count_exclusive].copy()
    if frame.empty:
        raise ValueError("No windows remain after count filtering.")

    raw_min = float(frame["raw_similarity_1_minus_jsd"].min())
    raw_max = float(frame["raw_similarity_1_minus_jsd"].max())
    if global_minmax:
        if raw_max <= raw_min:
            raise ValueError("Global min-max normalization requires non-identical values.")
        frame["similarity_1_minus_jsd"] = (
            frame["raw_similarity_1_minus_jsd"] - raw_min
        ) / (raw_max - raw_min)
        normalization = "global_minmax"
    else:
        frame["similarity_1_minus_jsd"] = frame["raw_similarity_1_minus_jsd"]
        normalization = "none"

    frame["anchor_type"] = pd.Categorical(
        frame["anchor_type"], categories=ANCHOR_ORDER, ordered=True
    )
    metadata: dict[str, float | int | str] = {
        "count_filter": (
            "none"
            if min_count_exclusive is None
            else f"total_count > {min_count_exclusive:g}"
        ),
        "normalization": normalization,
        "resolution_bp": resolution_bp,
        "global_raw_similarity_min": raw_min,
        "global_raw_similarity_max": raw_max,
        "n_retained": len(frame),
    }
    return frame.sort_values("anchor_type"), metadata


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby("anchor_type", observed=True)["similarity_1_minus_jsd"]
    summary = grouped.agg(n="size", mean="mean", median="median", min="min", max="max")
    quantiles = grouped.quantile([0.025, 0.25, 0.75, 0.975]).unstack()
    quantiles.columns = ["q025", "q25", "q75", "q975"]
    summary = summary.join(quantiles).reset_index()
    summary["mean_raw_similarity_1_minus_jsd"] = (
        frame.groupby("anchor_type", observed=True)["raw_similarity_1_minus_jsd"]
        .mean()
        .to_numpy()
    )
    summary["window_type"] = summary["anchor_type"].map(DISPLAY_LABELS)
    return summary[
        [
            "anchor_type",
            "window_type",
            "n",
            "mean",
            "median",
            "q025",
            "q25",
            "q75",
            "q975",
            "min",
            "max",
            "mean_raw_similarity_1_minus_jsd",
        ]
    ]


def plot(
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    output_prefix: Path,
    resolution_bp: int,
    min_count_exclusive: float | None,
    global_minmax: bool,
    x_label_override: str | None,
    hide_count_annotation: bool,
) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.linewidth": 0.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig, ax = plt.subplots(figsize=(3.35, 2.72))
    rng = np.random.default_rng(20260714)
    y_positions = dict(zip(ANCHOR_ORDER, range(len(ANCHOR_ORDER) - 1, -1, -1)))

    for anchor_type in ANCHOR_ORDER:
        values = frame.loc[
            frame["anchor_type"] == anchor_type, "similarity_1_minus_jsd"
        ].to_numpy()
        y = y_positions[anchor_type]
        jitter = rng.uniform(-0.18, 0.18, size=len(values))
        ax.scatter(
            values,
            np.full(len(values), y) + jitter,
            s=2.4,
            color=COLORS[anchor_type],
            alpha=0.12,
            linewidths=0,
            zorder=1,
        )

        stats = summary.loc[summary["anchor_type"] == anchor_type].iloc[0]
        ax.hlines(
            y,
            stats["q25"],
            stats["q75"],
            color=COLORS[anchor_type],
            linewidth=2.6,
            zorder=3,
        )
        ax.scatter(
            stats["mean"],
            y,
            s=18,
            facecolor="white",
            edgecolor=COLORS[anchor_type],
            linewidth=0.85,
            zorder=4,
        )

    labels = [
        f"{DISPLAY_LABELS[anchor_type]}\n(n={int(summary.loc[summary['anchor_type'] == anchor_type, 'n'].iloc[0]):,})"
        for anchor_type in ANCHOR_ORDER
    ]
    ax.set_yticks([y_positions[anchor_type] for anchor_type in ANCHOR_ORDER], labels)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.45, len(ANCHOR_ORDER) - 0.55)
    ax.set_xticks((0.0, 0.25, 0.5, 0.75, 1.0))
    x_label = f"Profile similarity (1 - JSD), {resolution_bp} bp"
    if global_minmax:
        x_label = f"Normalized 1 - JSD ({resolution_bp} bp)"
    if x_label_override is not None:
        x_label = x_label_override
    ax.set_xlabel(x_label)
    if min_count_exclusive is not None and not hide_count_annotation:
        ax.text(
            0.99,
            0.98,
            f"count > {min_count_exclusive:g}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.5,
        )
    ax.tick_params(axis="x", width=0.8, length=3)
    ax.tick_params(axis="y", length=0, pad=3)
    fig.subplots_adjust(left=0.30, right=0.98, bottom=0.20, top=0.98)

    fig.savefig(f"{output_prefix}.svg", bbox_inches="tight")
    fig.savefig(f"{output_prefix}.pdf", bbox_inches="tight")
    fig.savefig(f"{output_prefix}.png", dpi=600, bbox_inches="tight")
    fig.savefig(
        f"{output_prefix}.tiff",
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    frame, metadata = load_similarity(
        args.manifest,
        args.per_window_jsd,
        args.resolution_bp,
        args.min_count_exclusive,
        args.global_minmax,
    )
    summary = summarize(frame)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / args.figure_stem
    frame.to_csv(f"{prefix}_source_data.tsv", sep="\t", index=False)
    summary.to_csv(f"{prefix}_summary.tsv", sep="\t", index=False)
    pd.DataFrame([metadata]).to_csv(f"{prefix}_metadata.tsv", sep="\t", index=False)
    plot(
        frame,
        summary,
        prefix,
        args.resolution_bp,
        args.min_count_exclusive,
        args.global_minmax,
        args.x_label,
        args.hide_count_annotation,
    )
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"wrote {prefix}.svg")
    print(f"wrote {prefix}.pdf")
    print(f"wrote {prefix}.png")
    print(f"wrote {prefix}.tiff")


if __name__ == "__main__":
    main()
