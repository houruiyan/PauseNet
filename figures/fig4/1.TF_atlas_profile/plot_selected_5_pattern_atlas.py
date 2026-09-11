#!/usr/bin/env python3
"""Plot a ProCapNet-style atlas for five selected profile TF-MoDISco patterns."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import logomaker
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd


BASE = Path("/mnt/HDD8TB/houruiyan/pausing_site")
ATLAS_DIR = BASE / "4_plot_figure/fig3/1.TF_atlas_profile"
DATA_DIR = ATLAS_DIR / "data"

DEFAULT_PATTERNS = [
    "pattern_0",
    "pattern_12",
    "pattern_13",
    "pattern_18",
    "pattern_27",
]

MATRIX_FILE = DATA_DIR / "profile_pattern_pwm_cwm_matrices.tsv.gz"
MATRIX_SUMMARY = DATA_DIR / "profile_pattern_pwm_cwm_summary.tsv"
CWM_WEIGHTS = DATA_DIR / "profile_pattern_cwm_weights.tsv"
CURVES = DATA_DIR / "all_pattern_observed_predicted_profiles.tsv"
HIT_SUMMARY = DATA_DIR / "profile_pattern_profile_count_hits.tsv"
PROFILE_HITS = DATA_DIR / "profile_profile_pattern_hits.tsv.gz"
PATTERNS_H5 = (
    BASE
    / "3_model_explanation/TFMoDISco/results/hek293t_netseq/profile/"
    "profile_tfmodisco_patterns.h5"
)
SELECTED_MANIFEST = (
    BASE / "3_model_explanation/DeepSHAP/hek293t_netseq/selected_manifest.tsv"
)
DEFAULT_OUTDIR = ATLAS_DIR

BASES = ["A", "C", "G", "T"]
BASE_COLORS = {"A": "#16A23A", "C": "#2468C5", "G": "#F39C12", "T": "#E53935"}
OBSERVED_COLOR = "#6F3C68"
PREDICTED_COLOR = "#D66C67"
REGION_ORDER = ["TSS", "5SS", "3SS", "TES"]
REGION_COLORS = {
    "TSS": "#4E7FAF",
    "5SS": "#75B7B2",
    "3SS": "#FF7F0E",
    "TES": "#4DA64A",
}
SEQLET_BAR_COLOR = "#F6B262"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot PWM, CWM, CWM weight, profiles and motif-hit summaries."
    )
    parser.add_argument("--matrix-file", type=Path, default=MATRIX_FILE)
    parser.add_argument("--matrix-summary", type=Path, default=MATRIX_SUMMARY)
    parser.add_argument("--cwm-weights", type=Path, default=CWM_WEIGHTS)
    parser.add_argument("--curves", type=Path, default=CURVES)
    parser.add_argument("--hit-summary", type=Path, default=HIT_SUMMARY)
    parser.add_argument("--profile-hits", type=Path, default=PROFILE_HITS)
    parser.add_argument("--patterns-h5", type=Path, default=PATTERNS_H5)
    parser.add_argument("--selected-manifest", type=Path, default=SELECTED_MANIFEST)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=DEFAULT_PATTERNS,
        help="Patterns to include; they are always reordered by descending CWM weight.",
    )
    parser.add_argument(
        "--profile-window",
        type=int,
        default=200,
        help="Show mean profiles from -window to +window bp.",
    )
    return parser.parse_args()


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def load_inputs(args: argparse.Namespace):
    matrices = pd.read_csv(args.matrix_file, sep="\t")
    matrix_summary = pd.read_csv(args.matrix_summary, sep="\t")
    weights = pd.read_csv(args.cwm_weights, sep="\t")
    curves = pd.read_csv(args.curves, sep="\t")
    hits = pd.read_csv(args.hit_summary, sep="\t")

    if "motif_index" in hits.columns and "pattern_index" not in hits.columns:
        hits = hits.rename(columns={"motif_index": "pattern_index"})

    requested = list(dict.fromkeys(args.patterns))
    missing = sorted(set(requested).difference(weights["pattern"]))
    if missing:
        raise ValueError(f"Unknown patterns: {missing}")
    selected = weights[weights["pattern"].isin(requested)].copy()
    if len(selected) != len(requested):
        raise ValueError("Pattern selection contains duplicates or missing values")
    selected = selected.sort_values("cwm_weight", ascending=False).reset_index(drop=True)
    selected = selected.merge(
        hits[["pattern", "profile_hits", "count_hits", "tfmodisco_seqlets"]],
        on="pattern",
        how="left",
    ).merge(
        matrix_summary[
            ["pattern", "display_start_0based", "display_end_exclusive", "display_length"]
        ],
        on="pattern",
        how="left",
    )
    region_summary = load_seqlet_region_summary(
        args.patterns_h5, args.selected_manifest, requested
    )
    selected = selected.merge(region_summary, on="pattern", how="left", validate="one_to_one")
    if not np.array_equal(
        selected["tfmodisco_seqlets"].to_numpy(dtype=int),
        selected["seqlet_count"].to_numpy(dtype=int),
    ):
        mismatch = selected.loc[
            selected["tfmodisco_seqlets"] != selected["seqlet_count"],
            ["pattern", "tfmodisco_seqlets", "seqlet_count"],
        ]
        raise ValueError(f"HDF5/summary seqlet counts disagree:\n{mismatch}")
    return matrices, curves, selected


def load_seqlet_region_summary(
    patterns_h5: Path, selected_manifest: Path, patterns: list[str]
) -> pd.DataFrame:
    """Count each pattern's original TF-MoDISco seqlets by genomic region type."""
    manifest = pd.read_csv(selected_manifest, sep="\t")
    if "sample_id" not in manifest.columns:
        raise ValueError(f"sample_id is absent from {selected_manifest}")
    if "contribution_array_index" in manifest.columns:
        expected = np.arange(len(manifest), dtype=int)
        observed = manifest["contribution_array_index"].to_numpy(dtype=int)
        if not np.array_equal(observed, expected):
            raise ValueError("selected_manifest.tsv is not in contribution-array order")

    rows = []
    with h5py.File(patterns_h5, "r") as handle:
        group = handle["pos_patterns"]
        for pattern in patterns:
            if pattern not in group:
                raise KeyError(f"{pattern} is absent from pos_patterns in {patterns_h5}")
            example_idx = np.asarray(
                group[pattern]["seqlets"]["example_idx"], dtype=np.int64
            )
            if np.any(example_idx < 0) or np.any(example_idx >= len(manifest)):
                raise IndexError(f"{pattern}: example_idx outside selected_manifest.tsv")
            sample_ids = manifest.iloc[example_idx]["sample_id"].astype(str)
            labels = sample_ids.str.extract(r"\|(TSS|5SS|3SS|TES)\|", expand=False)
            if labels.isna().any():
                examples = sample_ids[labels.isna()].head(3).tolist()
                raise ValueError(f"{pattern}: cannot parse region type from {examples}")
            counts = labels.value_counts()
            row = {"pattern": pattern, "seqlet_count": int(len(example_idx))}
            for region in REGION_ORDER:
                row[f"{region}_seqlets"] = int(counts.get(region, 0))
                row[f"{region}_fraction"] = float(counts.get(region, 0) / len(example_idx))
            rows.append(row)
    return pd.DataFrame(rows)


def matrix_for_pattern(
    matrices: pd.DataFrame, pattern: str, matrix_name: str, start: int, end: int
) -> pd.DataFrame:
    subset = matrices[
        (matrices["pattern"] == pattern) & (matrices["matrix"] == matrix_name)
    ].sort_values("position")
    subset = subset[(subset["position"] >= start) & (subset["position"] < end)]
    frame = subset.set_index("position")[BASES]
    if len(frame) != end - start:
        raise ValueError(f"Incomplete {matrix_name} matrix for {pattern}")
    return frame


def draw_logo(ax: plt.Axes, matrix: pd.DataFrame) -> None:
    logomaker.Logo(
        matrix,
        ax=ax,
        color_scheme=BASE_COLORS,
        fade_below=0,
        shade_below=0,
        width=0.92,
    )
    ax.axhline(0, color="#666666", linewidth=0.35)
    ax.set_xlim(matrix.index.min() - 0.6, matrix.index.max() + 0.6)
    ax.set_axis_off()


def normalized_curve(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros_like(values)
    baseline = np.nanmin(values)
    shifted = np.maximum(values - baseline, 0)
    maximum = np.nanmax(shifted)
    return shifted / maximum if maximum > 0 else np.zeros_like(values)


def draw_profile(ax: plt.Axes, frame: pd.DataFrame, column: str, color: str) -> None:
    x = frame["position_bp"].to_numpy(dtype=float)
    y = normalized_curve(frame[column].to_numpy(dtype=float))
    ax.plot(x, y, color=color, linewidth=0.8)
    ax.axvline(0, color="#777777", linewidth=0.45, linestyle="--")
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylim(-0.03, 1.08)
    ax.set_axis_off()


def draw_weight(
    ax: plt.Axes,
    value: float,
    maximum: float,
    cmap: mpl.colors.Colormap,
    norm: Normalize,
) -> None:
    size = 560.0 * value / maximum
    ax.scatter([0.5], [0.5], s=size, c=[cmap(norm(value))], edgecolor="#17204E", linewidth=0.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()


def draw_hit_cell(
    ax: plt.Axes,
    value: int,
    cmap: mpl.colors.Colormap,
    norm: Normalize,
) -> None:
    color_value = np.sqrt(value)
    color = cmap(norm(color_value))
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="white", linewidth=0.8))
    text_color = "white" if norm(color_value) > 0.53 else "#111111"
    ax.text(0.5, 0.5, f"{int(value):,}", ha="center", va="center", fontsize=7.5, color=text_color)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()


def draw_seqlet_count(ax: plt.Axes, value: int, maximum: int) -> None:
    ax.set_facecolor("#FAF4F1")
    ax.barh([0], [value], color=SEQLET_BAR_COLOR, height=0.58)
    offset = 0.025 * maximum
    ax.text(value + offset, 0, f"{value:,}", ha="left", va="center", fontsize=8.5)
    ax.set_xlim(0, maximum * 1.22)
    ax.set_ylim(-0.55, 0.55)
    ax.set_axis_off()


def draw_region_fraction(ax: plt.Axes, row: pd.Series) -> None:
    left = 0.0
    for region in REGION_ORDER:
        value = float(row[f"{region}_fraction"])
        ax.barh(
            [0],
            [value],
            left=[left],
            height=0.58,
            color=REGION_COLORS[region],
            edgecolor="white",
            linewidth=0.55,
        )
        left += value
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.55, 0.55)
    ax.set_axis_off()


def plot_atlas(
    matrices: pd.DataFrame,
    curves: pd.DataFrame,
    selected: pd.DataFrame,
    profile_window: int,
    output_pdf: Path,
) -> pd.DataFrame:
    n = len(selected)
    width_ratios = [0.75, 1.75, 1.75, 0.62, 1.72, 1.72, 1.05, 1.65]
    fig = plt.figure(figsize=(11.8, 4.9))
    gs = fig.add_gridspec(
        nrows=n,
        ncols=len(width_ratios),
        width_ratios=width_ratios,
        hspace=0.16,
        wspace=0.17,
        left=0.035,
        right=0.992,
        top=0.84,
        bottom=0.16,
    )

    cwm_cmap = LinearSegmentedColormap.from_list(
        "cwm_weight", ["#F5D7C6", "#D66C67", "#6F3C68"]
    )
    weight_norm = Normalize(
        vmin=float(selected["cwm_weight"].min()),
        vmax=float(selected["cwm_weight"].max()),
    )
    hit_cmap = LinearSegmentedColormap.from_list(
        "hit_counts", ["#F7E0D2", "#D66C67", "#6F3C68"]
    )
    max_sqrt_hits = float(
        np.sqrt(selected[["profile_hits", "count_hits"]].to_numpy(dtype=float)).max()
    )
    hit_norm = Normalize(vmin=0, vmax=max_sqrt_hits)
    max_weight = float(selected["cwm_weight"].max())
    max_seqlets = int(selected["seqlet_count"].max())

    headers = [
        "",
        "PWM",
        "CWM",
        "CWM\nWeight",
        "Avg. Measured\nProfile",
        "Avg. Predicted\nProfile",
        "# seqlets",
        "% seqlets\n(region type)",
    ]
    summary_rows = []
    for row_i, row in selected.iterrows():
        axes = [fig.add_subplot(gs[row_i, col_i]) for col_i in range(len(width_ratios))]
        if row_i == 0:
            for ax, title in zip(axes, headers):
                ax.set_title(title, pad=7, fontsize=11)

        pattern = str(row["pattern"])
        pattern_index = int(row["pattern_index"])
        start = int(row["display_start_0based"])
        end = int(row["display_end_exclusive"])

        axes[0].text(0.98, 0.5, pattern, ha="right", va="center", fontsize=9.5)
        axes[0].set_axis_off()
        draw_logo(axes[1], matrix_for_pattern(matrices, pattern, "PWM_bits_logo", start, end))
        draw_logo(axes[2], matrix_for_pattern(matrices, pattern, "CWM", start, end))
        draw_weight(axes[3], float(row["cwm_weight"]), max_weight, cwm_cmap, weight_norm)

        curve = curves[curves["pattern_index"] == pattern_index].copy()
        curve = curve[curve["position_bp"].between(-profile_window, profile_window)]
        draw_profile(axes[4], curve, "observed_mean", OBSERVED_COLOR)
        draw_profile(axes[5], curve, "predicted_mean", PREDICTED_COLOR)

        draw_seqlet_count(axes[6], int(row["seqlet_count"]), max_seqlets)
        draw_region_fraction(axes[7], row)

        summary_rows.append(
            {
                "display_rank": row_i + 1,
                "pattern": pattern,
                "pattern_index": pattern_index,
                "cwm_weight": float(row["cwm_weight"]),
                "tfmodisco_seqlets": int(row["tfmodisco_seqlets"]),
                "profile_hits": int(row["profile_hits"]),
                "count_hits": int(row["count_hits"]),
                **{
                    f"{region}_seqlets": int(row[f"{region}_seqlets"])
                    for region in REGION_ORDER
                },
                **{
                    f"{region}_percent": 100.0 * float(row[f"{region}_fraction"])
                    for region in REGION_ORDER
                },
            }
        )

    fig.text(0.012, 0.965, "A", fontsize=21, fontweight="bold", ha="left", va="top")
    legend_handles = [
        mpl.patches.Patch(facecolor=REGION_COLORS[region], label=region)
        for region in REGION_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower right",
        bbox_to_anchor=(0.988, 0.018),
        ncol=4,
        frameon=False,
        handlelength=1.0,
        handletextpad=0.35,
        columnspacing=0.85,
    )
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(summary_rows)


def main() -> int:
    args = parse_args()
    configure_style()
    args.outdir.mkdir(parents=True, exist_ok=True)
    matrices, curves, selected = load_inputs(args)

    output_pdf = args.outdir / "selected_5_profile_pattern_atlas.pdf"
    output_tsv = args.outdir / "selected_5_profile_pattern_atlas_summary.tsv"
    summary = plot_atlas(
        matrices,
        curves,
        selected,
        args.profile_window,
        output_pdf,
    )
    summary.to_csv(output_tsv, sep="\t", index=False)
    print(summary.to_string(index=False))
    print(f"\nPDF: {output_pdf}")
    print(f"Summary: {output_tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
