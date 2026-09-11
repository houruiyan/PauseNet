#!/usr/bin/env python3
"""Replot the fixed PATZ1 and ZNF610 saturation-mutagenesis examples.

The two biological examples are deliberately frozen; this script changes only
their presentation.  It reads the previously computed single-nucleotide ISM
matrices and creates one 10 x 4 inch PDF per example with the layout:

    coloured sequence + selected motif-loss mutation
    gain/loss summary curves
    A/C/G/T signed ISM heatmap + genomic coordinates

The heatmap value is always

    predicted log1p(counts, mutant) - predicted log1p(counts, reference).

Positive values are gains.  Negative values are losses; the blue loss curve
shows their positive magnitude.  The sequence and x axis follow PauseNet's
stored transcriptional 5' -> 3' orientation, so minus-strand genomic
coordinates decrease from left to right.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd


ROOT = Path("/mnt/HDD8TB/houruiyan/pausing_site")
HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = HERE / "results"
DEFAULT_TEST_DIR = ROOT / "data/HEK293T_NETseq/dataset/test"

BASES = np.asarray(list("ACGT"))
BASE_COLORS = {
    "A": "#24A148",
    "C": "#2767B0",
    "G": "#F39C12",
    "T": "#E53935",
}
GAIN_COLOR = "#CC3D3D"
LOSS_COLOR = "#2F70B7"
MOTIF_COLOR = "#F04D8A"


@dataclass(frozen=True)
class FixedExample:
    tf_name: str
    pattern: str
    dataset_idx: int
    seqlet_index: int
    seqlet_center_crop: int
    core_start_crop: int
    core_end_crop: int
    selected_position_crop: int
    selected_reference_base: str
    selected_alternative_base: str
    is_revcomp: bool


# Exact instances selected by the original candidate-ranking analysis.
EXAMPLES = (
    FixedExample(
        tf_name="PATZ1",
        pattern="pos_pattern_2",
        dataset_idx=14180,
        seqlet_index=25,
        seqlet_center_crop=426,
        core_start_crop=423,
        core_end_crop=434,
        selected_position_crop=426,
        selected_reference_base="G",
        selected_alternative_base="A",
        is_revcomp=True,
    ),
    FixedExample(
        tf_name="ZNF610",
        pattern="pos_pattern_7",
        dataset_idx=12977,
        seqlet_index=37,
        seqlet_center_crop=191,
        core_start_crop=189,
        core_end_crop=199,
        selected_position_crop=191,
        selected_reference_base="G",
        selected_alternative_base="A",
        is_revcomp=False,
    ),
)


@dataclass
class PlotData:
    example: FixedExample
    manifest_row: pd.Series
    positions_crop: np.ndarray
    relative: np.ndarray
    reference_codes: np.ndarray
    matrix: np.ndarray
    reference_prediction: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_RESULTS,
        help="Directory containing the two existing saturation_matrix.npz files.",
    )
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--smooth-sigma", type=float, default=1.1)
    parser.add_argument(
        "--write-png",
        action="store_true",
        help="Also write PNG previews; PDF is the default/primary output.",
    )
    return parser.parse_args()


def configure_style() -> None:
    """Use final-figure-small typography on the requested 2x-wide canvas."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "axes.linewidth": 1.0,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.smooth_sigma < 0:
        raise ValueError("--smooth-sigma must be nonnegative")
    required = [args.test_dir / "manifest.tsv"]
    required.extend(
        args.source_dir / f"{example.tf_name}_saturation_matrix.npz"
        for example in EXAMPLES
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing input file(s): " + ", ".join(missing))


def load_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    required = {
        "array_index",
        "sample_id",
        "chrom",
        "output_start",
        "output_end",
        "strand",
        "region_type",
        "count",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    frame = frame.set_index("array_index", drop=False)
    if not frame.index.is_unique:
        raise ValueError("manifest array_index is not unique")
    return frame


def load_plot_data(
    example: FixedExample,
    source_dir: Path,
    manifest: pd.DataFrame,
) -> PlotData:
    if example.dataset_idx not in manifest.index:
        raise KeyError(f"dataset_idx {example.dataset_idx} is absent from manifest")

    path = source_dir / f"{example.tf_name}_saturation_matrix.npz"
    with np.load(path) as archive:
        positions_crop = np.asarray(
            archive["positions_crop_0based"], dtype=np.int64
        )
        relative = np.asarray(
            archive["positions_relative_to_seqlet_center"], dtype=np.int64
        )
        reference_codes = np.asarray(archive["reference_codes"], dtype=np.int64)
        matrix = np.asarray(
            archive["delta_predicted_log1p_count_mutant_minus_reference"],
            dtype=np.float64,
        )
        reference_prediction = float(
            np.asarray(archive["reference_predicted_log1p_count"]).reshape(-1)[0]
        )

    expected_relative = positions_crop - example.seqlet_center_crop
    if not np.array_equal(relative, expected_relative):
        raise ValueError(f"{example.tf_name}: relative coordinates do not match")
    if matrix.shape != (4, len(positions_crop)):
        raise ValueError(
            f"{example.tf_name}: expected a 4xL ISM matrix, got {matrix.shape}"
        )
    if reference_codes.shape != (len(positions_crop),):
        raise ValueError(f"{example.tf_name}: reference sequence length mismatch")
    if np.any((reference_codes < 0) | (reference_codes > 3)):
        raise ValueError(f"{example.tf_name}: non-ACGT code in display sequence")
    selected_columns = np.where(positions_crop == example.selected_position_crop)[0]
    if len(selected_columns) != 1:
        raise ValueError(f"{example.tf_name}: selected SNV is outside matrix window")
    observed_reference = str(BASES[reference_codes[selected_columns[0]]])
    if observed_reference != example.selected_reference_base:
        raise ValueError(
            f"{example.tf_name}: expected {example.selected_reference_base} at "
            f"selected site, found {observed_reference}"
        )

    return PlotData(
        example=example,
        manifest_row=manifest.loc[example.dataset_idx],
        positions_crop=positions_crop,
        relative=relative,
        reference_codes=reference_codes,
        matrix=matrix,
        reference_prediction=reference_prediction,
    )


def smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if sigma <= 0:
        return values.copy()
    radius = int(math.ceil(4 * sigma))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(offsets**2) / (2.0 * sigma**2))
    kernel /= kernel.sum()
    return np.convolve(values, kernel, mode="same")


def gain_loss_curves(matrix: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    gain = np.maximum(0.0, np.nanmax(matrix, axis=0))
    loss = np.maximum(0.0, -np.nanmin(matrix, axis=0))
    return smooth(gain, sigma), smooth(loss, sigma)


def genomic_position_1based(manifest_row: pd.Series, position_crop: int) -> int:
    if str(manifest_row.strand) == "+":
        return int(manifest_row.output_start) + int(position_crop) + 1
    return int(manifest_row.output_end) - int(position_crop)


def chromosome_label(chrom: str) -> str:
    text = str(chrom)
    return text[3:] if text.lower().startswith("chr") else text


def robust_shared_limit(plot_data: list[PlotData]) -> float:
    values = [
        np.abs(item.matrix[np.isfinite(item.matrix)])
        for item in plot_data
        if np.isfinite(item.matrix).any()
    ]
    if not values:
        return 0.1
    return max(0.05, float(np.percentile(np.concatenate(values), 99.0)))


def shared_curve_limit(plot_data: list[PlotData], sigma: float) -> float:
    maxima: list[float] = []
    for item in plot_data:
        gain, loss = gain_loss_curves(item.matrix, sigma)
        maxima.extend([float(np.nanmax(gain)), float(np.nanmax(loss))])
    return max(0.05, 1.08 * max(maxima))


def draw_example(
    data: PlotData,
    smooth_sigma: float,
    heatmap_limit: float,
    curve_limit: float,
    output_pdf: Path,
    output_png: Path | None,
) -> None:
    """Draw one 10 x 4 inch panel matching the supplied reference layout."""
    example = data.example
    relative = data.relative
    reference_sequence = "".join(BASES[data.reference_codes].tolist())
    gain, loss = gain_loss_curves(data.matrix, smooth_sigma)
    norm = TwoSlopeNorm(
        vmin=-heatmap_limit,
        vcenter=0.0,
        vmax=heatmap_limit,
    )

    # final-figure-small is 5 x 4 inches: double width, preserve height.
    fig = plt.figure(figsize=(10, 4), constrained_layout=False)
    ax_sequence = fig.add_axes([0.095, 0.745, 0.780, 0.225])
    ax_track = fig.add_axes([0.095, 0.405, 0.780, 0.310], sharex=ax_sequence)
    ax_heatmap = fig.add_axes([0.095, 0.145, 0.780, 0.215], sharex=ax_sequence)
    ax_colorbar = fig.add_axes([0.895, 0.145, 0.017, 0.215])

    core_left = example.core_start_crop - example.seqlet_center_crop - 0.5
    core_width = example.core_end_crop - example.core_start_crop
    selected_x = example.selected_position_crop - example.seqlet_center_crop
    x_left = float(relative[0]) - 0.5
    x_right = float(relative[-1]) + 0.5

    # Top: nucleotide-coloured reference sequence and motif-loss mutation.
    ax_sequence.set_xlim(x_left, x_right)
    ax_sequence.set_ylim(0.0, 1.0)
    ax_sequence.set_yticks([])
    ax_sequence.tick_params(
        axis="x", which="both", bottom=False, top=False, labelbottom=False
    )
    for spine in ax_sequence.spines.values():
        spine.set_visible(False)
    for x, base, position_crop in zip(
        relative, reference_sequence, data.positions_crop, strict=True
    ):
        in_core = example.core_start_crop <= position_crop < example.core_end_crop
        ax_sequence.text(
            x,
            0.27,
            base,
            color=BASE_COLORS[base],
            ha="center",
            va="center",
            fontfamily="DejaVu Sans Mono",
            fontsize=7.2 if in_core else 6.0,
            fontweight="bold" if in_core else "normal",
        )
    ax_sequence.add_patch(
        Rectangle(
            (core_left, 0.07),
            core_width,
            0.43,
            fill=False,
            edgecolor=MOTIF_COLOR,
            linewidth=1.5,
        )
    )
    ax_sequence.axvline(
        selected_x,
        ymin=0.0,
        ymax=0.52,
        color=MOTIF_COLOR,
        linewidth=1.0,
        linestyle=":",
    )
    mutation_label = (
        f"{example.selected_reference_base}>{example.selected_alternative_base}: "
        f"{example.tf_name} loss"
    )
    ax_sequence.annotate(
        mutation_label,
        xy=(selected_x, 0.51),
        xytext=(selected_x, 0.88),
        ha="center",
        va="center",
        fontsize=11,
        bbox={
            "boxstyle": "round,pad=0.10",
            "facecolor": "white",
            "edgecolor": "#444444",
            "linewidth": 0.8,
        },
        arrowprops={
            "arrowstyle": "-",
            "color": "#444444",
            "linewidth": 0.8,
        },
    )

    # Middle: strongest gain and strongest loss at every base position.
    ax_track.plot(relative, gain, color=GAIN_COLOR, linewidth=1.8, label="gain")
    ax_track.plot(relative, loss, color=LOSS_COLOR, linewidth=1.8, label="loss")
    ax_track.axhline(0.0, color="#B8B8B8", linewidth=0.8, linestyle="--")
    ax_track.axvline(
        selected_x,
        color=MOTIF_COLOR,
        linewidth=1.0,
        linestyle=":",
    )
    ax_track.set_xlim(x_left, x_right)
    ax_track.set_ylim(0.0, curve_limit)
    ax_track.tick_params(axis="x", which="both", labelbottom=False)
    ax_track.legend(
        frameon=False,
        loc="upper right",
        ncol=2,
        columnspacing=1.5,
        handlelength=2.0,
    )
    ax_track.spines["top"].set_visible(False)
    ax_track.spines["right"].set_visible(False)

    # Bottom: signed mutant-reference effect for A/C/G/T substitutions.
    image = ax_heatmap.imshow(
        data.matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        norm=norm,
        extent=[x_left, x_right, 3.5, -0.5],
    )
    ax_heatmap.set_yticks(np.arange(4))
    ax_heatmap.set_yticklabels(BASES)
    ax_heatmap.add_patch(
        Rectangle(
            (core_left, -0.48),
            core_width,
            3.96,
            fill=False,
            edgecolor=MOTIF_COLOR,
            linewidth=1.5,
        )
    )
    ax_heatmap.axvline(
        selected_x,
        color=MOTIF_COLOR,
        linewidth=1.0,
        linestyle=":",
    )
    ax_heatmap.spines["top"].set_visible(False)
    ax_heatmap.spines["right"].set_visible(False)
    ax_heatmap.set_xlim(x_left, x_right)

    center_crop = example.seqlet_center_crop
    ax_heatmap.xaxis.set_major_formatter(
        FuncFormatter(
            lambda value, _position: (
                f"{genomic_position_1based(data.manifest_row, center_crop + int(round(value))):,}"
            )
        )
    )
    ax_heatmap.set_xticks(np.linspace(relative[0], relative[-1], 5))
    ax_heatmap.set_xlabel(
        f"Chromosome {chromosome_label(data.manifest_row.chrom)}"
    )

    # One shared vertical title, while A/C/G/T remain the heatmap row labels.
    fig.text(
        0.030,
        0.445,
        "Δ predicted log1p(counts)",
        rotation=90,
        ha="center",
        va="center",
        fontsize=14,
    )
    colorbar = fig.colorbar(image, cax=ax_colorbar)
    colorbar.set_ticks([-heatmap_limit, 0.0, heatmap_limit])
    colorbar.set_ticklabels(
        [f"{-heatmap_limit:.2f}", "0", f"{heatmap_limit:.2f}"]
    )
    colorbar.ax.tick_params(labelsize=11, length=3)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    # No bbox_inches='tight': preserve the requested exact 10 x 4 inch size.
    fig.savefig(output_pdf)
    if output_png is not None:
        fig.savefig(output_png, dpi=300)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    validate_args(args)
    configure_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(args.test_dir / "manifest.tsv")
    plot_data = [
        load_plot_data(example, args.source_dir, manifest)
        for example in EXAMPLES
    ]
    heatmap_limit = robust_shared_limit(plot_data)
    curve_limit = shared_curve_limit(plot_data, args.smooth_sigma)

    summaries: list[dict[str, object]] = []
    for data in plot_data:
        example = data.example
        output_pdf = args.output_dir / f"{example.tf_name}_saturation_example.pdf"
        output_png = (
            args.output_dir / f"{example.tf_name}_saturation_example.png"
            if args.write_png
            else None
        )
        draw_example(
            data,
            args.smooth_sigma,
            heatmap_limit,
            curve_limit,
            output_pdf,
            output_png,
        )

        selected_column = int(
            np.where(data.positions_crop == example.selected_position_crop)[0][0]
        )
        selected_alt_code = int(
            np.where(BASES == example.selected_alternative_base)[0][0]
        )
        selected_effect = float(data.matrix[selected_alt_code, selected_column])
        summaries.append(
            {
                **asdict(example),
                "sample_id": str(data.manifest_row.sample_id),
                "chrom": str(data.manifest_row.chrom),
                "gene_strand": str(data.manifest_row.strand),
                "observed_count": float(data.manifest_row["count"]),
                "reference_predicted_log1p_count": data.reference_prediction,
                "selected_genomic_position_hg19_1based": genomic_position_1based(
                    data.manifest_row, example.selected_position_crop
                ),
                "selected_delta_mutant_minus_reference": selected_effect,
                "figure_pdf": str(output_pdf),
            }
        )
        print(
            f"WROTE {output_pdf} | {mutation_text(example)} | "
            f"Δ(mut-ref)={selected_effect:+.4f}",
            flush=True,
        )

    summary_path = args.output_dir / "PATZ1_ZNF610_fixed_examples.tsv"
    pd.DataFrame(summaries).to_csv(summary_path, sep="\t", index=False)
    metadata_path = args.output_dir / "PATZ1_ZNF610_replot_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "figure_size_inches": [10, 4],
                "style": "final-figure-small typography; width doubled",
                "effect_definition": (
                    "predicted_log1p_count(mutant) - "
                    "predicted_log1p_count(reference)"
                ),
                "gain_curve": "maximum positive effect per position",
                "loss_curve": "magnitude of most negative effect per position",
                "curve_smoothing_sigma_bp": args.smooth_sigma,
                "heatmap_smoothing": "none",
                "shared_heatmap_limit": heatmap_limit,
                "shared_curve_ymax": curve_limit,
                "examples": summaries,
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"Summary: {summary_path}", flush=True)
    print(f"Metadata: {metadata_path}", flush=True)
    return 0


def mutation_text(example: FixedExample) -> str:
    return (
        f"{example.selected_reference_base}>{example.selected_alternative_base}: "
        f"{example.tf_name} loss"
    )


if __name__ == "__main__":
    raise SystemExit(main())
