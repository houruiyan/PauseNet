#!/usr/bin/env python3
"""Plot one 5' splice-site observed/predicted profile as a vector PDF."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator
from scipy.spatial.distance import jensenshannon


ANCHOR = "5SS"
COLOR = "#D9656D"
DEFAULT_NPZ = Path(
    "/mnt/HDD8TB/houruiyan/pausing_site/2_train_model/evaluate/"
    "hek293t_netseq/test_profiles.npz"
)
DEFAULT_OUTPUT = Path(__file__).with_name(
    "hek293t_5ss_observed_predicted_example.pdf"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot one 5SS observed/predicted profile.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument(
        "--min-count",
        type=float,
        default=100.0,
        help="Minimum observed total count used to filter candidate windows.",
    )
    parser.add_argument(
        "--jsd",
        type=float,
        default=0.50,
        help=(
            "Maximum Jensen-Shannon distance allowed for a candidate; "
            "lower values require closer profile agreement."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=20,
        metavar="BP",
        help="Non-overlapping bin width used only for JSD calculation.",
    )
    parser.add_argument(
        "--min-nonzero-positions",
        type=int,
        default=0,
        help="Minimum number of positions with observed signal > 0.",
    )
    parser.add_argument(
        "--max-peak-fraction",
        type=float,
        default=1.0,
        help="Maximum observed single-position peak / observed total count.",
    )
    parser.add_argument(
        "--min-predicted-count-ratio",
        type=float,
        default=0.0,
        help="Minimum predicted_count / observed_count.",
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--figure-width",
        type=float,
        default=7.0,
        help="Physical PDF width in inches; figure height remains 4 inches.",
    )
    return parser.parse_args()


def bin_profiles(profiles: np.ndarray, resolution: int) -> np.ndarray:
    usable = (profiles.shape[1] // resolution) * resolution
    if usable == 0:
        raise ValueError("--resolution exceeds the profile length.")
    trimmed = profiles[:, :usable]
    return trimmed.reshape(
        len(trimmed),
        usable // resolution,
        resolution,
    ).sum(axis=2)


def select_example(
    npz_path: Path,
    min_count: float,
    max_jsd: float,
    resolution: int,
    min_nonzero_positions: int,
    max_peak_fraction: float,
    min_predicted_count_ratio: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    str,
    int,
    int,
    str,
    float,
    float,
    int,
    float,
    int,
    float,
]:
    if not npz_path.is_file():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")
    with np.load(npz_path, allow_pickle=True) as data:
        required = (
            "observed_profiles",
            "predicted_profiles",
            "observed_counts",
            "predicted_counts",
            "profile_masks",
            "manifest_sample_id",
            "manifest_chrom",
            "manifest_output_start",
            "manifest_output_end",
            "manifest_strand",
        )
        missing = [key for key in required if key not in data.files]
        if missing:
            raise KeyError("Missing NPZ arrays: " + ", ".join(missing))
        observed_all = np.asarray(
            data["observed_profiles"],
            dtype=np.float64,
        )
        predicted_all = np.asarray(
            data["predicted_profiles"],
            dtype=np.float64,
        )
        observed_counts = np.asarray(
            data["observed_counts"],
            dtype=np.float64,
        )
        predicted_counts = np.asarray(
            data["predicted_counts"],
            dtype=np.float64,
        )
        sample_ids = np.asarray(data["manifest_sample_id"]).astype(str)
        masks = np.asarray(data["profile_masks"], dtype=bool)
        region_mask = np.array(
            [f"|{ANCHOR}|" in sample_id for sample_id in sample_ids],
            dtype=bool,
        )
        candidates = np.flatnonzero(
            region_mask
            & masks
            & np.isfinite(observed_counts)
            & np.isfinite(predicted_counts)
            & (observed_counts > 0)
            & (observed_counts >= min_count)
            & (predicted_counts > 0)
        )
        if not len(candidates):
            raise ValueError(
                f"No {ANCHOR} windows satisfy observed_counts >= "
                f"{min_count:g}."
            )

        observed_candidates = observed_all[candidates]
        predicted_candidates = predicted_all[candidates]
        valid = (
            np.all(np.isfinite(observed_candidates), axis=1)
            & np.all(np.isfinite(predicted_candidates), axis=1)
            & np.all(observed_candidates >= 0, axis=1)
            & np.all(predicted_candidates >= 0, axis=1)
            & (np.sum(observed_candidates, axis=1) > 0)
            & (np.sum(predicted_candidates, axis=1) > 0)
        )
        candidates = candidates[valid]
        observed_candidates = observed_candidates[valid]
        predicted_candidates = predicted_candidates[valid]
        if not len(candidates):
            raise ValueError(f"No valid {ANCHOR} profiles remain.")

        js_distances = np.asarray(
            jensenshannon(
                bin_profiles(observed_candidates, resolution),
                bin_profiles(predicted_candidates, resolution),
                base=2,
                axis=1,
            ),
            dtype=np.float64,
        )
        keep = np.isfinite(js_distances) & (js_distances <= max_jsd)
        if not np.any(keep):
            minimum = float(np.nanmin(js_distances))
            raise ValueError(
                f"No {ANCHOR} windows satisfy JSD <= {max_jsd:g} at "
                f"{resolution} bp; the minimum among count-passing windows "
                f"is {minimum:.6f}."
            )
        candidates = candidates[keep]
        js_distances = js_distances[keep]
        observed_candidates = observed_candidates[keep]
        peaks = np.max(observed_candidates, axis=1)
        nonzero_positions = np.count_nonzero(
            observed_candidates > 0,
            axis=1,
        )
        peak_fractions = peaks / observed_counts[candidates]
        predicted_count_ratios = (
            predicted_counts[candidates] / observed_counts[candidates]
        )
        shape_keep = (
            (nonzero_positions >= min_nonzero_positions)
            & (peak_fractions <= max_peak_fraction)
            & (
                predicted_count_ratios
                >= min_predicted_count_ratio
            )
        )
        if not np.any(shape_keep):
            raise ValueError(
                f"No {ANCHOR} windows remain after shape filters: "
                f"nonzero positions >= {min_nonzero_positions}, "
                f"peak fraction <= {max_peak_fraction:g}, and "
                "predicted/observed count ratio >= "
                f"{min_predicted_count_ratio:g}."
            )
        candidates = candidates[shape_keep]
        js_distances = js_distances[shape_keep]
        observed_candidates = observed_candidates[shape_keep]
        peaks = peaks[shape_keep]
        order = np.lexsort(
            (
                candidates,
                -observed_counts[candidates],
                js_distances,
                -peaks,
            )
        )
        position = int(order[0])
        index = int(candidates[position])
        observed = observed_all[index]
        predicted_count = float(predicted_counts[index])
        predicted = predicted_all[index] * predicted_count
        return (
            observed,
            predicted,
            str(data["manifest_chrom"][index]),
            int(data["manifest_output_start"][index]),
            int(data["manifest_output_end"][index]),
            str(data["manifest_strand"][index]),
            float(observed_counts[index]),
            predicted_count,
            index,
            float(js_distances[position]),
            int(len(candidates)),
            float(peaks[position]),
        )


def configure_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(axis: plt.Axes, ylabel: str) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.set_ylabel(
        ylabel,
        fontsize=14,
        rotation=0,
        ha="right",
        va="center",
        labelpad=10,
    )
    axis.tick_params(axis="both", labelsize=12)
    axis.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))


def draw_track(
    axis: plt.Axes,
    x: np.ndarray,
    values: np.ndarray,
) -> None:
    axis.fill_between(
        x,
        0,
        values,
        step="mid",
        color=COLOR,
        alpha=0.82,
        linewidth=0,
    )
    axis.plot(x, values, color=COLOR, linewidth=0.55)
    maximum = float(np.max(values))
    axis.set_ylim(0, maximum * 1.08 if maximum > 0 else 1)
    axis.margins(x=0)


def main() -> None:
    args = parse_args()
    if args.output_pdf.suffix.lower() != ".pdf":
        raise ValueError("--output-pdf must end in .pdf.")
    if not np.isfinite(args.min_count) or args.min_count < 0:
        raise ValueError("--min-count must be finite and non-negative.")
    if not np.isfinite(args.jsd) or not 0 <= args.jsd <= 1:
        raise ValueError("--jsd must be finite and between 0 and 1.")
    if args.resolution <= 0:
        raise ValueError("--resolution must be positive.")
    if args.min_nonzero_positions < 0:
        raise ValueError("--min-nonzero-positions must be non-negative.")
    if (
        not np.isfinite(args.max_peak_fraction)
        or not 0 < args.max_peak_fraction <= 1
    ):
        raise ValueError("--max-peak-fraction must be in (0, 1].")
    if (
        not np.isfinite(args.min_predicted_count_ratio)
        or args.min_predicted_count_ratio < 0
    ):
        raise ValueError(
            "--min-predicted-count-ratio must be finite and non-negative."
        )
    if not np.isfinite(args.figure_width) or args.figure_width <= 0:
        raise ValueError("--figure-width must be finite and positive.")
    (
        observed,
        predicted,
        chrom,
        start,
        end,
        strand,
        observed_count,
        predicted_count,
        selected_index,
        selected_jsd,
        candidate_count,
        observed_peak,
    ) = select_example(
        args.npz,
        args.min_count,
        args.jsd,
        args.resolution,
        args.min_nonzero_positions,
        args.max_peak_fraction,
        args.min_predicted_count_ratio,
    )
    if strand == "-":
        observed = observed[::-1]
        predicted = predicted[::-1]

    configure_style()
    x = start + np.arange(len(observed))
    figure, (observed_axis, predicted_axis) = plt.subplots(
        2,
        1,
        figsize=(args.figure_width, 4),
        sharex=True,
        gridspec_kw={"hspace": 0.08},
    )
    draw_track(observed_axis, x, observed)
    draw_track(predicted_axis, x, predicted)
    style_axis(observed_axis, "Obs")
    style_axis(predicted_axis, "Pred")
    observed_axis.tick_params(axis="x", labelbottom=False)

    midpoint = start + (end - start) // 2
    predicted_axis.xaxis.set_major_locator(
        FixedLocator((start, midpoint, end))
    )
    predicted_axis.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value:,.0f}")
    )
    coordinate_padding = max((end - start) * 0.02, 1.0)
    predicted_axis.set_xlim(
        start - coordinate_padding,
        end + coordinate_padding,
    )
    predicted_axis.set_xlabel(f"{chrom} ({strand})", fontsize=14)
    figure.subplots_adjust(
        left=0.15,
        right=0.94,
        bottom=0.16,
        top=0.98,
    )
    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_pdf, format="pdf")
    plt.close(figure)
    print(
        f"{ANCHOR}: row={selected_index}, "
        f"{chrom}:{start:,}-{end:,} "
        f"({strand}), observed={observed_count:.6g}, "
        f"predicted={predicted_count:.6g}, "
        f"observed_peak={observed_peak:.6g}, "
        f"JSD={selected_jsd:.6f} at {args.resolution} bp, "
        f"passing_candidates={candidate_count}"
    )
    print(
        f"  nonzero_positions={np.count_nonzero(observed > 0)}, "
        f"peak_fraction={observed_peak / observed_count:.6f}, "
        "predicted/observed_count="
        f"{predicted_count / observed_count:.6f}"
    )
    print(f"Saved PDF: {args.output_pdf}")


if __name__ == "__main__":
    main()
