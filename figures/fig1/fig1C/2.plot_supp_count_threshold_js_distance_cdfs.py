#!/usr/bin/env python3
"""Draw three self-contained count-threshold JS-distance CDF panels.

All panels use 20-bp resolution and the same random-profile, predicted-profile,
and pseudoreplicate comparisons. Each panel reports its count threshold and n;
only the first panel contains a legend.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyBigWig  # noqa: E402
from matplotlib.ticker import FormatStrFormatter  # noqa: E402
from scipy.spatial.distance import jensenshannon  # noqa: E402


RESOLUTION = 20
PANEL_FIGSIZE = (5, 4)
N_PANELS = 3
FIGSIZE = (PANEL_FIGSIZE[0] * N_PANELS, PANEL_FIGSIZE[1])
TICK_FONTSIZE = 12
LABEL_FONTSIZE = 14
LEGEND_FONTSIZE = 12
ANNOTATION_FONTSIZE = 12

DEFAULT_FIG1C_DIR = Path(
    "/mnt/HDD8TB/houruiyan/pausing_site/4_plot_figure/fig1/fig1C"
)
DEFAULT_NPZ = Path(
    "/mnt/HDD8TB/houruiyan/pausing_site/2_train_model/evaluate/"
    "hek293t_netseq/test_profiles.npz"
)
DEFAULT_POS_BW = Path(
    "/mnt/HDD8TB/houruiyan/pausing_site/data/HEK293T_NETseq/"
    "merge/merged/HEK293T.merged.pos.bw"
)
DEFAULT_NEG_BW = Path(
    "/mnt/HDD8TB/houruiyan/pausing_site/data/HEK293T_NETseq/"
    "merge/merged/HEK293T.merged.neg.bw"
)
METHODS = (
    ("random", "Random Profile vs. Measured"),
    ("predicted", "Predicted vs. Measured"),
    ("pseudoreplicates", "Pseudoreplicates"),
)
COLORS = {
    "random": "#E71D36",
    "predicted": "#2EC4B6",
    "pseudoreplicates": "#30A9DE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Draw three HEK293T NET-seq count-threshold JS-distance CDF "
            "panels at 20-bp resolution."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--pos-bw", type=Path, default=DEFAULT_POS_BW)
    parser.add_argument("--neg-bw", type=Path, default=DEFAULT_NEG_BW)
    parser.add_argument("--signal-unit", type=float, default=1.0)
    parser.add_argument("--pseudoreplicate-seed", type=int, default=20260727)
    parser.add_argument("--random-seed", type=int, default=20260729)
    parser.add_argument(
        "--count-thresholds",
        nargs=3,
        type=float,
        default=[50.0, 100.0, 200.0],
        metavar=("COUNT1", "COUNT2", "COUNT3"),
        help="Exactly three observed-count thresholds.",
    )
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--observed-atol", type=float, default=1e-3)
    parser.add_argument("--skip-observed-check", action="store_true")
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=None,
        help=(
            "Optional explicit output PDF. By default, the filename is "
            "generated from the three count thresholds."
        ),
    )
    return parser.parse_args()


def apply_final_figure_small_style() -> None:
    mpl.rcParams.update(
        {
            "font.size": TICK_FONTSIZE,
            "axes.labelsize": LABEL_FONTSIZE,
            "xtick.labelsize": TICK_FONTSIZE,
            "ytick.labelsize": TICK_FONTSIZE,
            "legend.fontsize": LEGEND_FONTSIZE,
            "legend.frameon": False,
            "pdf.fonttype": 42,
        }
    )


def load_evaluation_data(
    npz_path: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    required = {
        "observed_profiles",
        "predicted_profiles",
        "manifest_chrom",
        "manifest_output_start",
        "manifest_output_end",
        "manifest_strand",
    }
    with np.load(npz_path, allow_pickle=False) as data:
        missing = sorted(required.difference(data.files))
        if missing:
            raise KeyError(f"NPZ is missing: {', '.join(missing)}")
        observed = np.asarray(data["observed_profiles"], dtype=np.float32)
        predicted = np.asarray(data["predicted_profiles"], dtype=np.float32)
        chroms = np.asarray(data["manifest_chrom"]).astype(str)
        starts = np.asarray(data["manifest_output_start"], dtype=np.int64)
        ends = np.asarray(data["manifest_output_end"], dtype=np.int64)
        strands = np.asarray(data["manifest_strand"]).astype(str)
        masks = (
            np.asarray(data["profile_masks"], dtype=bool)
            if "profile_masks" in data.files
            else np.ones(len(observed), dtype=bool)
        )
        counts = (
            np.asarray(data["observed_counts"], dtype=np.float64)
            if "observed_counts" in data.files
            else np.sum(observed, axis=1, dtype=np.float64)
        )
    if observed.ndim != 2 or predicted.shape != observed.shape:
        raise ValueError("Observed and predicted profiles must be matching 2D arrays.")
    n_rows, profile_length = observed.shape
    for label, values in (
        ("chrom", chroms),
        ("start", starts),
        ("end", ends),
        ("strand", strands),
        ("mask", masks),
        ("count", counts),
    ):
        if values.shape != (n_rows,):
            raise ValueError(f"{label} has shape {values.shape}; expected {(n_rows,)}.")
    if np.any((ends - starts) != profile_length):
        raise ValueError("Manifest output intervals do not match profile length.")
    if set(strands).difference({"+", "-"}):
        raise ValueError("Manifest contains an invalid strand.")
    if not np.all(np.isfinite(counts)) or np.any(counts < 0):
        raise ValueError("Observed counts contain invalid values.")
    return observed, predicted, masks, counts, chroms, starts, ends, strands


def merged_interval_groups(
    records: list[tuple[int, int, int]],
) -> Iterable[tuple[int, int, list[tuple[int, int, int]]]]:
    if not records:
        return
    ordered = sorted(records)
    region_start, region_end, index = ordered[0]
    members = [(region_start, region_end, index)]
    for start, end, index in ordered[1:]:
        if start <= region_end:
            region_end = max(region_end, end)
            members.append((start, end, index))
        else:
            yield region_start, region_end, members
            region_start, region_end = start, end
            members = [(start, end, index)]
    yield region_start, region_end, members


def split_count_signal(
    signal: np.ndarray,
    signal_unit: float,
    rng: np.random.Generator,
    region: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    signal = np.abs(
        np.nan_to_num(
            np.asarray(signal, dtype=np.float64),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
    )
    count_values = signal / signal_unit
    counts = np.rint(count_values)
    if not np.allclose(count_values, counts, rtol=0.0, atol=5e-3):
        raise ValueError(f"bigWig signal is not count-scaled at {region}.")
    counts = counts.astype(np.int64, copy=False)
    counts1 = rng.binomial(counts, 0.5)
    counts2 = counts - counts1
    return (
        (counts * signal_unit).astype(np.float32),
        (counts1 * signal_unit).astype(np.float32),
        (counts2 * signal_unit).astype(np.float32),
    )


def build_pseudoreplicates(
    args: argparse.Namespace,
    observed: np.ndarray,
    chroms: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    strands: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    n_rows, profile_length = observed.shape
    pseudorep1 = np.zeros_like(observed, dtype=np.float32)
    pseudorep2 = np.zeros_like(observed, dtype=np.float32)
    assigned = np.zeros(n_rows, dtype=bool)
    rng = np.random.default_rng(args.pseudoreplicate_seed)
    handles = {
        "+": pyBigWig.open(str(args.pos_bw)),
        "-": pyBigWig.open(str(args.neg_bw)),
    }
    if any(handle is None for handle in handles.values()):
        raise OSError("Could not open the strand-specific bigWigs.")
    n_regions = 0
    try:
        for strand, handle in handles.items():
            chrom_sizes = handle.chroms()
            records_by_chrom: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
            for index in np.flatnonzero(strands == strand):
                chrom = str(chroms[index])
                start = int(starts[index])
                end = int(ends[index])
                if chrom not in chrom_sizes or end > int(chrom_sizes[chrom]):
                    raise ValueError(f"Invalid bigWig interval: {chrom}:{start}-{end}.")
                records_by_chrom[chrom].append((start, end, int(index)))
            for chrom in sorted(records_by_chrom):
                for region_start, region_end, members in merged_interval_groups(
                    records_by_chrom[chrom]
                ):
                    n_regions += 1
                    signal = handle.values(
                        chrom,
                        region_start,
                        region_end,
                        numpy=True,
                    )
                    raw, split1, split2 = split_count_signal(
                        signal,
                        args.signal_unit,
                        rng,
                        f"{chrom}:{region_start}-{region_end}",
                    )
                    for start, end, index in members:
                        left = start - region_start
                        right = end - region_start
                        raw_profile = raw[left:right]
                        profile1 = split1[left:right]
                        profile2 = split2[left:right]
                        if strand == "-":
                            raw_profile = raw_profile[::-1]
                            profile1 = profile1[::-1]
                            profile2 = profile2[::-1]
                        if len(raw_profile) != profile_length:
                            raise RuntimeError("Unexpected extracted profile length.")
                        if (
                            not args.skip_observed_check
                            and not np.allclose(
                                raw_profile,
                                observed[index],
                                rtol=1e-5,
                                atol=args.observed_atol,
                            )
                        ):
                            difference = float(
                                np.max(np.abs(raw_profile - observed[index]))
                            )
                            raise ValueError(
                                "bigWig does not reproduce observed profile at "
                                f"row {index}; max difference={difference:.6g}."
                            )
                        pseudorep1[index] = profile1
                        pseudorep2[index] = profile2
                        assigned[index] = True
    finally:
        for handle in handles.values():
            if handle is not None:
                handle.close()
    if not np.all(assigned):
        missing = np.flatnonzero(~assigned)
        raise RuntimeError(f"Pseudoreplicate row {int(missing[0])} was not assigned.")
    return pseudorep1, pseudorep2, n_regions


def bin_profiles(profiles: np.ndarray) -> np.ndarray:
    profiles = np.asarray(profiles, dtype=np.float64)
    usable = profiles.shape[1] // RESOLUTION * RESOLUTION
    return profiles[:, :usable].reshape(
        profiles.shape[0],
        usable // RESOLUTION,
        RESOLUTION,
    ).sum(axis=2)


def valid_profile_rows(*profiles: np.ndarray) -> np.ndarray:
    valid = np.ones(len(profiles[0]), dtype=bool)
    for profile in profiles:
        valid &= (
            np.all(np.isfinite(profile), axis=1)
            & np.all(profile >= 0, axis=1)
            & (np.sum(profile, axis=1) > 0)
        )
    return valid


def calculate_common_distances(
    args: argparse.Namespace,
    observed: np.ndarray,
    predicted: np.ndarray,
    pseudorep1: np.ndarray,
    pseudorep2: np.ndarray,
    masks: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rows: list[np.ndarray] = []
    parts: dict[str, list[np.ndarray]] = {
        "random": [],
        "predicted": [],
        "pseudoreplicates": [],
    }
    rng = np.random.default_rng(args.random_seed)
    for start in range(0, len(observed), args.chunk_size):
        end = min(start + args.chunk_size, len(observed))
        obs = bin_profiles(observed[start:end])
        pred = bin_profiles(predicted[start:end])
        pseudo1 = bin_profiles(pseudorep1[start:end])
        pseudo2 = bin_profiles(pseudorep2[start:end])
        order = np.argsort(rng.random(obs.shape), axis=1)
        random_profile = np.take_along_axis(obs, order, axis=1)
        valid = (
            np.asarray(masks[start:end], dtype=bool)
            & valid_profile_rows(obs, pred, pseudo1, pseudo2, random_profile)
        )
        if not np.any(valid):
            continue
        rows.append(np.flatnonzero(valid) + start)
        parts["random"].append(
            jensenshannon(obs[valid], random_profile[valid], base=2, axis=1)
        )
        parts["predicted"].append(
            jensenshannon(obs[valid], pred[valid], base=2, axis=1)
        )
        parts["pseudoreplicates"].append(
            jensenshannon(pseudo1[valid], pseudo2[valid], base=2, axis=1)
        )
    if not rows:
        raise ValueError("No valid test windows remain.")
    row_indices = np.concatenate(rows).astype(np.int64, copy=False)
    distances = {
        method: np.clip(np.concatenate(values), 0.0, 1.0)
        for method, values in parts.items()
    }
    return row_indices, distances


def empirical_cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=np.float64))
    y = np.arange(1, len(x) + 1, dtype=np.float64) / len(x)
    return x, y


def threshold_filename_token(value: float) -> str:
    return (
        f"{value:g}"
        .replace("-", "neg")
        .replace(".", "p")
    )


def automatic_output_pdf(thresholds: list[float]) -> Path:
    threshold_token = "_".join(
        threshold_filename_token(value)
        for value in thresholds
    )
    return (
        DEFAULT_FIG1C_DIR
        / (
            "hek293t_netseq_js_distance_cdf_supp_res20bp_"
            f"counts_{threshold_token}.pdf"
        )
    )


def validate_args(args: argparse.Namespace) -> None:
    for path, label in (
        (args.npz, "NPZ"),
        (args.pos_bw, "positive-strand bigWig"),
        (args.neg_bw, "negative-strand bigWig"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} is missing: {path}")
    if args.signal_unit <= 0:
        raise ValueError("--signal-unit must be positive.")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive.")
    if args.pseudoreplicate_seed < 0 or args.random_seed < 0:
        raise ValueError("Random seeds must be non-negative.")
    thresholds = np.asarray(args.count_thresholds, dtype=np.float64)
    if not np.all(np.isfinite(thresholds)) or np.any(thresholds < 0):
        raise ValueError("Count thresholds must be finite and non-negative.")
    if len(set(args.count_thresholds)) != N_PANELS:
        raise ValueError("The three count thresholds must be distinct.")
    args.count_thresholds = sorted(args.count_thresholds)
    if args.output_pdf is None:
        args.output_pdf = automatic_output_pdf(args.count_thresholds)
    elif args.output_pdf.suffix.lower() != ".pdf":
        raise ValueError("--output-pdf must end with .pdf.")


def calculate_distances(
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, int]:
    (
        observed,
        predicted,
        masks,
        observed_counts,
        chroms,
        starts,
        ends,
        strands,
    ) = load_evaluation_data(args.npz)
    pseudorep1, pseudorep2, n_regions = build_pseudoreplicates(
        args,
        observed,
        chroms,
        starts,
        ends,
        strands,
    )
    rows, distances = calculate_common_distances(
        args,
        observed,
        predicted,
        pseudorep1,
        pseudorep2,
        masks,
    )
    return rows, distances, observed_counts, n_regions


def threshold_text(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


def plot_supplement(
    rows: np.ndarray,
    distances: dict[str, np.ndarray],
    observed_counts: np.ndarray,
    thresholds: list[float],
    output_pdf: Path,
) -> list[int]:
    apply_final_figure_small_style()
    fig, axes = plt.subplots(
        1,
        N_PANELS,
        figsize=FIGSIZE,
        sharex=True,
        sharey=True,
    )
    panel_counts: list[int] = []

    for panel_index, (ax, threshold) in enumerate(zip(axes, thresholds)):
        keep = observed_counts[rows] >= threshold
        n_windows = int(np.sum(keep))
        if n_windows == 0:
            raise ValueError(
                f"No test windows satisfy observed_counts >= {threshold:g}."
            )
        panel_counts.append(n_windows)

        for method, label in METHODS:
            x, y = empirical_cdf(distances[method][keep])
            ax.plot(
                x,
                y,
                color=COLORS[method],
                linewidth=1.8,
                label=label,
            )

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks((0.0, 0.5, 1.0))
        ax.set_yticks((0.0, 0.5, 1.0))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.set_xlabel("Jensen-Shannon Distance", fontsize=LABEL_FONTSIZE)
        ax.tick_params(axis="x", labelsize=TICK_FONTSIZE)
        ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
        ax.text(
            0.97,
            0.04,
            f"count \u2265 {threshold_text(threshold)}\n"
            f"n = {n_windows:,}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=ANNOTATION_FONTSIZE,
        )
        if panel_index == 0:
            ax.set_ylabel(
                "Cum. Frac. Test Windows",
                fontsize=LABEL_FONTSIZE,
            )
            ax.legend(
                loc="upper left",
                fontsize=LEGEND_FONTSIZE,
                frameon=False,
                handlelength=1.8,
                handletextpad=0.5,
                borderaxespad=0.2,
            )

    fig.tight_layout(w_pad=1.5)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, format="pdf")
    plt.close(fig)
    return panel_counts


def main() -> None:
    args = parse_args()
    validate_args(args)
    rows, distances, observed_counts, n_regions = calculate_distances(args)
    panel_counts = plot_supplement(
        rows,
        distances,
        observed_counts,
        args.count_thresholds,
        args.output_pdf,
    )

    print("[supplement] HEK293T NET-seq JS-distance CDFs")
    print(f"  resolution: {RESOLUTION} bp")
    print(f"  pseudoreplicate merged regions: {n_regions:,}")
    for threshold, n_windows in zip(args.count_thresholds, panel_counts):
        print(
            f"  count >= {threshold_text(threshold)}: "
            f"n={n_windows:,}"
        )
    print(f"  wrote {args.output_pdf}")


if __name__ == "__main__":
    main()
