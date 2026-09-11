#!/usr/bin/env python3
"""Draw a self-contained main-figure JS-distance CDF.

The figure compares a within-window position-randomized measured profile, the
predicted profile, and count-split pseudoreplicates. Multiple reproducible
pseudoreplicate splits can be generated; their JS distances are aggregated by
the per-window median. The figure has no title or lower-right annotation.
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
from scipy.special import xlogy  # noqa: E402


RESOLUTION = 20
COUNT_THRESHOLD = 50.0
FIGSIZE = (5, 4)
TICK_FONTSIZE = 12
LABEL_FONTSIZE = 14
LEGEND_FONTSIZE = 12

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
DEFAULT_OUTPUT = (
    DEFAULT_FIG1C_DIR
    / "hek293t_netseq_js_distance_cdf_main_res20bp_count_ge_50.pdf"
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
            "Draw the main JS-distance CDF at resolution=20 bp and "
            "observed_counts >= 50."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument(
        "--pos-bw",
        nargs="+",
        type=Path,
        default=[DEFAULT_POS_BW],
        help="One or more positive-strand source-count bigWigs.",
    )
    parser.add_argument(
        "--neg-bw",
        nargs="+",
        type=Path,
        default=[DEFAULT_NEG_BW],
        help="Negative-strand bigWigs matching --pos-bw.",
    )
    parser.add_argument(
        "--signal-unit",
        nargs="+",
        type=float,
        default=[1.0],
        help=(
            "Signal value representing one read in each bigWig pair. A "
            "single value is broadcast to all pairs."
        ),
    )
    parser.add_argument("--pseudoreplicate-seed", type=int, default=20260727)
    parser.add_argument(
        "--pseudoreplicate-repeats",
        type=int,
        default=10,
        help=(
            "Number of independent count splits. The plotted "
            "pseudoreplicate value is the per-window median JS distance."
        ),
    )
    parser.add_argument("--random-seed", type=int, default=20260729)
    parser.add_argument(
        "--resolution",
        type=int,
        default=RESOLUTION,
        metavar="BP",
        help="Non-overlapping bin width used for JS-distance calculation.",
    )
    parser.add_argument(
        "--count-threshold",
        type=float,
        default=COUNT_THRESHOLD,
        help="Require observed_counts >= this value.",
    )
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--observed-atol", type=float, default=1e-3)
    parser.add_argument("--skip-observed-check", action="store_true")
    parser.add_argument("--output-pdf", type=Path, default=DEFAULT_OUTPUT)
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
    rng: np.random.Generator,
    verify_observed: bool,
) -> tuple[np.ndarray, np.ndarray, int]:
    n_rows, profile_length = observed.shape
    pseudorep1 = np.zeros_like(observed, dtype=np.float32)
    pseudorep2 = np.zeros_like(observed, dtype=np.float32)
    assigned = np.zeros(n_rows, dtype=bool)
    handle_pairs = [
        (pyBigWig.open(str(pos_path)), pyBigWig.open(str(neg_path)))
        for pos_path, neg_path in zip(args.pos_bw, args.neg_bw)
    ]
    if any(
        pos_handle is None or neg_handle is None
        for pos_handle, neg_handle in handle_pairs
    ):
        raise OSError("Could not open one or more bigWig pairs.")
    n_regions = 0
    try:
        for strand in ("+", "-"):
            strand_handles = [
                pos_handle if strand == "+" else neg_handle
                for pos_handle, neg_handle in handle_pairs
            ]
            chrom_sizes = [handle.chroms() for handle in strand_handles]
            records_by_chrom: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
            for index in np.flatnonzero(strands == strand):
                chrom = str(chroms[index])
                start = int(starts[index])
                end = int(ends[index])
                for source_index, sizes in enumerate(chrom_sizes, start=1):
                    if chrom not in sizes or end > int(sizes[chrom]):
                        raise ValueError(
                            "Invalid bigWig interval in source "
                            f"{source_index}: {chrom}:{start}-{end}."
                        )
                records_by_chrom[chrom].append((start, end, int(index)))
            for chrom in sorted(records_by_chrom):
                for region_start, region_end, members in merged_interval_groups(
                    records_by_chrom[chrom]
                ):
                    n_regions += 1
                    region_length = region_end - region_start
                    raw = np.zeros(region_length, dtype=np.float32)
                    split1 = np.zeros(region_length, dtype=np.float32)
                    split2 = np.zeros(region_length, dtype=np.float32)
                    for source_index, (handle, signal_unit) in enumerate(
                        zip(strand_handles, args.signal_unit),
                        start=1,
                    ):
                        signal = handle.values(
                            chrom,
                            region_start,
                            region_end,
                            numpy=True,
                        )
                        source_raw, source_split1, source_split2 = (
                            split_count_signal(
                                signal,
                                signal_unit,
                                rng,
                                (
                                    f"{chrom}:{region_start}-{region_end},"
                                    f" source {source_index}"
                                ),
                            )
                        )
                        raw += source_raw
                        split1 += source_split1
                        split2 += source_split2
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
                            verify_observed
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
        for pos_handle, neg_handle in handle_pairs:
            if pos_handle is not None:
                pos_handle.close()
            if neg_handle is not None:
                neg_handle.close()
    if not np.all(assigned):
        missing = np.flatnonzero(~assigned)
        raise RuntimeError(f"Pseudoreplicate row {int(missing[0])} was not assigned.")
    return pseudorep1, pseudorep2, n_regions


def bin_profiles(
    profiles: np.ndarray,
    resolution: int,
) -> np.ndarray:
    profiles = np.asarray(profiles, dtype=np.float64)
    usable = profiles.shape[1] // resolution * resolution
    if usable == 0:
        raise ValueError("--resolution exceeds the profile length.")
    return profiles[:, :usable].reshape(
        profiles.shape[0],
        usable // resolution,
        resolution,
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


def jensen_shannon_distance(
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    """Return numerically stable base-2 Jensen-Shannon distance by row."""

    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first = first / np.sum(first, axis=1, keepdims=True)
    second = second / np.sum(second, axis=1, keepdims=True)
    mixture = 0.5 * (first + second)
    first_ratio = np.divide(
        first,
        mixture,
        out=np.ones_like(first),
        where=mixture > 0,
    )
    second_ratio = np.divide(
        second,
        mixture,
        out=np.ones_like(second),
        where=mixture > 0,
    )
    divergence = 0.5 * (
        np.sum(xlogy(first, first_ratio), axis=1)
        + np.sum(xlogy(second, second_ratio), axis=1)
    ) / np.log(2.0)
    return np.sqrt(np.clip(divergence, 0.0, 1.0))


def calculate_reference_distances(
    args: argparse.Namespace,
    observed: np.ndarray,
    predicted: np.ndarray,
    masks: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rows: list[np.ndarray] = []
    parts: dict[str, list[np.ndarray]] = {
        "random": [],
        "predicted": [],
    }
    rng = np.random.default_rng(args.random_seed)
    for start in range(0, len(observed), args.chunk_size):
        end = min(start + args.chunk_size, len(observed))
        obs = bin_profiles(observed[start:end], args.resolution)
        pred = bin_profiles(predicted[start:end], args.resolution)
        order = np.argsort(rng.random(obs.shape), axis=1)
        random_profile = np.take_along_axis(obs, order, axis=1)
        valid = (
            np.asarray(masks[start:end], dtype=bool)
            & valid_profile_rows(obs, pred, random_profile)
        )
        if not np.any(valid):
            continue
        rows.append(np.flatnonzero(valid) + start)
        parts["random"].append(
            jensen_shannon_distance(
                obs[valid],
                random_profile[valid],
            )
        )
        parts["predicted"].append(
            jensen_shannon_distance(
                obs[valid],
                pred[valid],
            )
        )
    if not rows:
        raise ValueError("No valid test windows remain.")
    row_indices = np.concatenate(rows).astype(np.int64, copy=False)
    distances = {
        method: np.clip(np.concatenate(values), 0.0, 1.0)
        for method, values in parts.items()
    }
    return row_indices, distances


def calculate_pseudoreplicate_distances(
    args: argparse.Namespace,
    pseudorep1: np.ndarray,
    pseudorep2: np.ndarray,
    masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate one split's pseudoreplicate JS distance per valid row."""

    rows: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for start in range(0, len(pseudorep1), args.chunk_size):
        end = min(start + args.chunk_size, len(pseudorep1))
        pseudo1 = bin_profiles(
            pseudorep1[start:end],
            args.resolution,
        )
        pseudo2 = bin_profiles(
            pseudorep2[start:end],
            args.resolution,
        )
        valid = (
            np.asarray(masks[start:end], dtype=bool)
            & valid_profile_rows(pseudo1, pseudo2)
        )
        if not np.any(valid):
            continue
        rows.append(np.flatnonzero(valid) + start)
        values.append(
            jensen_shannon_distance(
                pseudo1[valid],
                pseudo2[valid],
            )
        )
    if not rows:
        raise ValueError(
            "No valid pseudoreplicate windows remain for this split."
        )
    return (
        np.concatenate(rows).astype(np.int64, copy=False),
        np.clip(np.concatenate(values), 0.0, 1.0),
    )


def empirical_cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=np.float64))
    y = np.arange(1, len(x) + 1, dtype=np.float64) / len(x)
    return x, y


def validate_args(args: argparse.Namespace) -> None:
    if len(args.pos_bw) != len(args.neg_bw):
        raise ValueError("--pos-bw and --neg-bw must contain equal counts.")
    if len(args.signal_unit) == 1 and len(args.pos_bw) > 1:
        args.signal_unit = args.signal_unit * len(args.pos_bw)
    if len(args.signal_unit) != len(args.pos_bw):
        raise ValueError(
            "--signal-unit must contain one value or one value per "
            "positive/negative bigWig pair."
        )
    paths_and_labels = [(args.npz, "NPZ")]
    paths_and_labels.extend(
        (path, f"positive-strand bigWig {index}")
        for index, path in enumerate(args.pos_bw, start=1)
    )
    paths_and_labels.extend(
        (path, f"negative-strand bigWig {index}")
        for index, path in enumerate(args.neg_bw, start=1)
    )
    for path, label in paths_and_labels:
        if not path.is_file():
            raise FileNotFoundError(f"{label} is missing: {path}")
    if any(
        not np.isfinite(signal_unit) or signal_unit <= 0
        for signal_unit in args.signal_unit
    ):
        raise ValueError("--signal-unit values must be finite and positive.")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive.")
    if args.resolution <= 0:
        raise ValueError("--resolution must be positive.")
    if (
        not np.isfinite(args.count_threshold)
        or args.count_threshold < 0
    ):
        raise ValueError(
            "--count-threshold must be finite and non-negative."
        )
    if args.pseudoreplicate_seed < 0 or args.random_seed < 0:
        raise ValueError("Random seeds must be non-negative.")
    if args.pseudoreplicate_repeats <= 0:
        raise ValueError("--pseudoreplicate-repeats must be positive.")
    if args.output_pdf.suffix.lower() != ".pdf":
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
    rows, distances = calculate_reference_distances(
        args,
        observed,
        predicted,
        masks,
    )
    pseudoreplicate_distances = np.full(
        (args.pseudoreplicate_repeats, len(observed)),
        np.nan,
        dtype=np.float64,
    )
    n_regions: int | None = None
    for repeat_index in range(args.pseudoreplicate_repeats):
        split_number = repeat_index + 1
        split_seed = args.pseudoreplicate_seed + repeat_index
        print(
            "  pseudoreplicate split "
            f"{split_number}/{args.pseudoreplicate_repeats} "
            f"(seed={split_seed})",
            flush=True,
        )
        pseudorep1, pseudorep2, repeat_n_regions = (
            build_pseudoreplicates(
                args,
                observed,
                chroms,
                starts,
                ends,
                strands,
                rng=np.random.default_rng(split_seed),
                verify_observed=(
                    repeat_index == 0
                    and not args.skip_observed_check
                ),
            )
        )
        if n_regions is None:
            n_regions = repeat_n_regions
        elif repeat_n_regions != n_regions:
            raise RuntimeError(
                "Merged-region count changed between pseudoreplicate "
                "splits."
            )
        pseudo_rows, pseudo_values = (
            calculate_pseudoreplicate_distances(
                args,
                pseudorep1,
                pseudorep2,
                masks,
            )
        )
        pseudoreplicate_distances[
            repeat_index,
            pseudo_rows,
        ] = pseudo_values

    valid_repeat_counts = np.sum(
        np.isfinite(pseudoreplicate_distances),
        axis=0,
    )
    median_by_row = np.full(len(observed), np.nan, dtype=np.float64)
    has_valid_repeat = valid_repeat_counts > 0
    median_by_row[has_valid_repeat] = np.nanmedian(
        pseudoreplicate_distances[:, has_valid_repeat],
        axis=0,
    )
    reference_has_pseudoreplicate = np.isfinite(median_by_row[rows])
    rows = rows[reference_has_pseudoreplicate]
    distances = {
        method: values[reference_has_pseudoreplicate]
        for method, values in distances.items()
    }
    distances["pseudoreplicates"] = median_by_row[rows]
    if n_regions is None:
        raise RuntimeError("No pseudoreplicate split was generated.")
    return rows, distances, observed_counts, n_regions


def plot_main(
    distances: dict[str, np.ndarray],
    output_pdf: Path,
) -> None:
    apply_final_figure_small_style()
    fig, ax = plt.subplots(figsize=FIGSIZE)

    for method, label in METHODS:
        x, y = empirical_cdf(distances[method])
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
    ax.set_ylabel("Cum. Frac. Test Windows", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="x", labelsize=TICK_FONTSIZE)
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
    ax.legend(
        loc="upper left",
        fontsize=LEGEND_FONTSIZE,
        frameon=False,
        handlelength=1.8,
        handletextpad=0.5,
        borderaxespad=0.2,
    )

    fig.tight_layout()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, format="pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    validate_args(args)
    rows, distances, observed_counts, n_regions = calculate_distances(args)
    keep = observed_counts[rows] >= args.count_threshold
    if not np.any(keep):
        raise ValueError(
            "No test windows satisfy observed_counts >= "
            f"{args.count_threshold:g}."
        )
    selected = {
        method: values[keep]
        for method, values in distances.items()
    }
    plot_main(selected, args.output_pdf)

    print(f"[main figure] {args.npz.parent.name} JS-distance CDF")
    print(f"  resolution: {args.resolution} bp")
    print(
        "  count threshold: observed_counts >= "
        f"{args.count_threshold:g}"
    )
    print(
        "  pseudoreplicate splits: "
        f"{args.pseudoreplicate_repeats} "
        "(per-window median JS distance)"
    )
    print(f"  n: {int(np.sum(keep)):,}")
    print(f"  pseudoreplicate merged regions: {n_regions:,}")
    for method, _label in METHODS:
        values = selected[method]
        print(
            f"  {method}: median={np.median(values):.4f}, "
            f"mean={np.mean(values):.4f}"
        )
    print(f"  wrote {args.output_pdf}")


if __name__ == "__main__":
    main()
