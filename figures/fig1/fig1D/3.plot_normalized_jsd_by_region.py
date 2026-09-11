#!/usr/bin/env python3
"""Plot normalized Jensen-Shannon distance by test-window class.

The script reads observed and predicted profiles from PauseNet's
``test_profiles.npz``. It calculates three Jensen-Shannon distances per valid
test window:

1. model prediction versus measured profile;
2. two count-split pseudoreplicates;
3. a position-randomized measured profile versus the original measured
   profile.

SciPy's ``jensenshannon`` function directly returns JS distance. The existing
per-example normalization is:

    normalized_js_distance = clip(
        (model_js_distance - pseudoreplicate_js_distance)
        / (random_js_distance - pseudoreplicate_js_distance),
        0,
        1,
    )

Here, pseudoreplicate performance is the upper performance bound and the
random profile is the lower performance bound. Consequently, normalized JS
distance is 0 for pseudoreplicate-level performance and 1 for random-profile
performance; lower values are better. The random baseline is generated after
resolution binning by independently permuting the measured bin positions
within each test window.

TSS, 5' splice-site, 3' splice-site, and TES labels are read from
``manifest_region_type`` when available and otherwise inferred from the
pipe-delimited ``manifest_sample_id`` field.

Example
-------
python 3.plot_normalized_jsd_by_region.py \
    --npz /path/to/hek293t_netseq/test_profiles.npz \
    --pos-bw /path/to/HEK293T.merged.pos.bw \
    --neg-bw /path/to/HEK293T.merged.neg.bw \
    --signal-unit 1 \
    --random-seed 20260729 \
    --resolution 20 \
    --count-threshold 0 \
    --output-pdf /path/to/hek293t_netseq_normalized_js_distance_random_baseline_res20bp.pdf
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyBigWig  # noqa: E402
from matplotlib.cbook import boxplot_stats  # noqa: E402
from matplotlib.ticker import FormatStrFormatter  # noqa: E402
from scipy.spatial.distance import jensenshannon  # noqa: E402


ANCHOR_ORDER = ("TSS", "5SS", "3SS", "TES")
DISPLAY_LABELS = {
    "TSS": "TSS",
    "5SS": "5' splice site",
    "3SS": "3' splice site",
    "TES": "TES",
}
COLORS = {
    "TSS": "#626262",
    "5SS": "#D9656D",
    "3SS": "#D9A83E",
    "TES": "#62A5BE",
}
DISPLAY_TITLES = {
    "hek293t_groseq": "HEK293T GRO-seq",
    "hek293t_netseq": "HEK293T NET-seq",
    "hek293t_proseq": "HEK293T PRO-seq",
}

# final-figure-small contract
FIGSIZE = (5, 4)
TICK_FONTSIZE = 12
LABEL_FONTSIZE = 14
TITLE_FONTSIZE = 14
LEGEND_FONTSIZE = 12
PANEL_LABEL_FONTSIZE = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate normalized per-window JS distance and plot its "
            "distribution by TSS/5SS/3SS/TES."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--npz",
        required=True,
        type=Path,
        help="PauseNet test_profiles.npz.",
    )

    parser.add_argument(
        "--pos-bw",
        nargs="+",
        required=True,
        type=Path,
        help="One or more positive-strand source-count bigWigs.",
    )
    parser.add_argument(
        "--neg-bw",
        nargs="+",
        required=True,
        type=Path,
        help="Negative-strand bigWigs matching --pos-bw.",
    )
    parser.add_argument(
        "--signal-unit",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Signal value representing one read in each bigWig pair; use 1 "
            "for integer-count tracks."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260727,
        help="Seed for reproducible 50:50 pseudoreplicate splitting.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=20260729,
        help=(
            "Seed for independently permuting the binned measured profile "
            "within every test window."
        ),
    )
    parser.add_argument(
        "--observed-atol",
        type=float,
        default=1e-3,
        help="Absolute tolerance for bigWig versus observed-profile QA.",
    )
    parser.add_argument(
        "--skip-observed-check",
        action="store_true",
        help="Skip source bigWig versus NPZ observed-profile verification.",
    )

    parser.add_argument(
        "--resolution",
        type=int,
        default=1,
        metavar="BP",
        help=(
            "Adjustable non-overlapping bin width in bp before JS-distance "
            "calculation, for example 1, 5, 10, or 20. If the profile length "
            "is not divisible by BP, trailing positions are discarded."
        ),
    )
    parser.add_argument(
        "--count-threshold",
        type=float,
        default=0.0,
        help="Require observed_counts >= this threshold.",
    )
    parser.add_argument(
        "--require-ordered-baselines",
        action="store_true",
        help=(
            "Require pseudoreplicate JS distance < random-profile JS "
            "distance. By default, normalization is clipped without this "
            "extra filter."
        ),
    )
    parser.add_argument(
        "--normalization-epsilon",
        type=float,
        default=1e-12,
        help="Minimum absolute normalization denominator.",
    )
    parser.add_argument("--chunk-size", type=int, default=512)

    parser.add_argument(
        "--output-pdf",
        required=True,
        type=Path,
        help="Output vector PDF.",
    )
    parser.add_argument(
        "--values-tsv",
        type=Path,
        default=None,
        help="Optional per-window source-data TSV.",
    )
    parser.add_argument(
        "--summary-tsv",
        type=Path,
        default=None,
        help="Optional per-group summary TSV.",
    )
    parser.add_argument("--title", default=None)
    parser.add_argument(
        "--show-title",
        action="store_true",
        help="Infer a dataset title from the NPZ parent directory.",
    )
    parser.add_argument("--panel-label", default=None)
    parser.add_argument("--jitter-seed", type=int, default=20260726)
    parser.add_argument("--point-size", type=float, default=4.0)
    parser.add_argument("--point-alpha", type=float, default=0.30)
    parser.add_argument(
        "--rasterize-points",
        action="store_true",
        help="Rasterize only the point clouds to reduce PDF size.",
    )
    parser.add_argument(
        "--hide-all-median",
        action="store_true",
        help="Hide the dashed median across all retained windows.",
    )

    parser.add_argument("--observed-key", default="observed_profiles")
    parser.add_argument("--predicted-key", default="predicted_profiles")
    parser.add_argument("--count-key", default="observed_counts")
    parser.add_argument("--mask-key", default="profile_masks")
    parser.add_argument("--sample-id-key", default="manifest_sample_id")
    parser.add_argument("--region-key", default="manifest_region_type")
    parser.add_argument("--chrom-key", default="manifest_chrom")
    parser.add_argument("--start-key", default="manifest_output_start")
    parser.add_argument("--end-key", default="manifest_output_end")
    parser.add_argument("--strand-key", default="manifest_strand")
    return parser.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def validate_args(args: argparse.Namespace) -> None:
    if len(args.pos_bw) != len(args.neg_bw):
        raise ValueError("--pos-bw and --neg-bw must have the same length.")
    if args.signal_unit is None:
        args.signal_unit = [1.0] * len(args.pos_bw)
    if len(args.signal_unit) != len(args.pos_bw):
        raise ValueError(
            "--signal-unit must contain one value per positive/negative pair."
        )
    if any(
        not np.isfinite(unit) or unit <= 0
        for unit in args.signal_unit
    ):
        raise ValueError("--signal-unit values must be finite and positive.")
    if args.resolution <= 0:
        raise ValueError("--resolution must be positive.")
    if not np.isfinite(args.count_threshold) or args.count_threshold < 0:
        raise ValueError("--count-threshold must be finite and non-negative.")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive.")
    if args.seed < 0 or args.random_seed < 0:
        raise ValueError("--seed and --random-seed must be non-negative.")
    if not np.isfinite(args.observed_atol) or args.observed_atol < 0:
        raise ValueError("--observed-atol must be finite and non-negative.")
    if (
        not np.isfinite(args.normalization_epsilon)
        or args.normalization_epsilon <= 0
    ):
        raise ValueError("--normalization-epsilon must be finite and positive.")
    if args.point_size <= 0:
        raise ValueError("--point-size must be positive.")
    if not 0 < args.point_alpha <= 1:
        raise ValueError("--point-alpha must be in (0, 1].")
def load_evaluation_npz(
    args: argparse.Namespace,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Load profiles, counts, coordinates, and region-label fields."""

    require_file(args.npz, "Evaluation NPZ")
    required = {
        args.observed_key,
        args.predicted_key,
        args.chrom_key,
        args.start_key,
        args.end_key,
        args.strand_key,
        args.sample_id_key,
    }
    with np.load(args.npz, allow_pickle=False) as data:
        keys = set(data.files)
        missing = sorted(required.difference(keys))
        if missing:
            raise KeyError(
                f"{args.npz} is missing required key(s): {', '.join(missing)}"
            )
        observed = np.asarray(data[args.observed_key], dtype=np.float32)
        predicted = np.asarray(data[args.predicted_key], dtype=np.float32)
        chroms = np.asarray(data[args.chrom_key]).astype(str)
        starts = np.asarray(data[args.start_key], dtype=np.int64)
        ends = np.asarray(data[args.end_key], dtype=np.int64)
        strands = np.asarray(data[args.strand_key]).astype(str)
        sample_ids = np.asarray(data[args.sample_id_key]).astype(str)
        if args.region_key in keys:
            region_types = np.asarray(data[args.region_key]).astype(str)
        else:
            region_types = np.full(len(observed), "", dtype="<U1")
        if args.mask_key in keys:
            masks = np.asarray(data[args.mask_key], dtype=bool)
        else:
            masks = np.ones(len(observed), dtype=bool)
        if args.count_key in keys:
            counts = np.asarray(data[args.count_key], dtype=np.float64)
        else:
            counts = np.sum(observed, axis=1, dtype=np.float64)

    if observed.ndim != 2 or predicted.shape != observed.shape:
        raise ValueError(
            "Observed and predicted profiles must have the same 2D shape; "
            f"got {observed.shape} and {predicted.shape}."
        )
    n_loci, profile_length = observed.shape
    expected = (n_loci,)
    for label, values in (
        (args.chrom_key, chroms),
        (args.start_key, starts),
        (args.end_key, ends),
        (args.strand_key, strands),
        (args.sample_id_key, sample_ids),
        (args.region_key, region_types),
        (args.mask_key, masks),
        (args.count_key, counts),
    ):
        if values.shape != expected:
            raise ValueError(
                f"{label!r} has shape {values.shape}; expected {expected}."
            )
    if np.any((ends - starts) != profile_length):
        raise ValueError(
            "Manifest output coordinates do not match the profile length."
        )
    if np.any(starts < 0) or np.any(ends <= starts):
        raise ValueError("Manifest contains invalid output coordinates.")
    invalid_strands = sorted(set(strands).difference({"+", "-"}))
    if invalid_strands:
        raise ValueError(f"Invalid strand value(s): {invalid_strands}.")
    if not np.all(np.isfinite(counts)) or np.any(counts < 0):
        raise ValueError("Observed counts contain non-finite or negative values.")
    return (
        observed,
        predicted,
        masks,
        counts,
        chroms,
        starts,
        ends,
        strands,
        sample_ids,
        region_types,
    )


def canonical_anchor(value: str) -> str | None:
    token = value.strip().upper().replace(" ", "").replace("_", "")
    mapping = {
        "TSS": "TSS",
        "TES": "TES",
        "5SS": "5SS",
        "5'SPLICESITE": "5SS",
        "5PRIMESPLICESITE": "5SS",
        "DONOR": "5SS",
        "3SS": "3SS",
        "3'SPLICESITE": "3SS",
        "3PRIMESPLICESITE": "3SS",
        "ACCEPTOR": "3SS",
    }
    return mapping.get(token)


def infer_anchor_types(
    region_types: np.ndarray,
    sample_ids: np.ndarray,
) -> np.ndarray:
    """Infer one canonical anchor label for every NPZ row."""

    labels: list[str] = []
    for region_type, sample_id in zip(region_types, sample_ids):
        anchor = canonical_anchor(str(region_type))
        if anchor is None:
            for token in str(sample_id).split("|"):
                anchor = canonical_anchor(token)
                if anchor is not None:
                    break
        labels.append(anchor or "")
    return np.asarray(labels, dtype="<U3")


def merged_interval_groups(
    records: list[tuple[int, int, int]],
) -> Iterable[tuple[int, int, list[tuple[int, int, int]]]]:
    """Yield merged genomic intervals and their member NPZ rows."""

    if not records:
        return
    ordered = sorted(records, key=lambda value: (value[0], value[1], value[2]))
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
    chrom: str,
    start: int,
    end: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recover integer counts and split them into complementary halves."""

    signal = np.asarray(signal, dtype=np.float64)
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    signal = np.abs(signal)
    counts_float = signal / signal_unit
    rounded = np.rint(counts_float)
    if not np.allclose(counts_float, rounded, rtol=0.0, atol=5e-3):
        difference = float(np.max(np.abs(counts_float - rounded)))
        raise ValueError(
            "The bigWig signal is not an integer multiple of its "
            f"--signal-unit ({signal_unit:.12g}) at "
            f"{chrom}:{start}-{end}; maximum count-space error is "
            f"{difference:.6g}."
        )
    counts = rounded.astype(np.int64, copy=False)
    pseudorep1_counts = rng.binomial(counts, 0.5)
    pseudorep2_counts = counts - pseudorep1_counts
    raw = (counts * signal_unit).astype(np.float32)
    pseudorep1 = (pseudorep1_counts * signal_unit).astype(np.float32)
    pseudorep2 = (pseudorep2_counts * signal_unit).astype(np.float32)
    return raw, pseudorep1, pseudorep2


def build_pseudoreplicate_profiles(
    pos_bw_paths: list[Path],
    neg_bw_paths: list[Path],
    signal_units: list[float],
    chroms: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    strands: np.ndarray,
    observed_profiles: np.ndarray,
    seed: int,
    verify_observed: bool,
    observed_atol: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Create reproducible, strand-oriented count pseudoreplicates."""

    for index, (pos_path, neg_path) in enumerate(
        zip(pos_bw_paths, neg_bw_paths),
        start=1,
    ):
        require_file(pos_path, f"Positive-strand bigWig {index}")
        require_file(neg_path, f"Negative-strand bigWig {index}")

    n_loci, profile_length = observed_profiles.shape
    pseudorep1 = np.zeros((n_loci, profile_length), dtype=np.float32)
    pseudorep2 = np.zeros((n_loci, profile_length), dtype=np.float32)
    assigned = np.zeros(n_loci, dtype=bool)
    rng = np.random.default_rng(seed)
    n_merged_regions = 0

    handle_pairs = [
        (pyBigWig.open(str(pos_path)), pyBigWig.open(str(neg_path)))
        for pos_path, neg_path in zip(pos_bw_paths, neg_bw_paths)
    ]
    if any(
        pos_handle is None or neg_handle is None
        for pos_handle, neg_handle in handle_pairs
    ):
        raise OSError("Could not open one or more bigWig files.")

    try:
        for strand in ("+", "-"):
            strand_handles = [
                pos_handle if strand == "+" else neg_handle
                for pos_handle, neg_handle in handle_pairs
            ]
            chrom_sizes = [handle.chroms() for handle in strand_handles]
            records_by_chrom: dict[
                str,
                list[tuple[int, int, int]],
            ] = defaultdict(list)
            for index in np.flatnonzero(strands == strand):
                chrom = str(chroms[index])
                start = int(starts[index])
                end = int(ends[index])
                for source_index, sizes in enumerate(chrom_sizes, start=1):
                    if chrom not in sizes:
                        raise KeyError(
                            f"Chromosome {chrom!r} is absent from bigWig pair "
                            f"{source_index} ({strand} strand)."
                        )
                    if end > int(sizes[chrom]):
                        raise ValueError(
                            f"Window {chrom}:{start}-{end} exceeds bigWig pair "
                            f"{source_index} chromosome length {sizes[chrom]}."
                        )
                records_by_chrom[chrom].append((start, end, int(index)))

            for chrom in sorted(records_by_chrom):
                for region_start, region_end, members in merged_interval_groups(
                    records_by_chrom[chrom]
                ):
                    n_merged_regions += 1
                    region_length = region_end - region_start
                    raw = np.zeros(region_length, dtype=np.float32)
                    split1 = np.zeros(region_length, dtype=np.float32)
                    split2 = np.zeros(region_length, dtype=np.float32)
                    for handle, signal_unit in zip(
                        strand_handles,
                        signal_units,
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
                                chrom,
                                region_start,
                                region_end,
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
                            raise RuntimeError(
                                f"Unexpected extracted length at row {index}."
                            )
                        if verify_observed and not np.allclose(
                            raw_profile,
                            observed_profiles[index],
                            rtol=1e-5,
                            atol=observed_atol,
                        ):
                            max_difference = float(
                                np.max(
                                    np.abs(
                                        raw_profile
                                        - observed_profiles[index]
                                    )
                                )
                            )
                            raise ValueError(
                                "The supplied bigWigs do not reproduce "
                                f"observed_profiles at row {index} "
                                f"({chrom}:{start}-{end}, strand {strand}); "
                                f"maximum absolute difference is "
                                f"{max_difference:.6g}."
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
        raise RuntimeError(
            f"Failed to assign {len(missing)} rows; "
            f"first missing row: {int(missing[0])}."
        )
    return pseudorep1, pseudorep2, n_merged_regions


def bin_profiles(profiles: np.ndarray, resolution: int) -> np.ndarray:
    profiles = np.asarray(profiles, dtype=np.float64)
    if profiles.ndim != 2:
        raise ValueError(f"Profiles must be 2D, got {profiles.shape}.")
    if resolution > profiles.shape[1]:
        raise ValueError(
            f"Resolution {resolution} exceeds profile length {profiles.shape[1]}."
        )
    if resolution == 1:
        return profiles
    usable = profiles.shape[1] // resolution * resolution
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


def calculate_distances(
    observed: np.ndarray,
    predicted: np.ndarray,
    pseudorep1: np.ndarray,
    pseudorep2: np.ndarray,
    masks: np.ndarray,
    resolution: int,
    chunk_size: int,
    random_seed: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Calculate three JS-distance arrays on one common set of valid rows."""

    row_parts: list[np.ndarray] = []
    parts: dict[str, list[np.ndarray]] = {
        "random": [],
        "model": [],
        "pseudoreplicate": [],
    }
    random_generator = np.random.default_rng(random_seed)

    for start in range(0, len(observed), chunk_size):
        end = min(start + chunk_size, len(observed))
        observed_chunk = bin_profiles(observed[start:end], resolution)
        predicted_chunk = bin_profiles(predicted[start:end], resolution)
        pseudorep1_chunk = bin_profiles(pseudorep1[start:end], resolution)
        pseudorep2_chunk = bin_profiles(pseudorep2[start:end], resolution)
        random_order = np.argsort(
            random_generator.random(observed_chunk.shape),
            axis=1,
        )
        random_chunk = np.take_along_axis(
            observed_chunk,
            random_order,
            axis=1,
        )
        valid = (
            np.asarray(masks[start:end], dtype=bool)
            & valid_profile_rows(
                observed_chunk,
                predicted_chunk,
                pseudorep1_chunk,
                pseudorep2_chunk,
                random_chunk,
            )
        )
        if not np.any(valid):
            continue

        row_parts.append(np.flatnonzero(valid) + start)
        parts["random"].append(
            jensenshannon(
                observed_chunk[valid],
                random_chunk[valid],
                base=2,
                axis=1,
            )
        )
        parts["model"].append(
            jensenshannon(
                observed_chunk[valid],
                predicted_chunk[valid],
                base=2,
                axis=1,
            )
        )
        parts["pseudoreplicate"].append(
            jensenshannon(
                pseudorep1_chunk[valid],
                pseudorep2_chunk[valid],
                base=2,
                axis=1,
            )
        )

    if not row_parts:
        raise ValueError(
            "No test windows are valid for all three JS-distance "
            "comparisons."
        )
    rows = np.concatenate(row_parts).astype(np.int64, copy=False)
    distances = {
        name: np.clip(
            np.asarray(np.concatenate(values), dtype=np.float64),
            0.0,
            1.0,
        )
        for name, values in parts.items()
    }
    return rows, distances


def normalize_profile_js_distance(
    model_js_distance: np.ndarray,
    pseudoreplicate_js_distance: np.ndarray,
    random_js_distance: np.ndarray,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize JS distance between pseudoreplicate and random baselines."""

    denominator = random_js_distance - pseudoreplicate_js_distance
    valid = (
        np.isfinite(model_js_distance)
        & np.isfinite(pseudoreplicate_js_distance)
        & np.isfinite(random_js_distance)
        & (np.abs(denominator) > epsilon)
    )
    normalized = np.divide(
        model_js_distance - pseudoreplicate_js_distance,
        denominator,
        out=np.full(
            model_js_distance.shape,
            np.nan,
            dtype=np.float64,
        ),
        where=valid,
    )
    normalized[valid] = np.clip(normalized[valid], 0.0, 1.0)
    baseline_order_valid = (
        pseudoreplicate_js_distance < random_js_distance
    )
    return normalized, valid, baseline_order_valid


def summarize_groups(
    labels: np.ndarray,
    values: np.ndarray,
) -> list[dict[str, float | int | str]]:
    summary: list[dict[str, float | int | str]] = []
    for anchor in ANCHOR_ORDER:
        group_values = values[labels == anchor]
        if not len(group_values):
            raise ValueError(f"No retained normalized values for {anchor}.")
        stats = boxplot_stats(group_values, whis=1.5)[0]
        summary.append(
            {
                "anchor_type": anchor,
                "n": int(len(group_values)),
                "mean": float(np.mean(group_values)),
                "median": float(stats["med"]),
                "q25": float(stats["q1"]),
                "q75": float(stats["q3"]),
                "whisker_low": float(stats["whislo"]),
                "whisker_high": float(stats["whishi"]),
            }
        )
    return summary


def apply_final_figure_small_style() -> None:
    """Apply the final-figure-small Matplotlib contract."""

    mpl.rcParams.update(
        {
            "font.size": TICK_FONTSIZE,
            "axes.labelsize": LABEL_FONTSIZE,
            "axes.titlesize": TITLE_FONTSIZE,
            "axes.titleweight": "normal",
            "xtick.labelsize": TICK_FONTSIZE,
            "ytick.labelsize": TICK_FONTSIZE,
            "legend.fontsize": LEGEND_FONTSIZE,
            "legend.frameon": False,
            "pdf.fonttype": 42,
        }
    )


def draw_distribution(
    ax: plt.Axes,
    values: np.ndarray,
    position: float,
    color: str,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> None:
    """Draw every retained value plus a centered horizontal boxplot."""

    jitter = rng.uniform(-0.22, 0.22, size=len(values))
    ax.scatter(
        values,
        np.full(len(values), position) + jitter,
        color=color,
        s=args.point_size,
        alpha=args.point_alpha,
        linewidths=0,
        rasterized=args.rasterize_points,
        zorder=1,
    )
    stats = boxplot_stats(values, whis=1.5)[0]
    ax.hlines(
        position,
        stats["whislo"],
        stats["whishi"],
        color=color,
        linewidth=0.8,
        zorder=3,
    )
    ax.hlines(
        position,
        stats["q1"],
        stats["q3"],
        color=color,
        linewidth=3.2,
        capstyle="butt",
        zorder=4,
    )
    ax.scatter(
        stats["med"],
        position,
        s=19,
        facecolor="white",
        edgecolor=color,
        linewidths=0.9,
        clip_on=False,
        zorder=5,
    )


def plot_pdf(
    labels: np.ndarray,
    values: np.ndarray,
    summary: list[dict[str, float | int | str]],
    args: argparse.Namespace,
    title: str | None,
) -> None:
    apply_final_figure_small_style()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    rng = np.random.default_rng(args.jitter_seed)
    y_positions = {
        anchor: len(ANCHOR_ORDER) - index - 1
        for index, anchor in enumerate(ANCHOR_ORDER)
    }

    if not args.hide_all_median:
        ax.axvline(
            float(np.median(values)),
            color="#333333",
            linewidth=0.8,
            linestyle=(0, (3, 2)),
            alpha=0.60,
            zorder=0,
        )

    for anchor in ANCHOR_ORDER:
        draw_distribution(
            ax,
            values[labels == anchor],
            y_positions[anchor],
            COLORS[anchor],
            rng,
            args,
        )

    n_by_anchor = {
        str(row["anchor_type"]): int(row["n"])
        for row in summary
    }
    tick_labels = [
        f"{DISPLAY_LABELS[anchor]}\n(n={n_by_anchor[anchor]:,})"
        for anchor in ANCHOR_ORDER
    ]
    ax.set_yticks(
        [y_positions[anchor] for anchor in ANCHOR_ORDER],
        tick_labels,
        fontsize=TICK_FONTSIZE,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.45, len(ANCHOR_ORDER) - 0.45)
    ax.set_xticks((0.0, 0.25, 0.50, 0.75, 1.0))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_xlabel(
        "Normalized JS Distance",
        fontsize=LABEL_FONTSIZE,
    )
    if title:
        ax.set_title(
            title,
            fontsize=TITLE_FONTSIZE,
            fontweight="normal",
            pad=6,
        )
    if args.panel_label:
        ax.text(
            -0.30,
            1.04,
            args.panel_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=PANEL_LABEL_FONTSIZE,
            fontweight="bold",
        )

    ax.tick_params(
        axis="x",
        labelsize=TICK_FONTSIZE,
        direction="out",
        length=3,
        pad=2,
    )
    ax.tick_params(
        axis="y",
        labelsize=TICK_FONTSIZE,
        length=0,
        pad=3,
    )
    fig.tight_layout()

    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_pdf, format="pdf")
    plt.close(fig)


def save_values_tsv(
    path: Path,
    rows: np.ndarray,
    keep: np.ndarray,
    labels: np.ndarray,
    sample_ids: np.ndarray,
    chroms: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    strands: np.ndarray,
    counts: np.ndarray,
    distances: dict[str, np.ndarray],
    normalized_js_distance: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "npz_row",
                "sample_id",
                "anchor_type",
                "chrom",
                "output_start",
                "output_end",
                "strand",
                "observed_count",
                "model_js_distance",
                "pseudoreplicate_js_distance",
                "random_profile_js_distance",
                "normalized_js_distance",
            ]
        )
        for position in np.flatnonzero(keep):
            row = int(rows[position])
            writer.writerow(
                [
                    row,
                    str(sample_ids[row]),
                    str(labels[position]),
                    str(chroms[row]),
                    int(starts[row]),
                    int(ends[row]),
                    str(strands[row]),
                    f"{counts[row]:.9g}",
                    f"{distances['model'][position]:.9g}",
                    f"{distances['pseudoreplicate'][position]:.9g}",
                    f"{distances['random'][position]:.9g}",
                    f"{normalized_js_distance[position]:.9g}",
                ]
            )


def save_summary_tsv(
    path: Path,
    summary: list[dict[str, float | int | str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(summary[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(summary)


def main() -> None:
    args = parse_args()
    validate_args(args)
    (
        observed,
        predicted,
        masks,
        counts,
        chroms,
        starts,
        ends,
        strands,
        sample_ids,
        region_types,
    ) = load_evaluation_npz(args)
    anchor_types = infer_anchor_types(region_types, sample_ids)
    raw_anchor_counts = {
        anchor: int(np.sum(anchor_types == anchor))
        for anchor in ANCHOR_ORDER
    }
    absent = [
        anchor
        for anchor, count in raw_anchor_counts.items()
        if count == 0
    ]
    if absent:
        raise ValueError(
            "Could not infer anchor types for: " + ", ".join(absent)
        )

    pseudorep1, pseudorep2, n_regions = build_pseudoreplicate_profiles(
        args.pos_bw,
        args.neg_bw,
        args.signal_unit,
        chroms,
        starts,
        ends,
        strands,
        observed,
        seed=args.seed,
        verify_observed=not args.skip_observed_check,
        observed_atol=args.observed_atol,
    )
    rows, distances = calculate_distances(
        observed,
        predicted,
        pseudorep1,
        pseudorep2,
        masks,
        args.resolution,
        args.chunk_size,
        args.random_seed,
    )
    normalized_js_distance, normalization_valid, baseline_order_valid = (
        normalize_profile_js_distance(
            distances["model"],
            distances["pseudoreplicate"],
            distances["random"],
            args.normalization_epsilon,
        )
    )
    row_labels = anchor_types[rows]
    keep = (
        normalization_valid
        & np.isin(row_labels, ANCHOR_ORDER)
        & (counts[rows] >= args.count_threshold)
    )
    if args.require_ordered_baselines:
        keep &= baseline_order_valid
    if not np.any(keep):
        raise ValueError("No rows remain after normalization and filtering.")

    retained_labels = row_labels[keep]
    retained_values = normalized_js_distance[keep]
    summary = summarize_groups(retained_labels, retained_values)

    dataset = args.npz.parent.name
    if args.title is not None:
        title = args.title
    elif args.show_title:
        title = DISPLAY_TITLES.get(dataset, dataset.replace("_", " "))
    else:
        title = None
    plot_pdf(retained_labels, retained_values, summary, args, title)

    if args.values_tsv is not None:
        save_values_tsv(
            args.values_tsv,
            rows,
            keep,
            row_labels,
            sample_ids,
            chroms,
            starts,
            ends,
            strands,
            counts,
            distances,
            normalized_js_distance,
        )
    if args.summary_tsv is not None:
        save_summary_tsv(args.summary_tsv, summary)

    print(f"[{dataset}] normalized per-window JS distance")
    print("  raw metric: scipy Jensen-Shannon distance, base=2")
    print(f"  resolution: {args.resolution} bp")
    print(f"  count threshold: observed_counts >= {args.count_threshold:g}")
    print("  plot value: normalized JS distance (lower is better)")
    print(f"  pseudoreplicate merged regions split once: {n_regions:,}")
    print(
        "  random lower bound: independent within-window permutation of "
        f"binned measured values, seed={args.random_seed}"
    )
    print(
        f"  normalization-valid common windows: "
        f"{int(np.sum(normalization_valid)):,}"
    )
    if args.require_ordered_baselines:
        print(
            "  required baseline order: pseudoreplicate JS distance "
            "< random-profile JS distance"
        )
    for row in summary:
        print(
            f"  {row['anchor_type']}: n={int(row['n']):,}, "
            f"median={float(row['median']):.4f}, "
            f"mean={float(row['mean']):.4f}"
        )
    print(f"  wrote {args.output_pdf}")
    if args.values_tsv is not None:
        print(f"  wrote {args.values_tsv}")
    if args.summary_tsv is not None:
        print(f"  wrote {args.summary_tsv}")


if __name__ == "__main__":
    main()
