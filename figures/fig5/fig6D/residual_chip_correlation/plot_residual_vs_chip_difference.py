#!/usr/bin/env python3
"""Relate cell-specific pausing residuals to cell-specific ChIP signal.

For matched genomic windows in cell A and cell B, this program calculates:

    residual_cell = log(observed + pseudocount)
                    - log(sequence_prediction + pseudocount)
    delta_residual = residual_A - residual_B
    delta_chip = z(log1p(mean_chip_A)) - z(log1p(mean_chip_B))

The resulting paired values are shown with HiLearn's density-colored
correlation plot, a Pearson correlation, and an optional linear fit.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyBigWig  # noqa: E402
from hilearn.plot import corr_plot  # noqa: E402
from scipy.stats import linregress, pearsonr  # noqa: E402


HG19_CHR1_LENGTH = 249_250_621
HG38_CHR1_LENGTH = 248_956_422
MATCH_FIELDS = (
    "sample_id",
    "chrom",
    "start",
    "end",
    "strand",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the difference in sequence-only pausing residual against "
            "the difference in standardized log1p ChIP signal for two cells."
        )
    )
    parser.add_argument("--cell-a-npz", required=True)
    parser.add_argument("--cell-b-npz", required=True)
    parser.add_argument("--cell-a-bigwig", required=True)
    parser.add_argument("--cell-b-bigwig", required=True)
    parser.add_argument("--cell-a-label", required=True)
    parser.add_argument("--cell-b-label", required=True)
    parser.add_argument(
        "--signal-name",
        required=True,
        help="Signal name used in the title and x-axis label, e.g. CTCF.",
    )
    parser.add_argument("--output-pdf", required=True)
    parser.add_argument(
        "--output-tsv",
        default=None,
        help="Optional table containing all matched values used in the plot.",
    )
    parser.add_argument(
        "--fit",
        choices=("linear", "none"),
        default="linear",
    )
    parser.add_argument(
        "--pseudocount",
        type=float,
        default=1.0,
        help="Added to observed and predicted counts before log (default: 1).",
    )
    parser.add_argument(
        "--max-plot-points",
        type=int,
        default=10_000,
        help=(
            "Maximum number of points drawn by HiLearn; Pearson R is still "
            "calculated from all valid windows (default: 10000)."
        ),
    )
    parser.add_argument("--scatter-size", type=float, default=5.0)
    parser.add_argument("--scatter-alpha", type=float, default=0.8)
    parser.add_argument(
        "--density-color-rate",
        type=float,
        default=10.0,
        help="HiLearn density color-rate parameter (default: 10).",
    )
    parser.add_argument(
        "--expected-assembly",
        choices=("hg19", "hg38", "none"),
        default="hg19",
    )
    parser.add_argument(
        "--allow-partial-match",
        action="store_true",
        help="Use the sample-ID intersection instead of requiring identical sets.",
    )
    parser.add_argument("--title", default=None)
    return parser.parse_args()


def apply_figure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "axes.linewidth": 1.0,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.major.size": 4.0,
            "ytick.major.size": 4.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def require_array(
    data: np.lib.npyio.NpzFile,
    key: str,
    expected_length: int | None = None,
) -> np.ndarray:
    if key not in data.files:
        raise KeyError(
            f"Missing NPZ key {key!r}. Available keys: {', '.join(data.files)}"
        )
    values = np.asarray(data[key])
    if expected_length is not None and len(values) != expected_length:
        raise ValueError(
            f"NPZ key {key!r} has length {len(values):,}; "
            f"expected {expected_length:,}."
        )
    return values


def load_evaluation(npz_path: Path, label: str) -> dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as data:
        observed = require_array(data, "observed_counts").astype(
            np.float64, copy=False
        )
        n_windows = len(observed)
        arrays = {
            "observed": observed,
            "predicted": require_array(
                data, "predicted_counts", n_windows
            ).astype(np.float64, copy=False),
            "sample_id": require_array(
                data, "manifest_sample_id", n_windows
            ).astype(str),
            "chrom": require_array(data, "manifest_chrom", n_windows).astype(
                str
            ),
            "start": require_array(
                data, "manifest_output_start", n_windows
            ).astype(np.int64, copy=False),
            "end": require_array(
                data, "manifest_output_end", n_windows
            ).astype(np.int64, copy=False),
            "strand": require_array(
                data, "manifest_strand", n_windows
            ).astype(str),
        }

    if n_windows == 0:
        raise ValueError(f"No windows found in {npz_path}.")
    if len(set(arrays["sample_id"])) != n_windows:
        raise ValueError(f"{label} manifest_sample_id values are not unique.")

    for key in ("observed", "predicted"):
        values = arrays[key]
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{label} {key} contains non-finite values.")
        if np.any(values < 0):
            raise ValueError(f"{label} {key} contains negative values.")

    lengths = arrays["end"] - arrays["start"]
    if np.any(lengths <= 0):
        raise ValueError(f"{label} contains non-positive output intervals.")
    print(
        f"Loaded {label}: {n_windows:,} windows from {npz_path}; "
        f"window length(s): {np.unique(lengths).tolist()} bp"
    )
    return arrays


def subset_arrays(
    arrays: dict[str, np.ndarray], indices: np.ndarray
) -> dict[str, np.ndarray]:
    return {key: values[indices] for key, values in arrays.items()}


def match_evaluations(
    cell_a: dict[str, np.ndarray],
    cell_b: dict[str, np.ndarray],
    label_a: str,
    label_b: str,
    allow_partial: bool,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    ids_a = cell_a["sample_id"]
    ids_b = cell_b["sample_id"]
    set_a = set(ids_a)
    set_b = set(ids_b)
    common = set_a & set_b
    if not common:
        raise ValueError("The two NPZ files have no shared manifest_sample_id values.")

    missing_a = set_b - set_a
    missing_b = set_a - set_b
    if (missing_a or missing_b) and not allow_partial:
        raise ValueError(
            "The two NPZ files do not contain identical sample-ID sets: "
            f"{len(missing_b):,} only in {label_a}, "
            f"{len(missing_a):,} only in {label_b}. "
            "Use --allow-partial-match to analyze the intersection."
        )

    lookup_b = {sample_id: index for index, sample_id in enumerate(ids_b)}
    indices_a = np.asarray(
        [index for index, sample_id in enumerate(ids_a) if sample_id in common],
        dtype=np.int64,
    )
    indices_b = np.asarray(
        [lookup_b[ids_a[index]] for index in indices_a], dtype=np.int64
    )
    matched_a = subset_arrays(cell_a, indices_a)
    matched_b = subset_arrays(cell_b, indices_b)

    for field in ("sample_id", "chrom", "start", "end", "strand"):
        disagreement = matched_a[field] != matched_b[field]
        if np.any(disagreement):
            first = int(np.flatnonzero(disagreement)[0])
            raise ValueError(
                f"Matched rows disagree in {field!r} at row {first:,}: "
                f"{matched_a[field][first]!r} versus "
                f"{matched_b[field][first]!r}."
            )

    print(
        f"Matched {len(indices_a):,} windows by manifest_sample_id; "
        f"{len(missing_b):,} {label_a}-only and "
        f"{len(missing_a):,} {label_b}-only windows."
    )
    return matched_a, matched_b


def validate_bigwig_assembly(
    chrom_sizes: dict[str, int], expected_assembly: str, path: Path
) -> None:
    chr1_length = chrom_sizes.get("chr1")
    if chr1_length is None:
        raise ValueError(f"BigWig lacks chr1: {path}")
    detected = {
        HG19_CHR1_LENGTH: "hg19",
        HG38_CHR1_LENGTH: "hg38",
    }.get(chr1_length, "unknown")
    print(f"BigWig {path.name}: chr1={chr1_length:,} ({detected})")
    if expected_assembly == "none":
        return
    expected_length = {
        "hg19": HG19_CHR1_LENGTH,
        "hg38": HG38_CHR1_LENGTH,
    }[expected_assembly]
    if chr1_length != expected_length:
        raise ValueError(
            f"Expected {expected_assembly} chr1 length {expected_length:,}, "
            f"but {path} has {chr1_length:,}."
        )


def read_mean_bigwig_signal(
    bigwig_path: Path,
    chroms: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    expected_assembly: str,
    label: str,
) -> np.ndarray:
    means = np.empty(len(chroms), dtype=np.float64)
    negative_positions = 0
    with pyBigWig.open(str(bigwig_path)) as bigwig:
        chrom_sizes = bigwig.chroms()
        validate_bigwig_assembly(chrom_sizes, expected_assembly, bigwig_path)
        missing = sorted(set(chroms) - set(chrom_sizes))
        if missing:
            raise ValueError(
                f"Chromosomes absent from {bigwig_path}: {', '.join(missing)}"
            )

        for index, (chrom, start, end) in enumerate(
            zip(chroms, starts, ends, strict=True)
        ):
            start_i = int(start)
            end_i = int(end)
            if start_i < 0 or end_i > chrom_sizes[chrom]:
                raise ValueError(
                    f"Window {chrom}:{start_i}-{end_i} is outside {bigwig_path}."
                )
            profile = np.asarray(
                bigwig.values(chrom, start_i, end_i, numpy=True),
                dtype=np.float64,
            )
            if len(profile) != end_i - start_i:
                raise RuntimeError(
                    f"Unexpected profile length at {chrom}:{start_i}-{end_i}."
                )
            profile = np.nan_to_num(
                profile, nan=0.0, posinf=0.0, neginf=0.0
            )
            negative_positions += int(np.sum(profile < 0))
            np.maximum(profile, 0.0, out=profile)
            means[index] = profile.mean(dtype=np.float64)
            if (index + 1) % 5_000 == 0 or index + 1 == len(chroms):
                print(f"Read {label}: {index + 1:,}/{len(chroms):,} windows")

    if negative_positions:
        print(
            f"Warning: clipped {negative_positions:,} negative positions "
            f"to zero in {label}."
        )
    if not np.all(np.isfinite(means)):
        raise ValueError(f"Non-finite window means calculated for {label}.")
    return means


def standardize(values: np.ndarray, label: str) -> tuple[np.ndarray, float, float]:
    mean = float(values.mean(dtype=np.float64))
    std = float(values.std(dtype=np.float64, ddof=0))
    if not math.isfinite(std) or std <= 0:
        raise ValueError(f"Cannot z-score constant signal: {label}.")
    return (values - mean) / std, mean, std


def write_values_tsv(
    output_path: Path,
    cell_a: dict[str, np.ndarray],
    cell_b: dict[str, np.ndarray],
    residual_a: np.ndarray,
    residual_b: np.ndarray,
    chip_mean_a: np.ndarray,
    chip_mean_b: np.ndarray,
    chip_log_a: np.ndarray,
    chip_log_b: np.ndarray,
    chip_z_a: np.ndarray,
    chip_z_b: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "sample_id",
        "chrom",
        "output_start",
        "output_end",
        "strand",
        "observed_cell_a",
        "predicted_cell_a",
        "residual_cell_a",
        "observed_cell_b",
        "predicted_cell_b",
        "residual_cell_b",
        "delta_residual_a_minus_b",
        "chip_mean_cell_a",
        "chip_log1p_cell_a",
        "chip_z_cell_a",
        "chip_mean_cell_b",
        "chip_log1p_cell_b",
        "chip_z_cell_b",
        "delta_chip_z_a_minus_b",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        for index in range(len(residual_a)):
            writer.writerow(
                [
                    cell_a["sample_id"][index],
                    cell_a["chrom"][index],
                    int(cell_a["start"][index]),
                    int(cell_a["end"][index]),
                    cell_a["strand"][index],
                    f"{cell_a['observed'][index]:.10g}",
                    f"{cell_a['predicted'][index]:.10g}",
                    f"{residual_a[index]:.10g}",
                    f"{cell_b['observed'][index]:.10g}",
                    f"{cell_b['predicted'][index]:.10g}",
                    f"{residual_b[index]:.10g}",
                    f"{residual_a[index] - residual_b[index]:.10g}",
                    f"{chip_mean_a[index]:.10g}",
                    f"{chip_log_a[index]:.10g}",
                    f"{chip_z_a[index]:.10g}",
                    f"{chip_mean_b[index]:.10g}",
                    f"{chip_log_b[index]:.10g}",
                    f"{chip_z_b[index]:.10g}",
                    f"{chip_z_a[index] - chip_z_b[index]:.10g}",
                ]
            )
    print(f"Wrote values: {output_path}")


def plot_difference(
    delta_chip: np.ndarray,
    delta_residual: np.ndarray,
    pearson_r: float,
    output_pdf: Path,
    args: argparse.Namespace,
) -> None:
    apply_figure_style()
    fig, axis = plt.subplots(figsize=(5.0, 4.0))

    plt.sca(axis)
    corr_plot(
        delta_chip,
        delta_residual,
        max_num=args.max_plot_points,
        outlier=0.0,
        line_on=False,
        legend_on=False,
        size=args.scatter_size,
        dot_color=None,
        alpha=args.scatter_alpha,
        color_rate=args.density_color_rate,
    )
    for collection in axis.collections:
        collection.set_rasterized(True)

    if args.fit == "linear":
        fit = linregress(delta_chip, delta_residual)
        x_line = np.asarray(
            np.quantile(delta_chip, [0.005, 0.995]), dtype=np.float64
        )
        y_line = fit.intercept + fit.slope * x_line
        axis.plot(
            x_line,
            y_line,
            color="#222222",
            linewidth=1.4,
            linestyle=(0, (4, 3)),
            zorder=4,
        )

    difference_label = f"{args.cell_a_label} - {args.cell_b_label}"
    axis.set_xlabel(
        f"Δ z[log1p(mean {args.signal_name})]\n({difference_label})"
    )
    axis.set_ylabel(
        "Δ pausing residual\n"
        f"({difference_label})"
    )
    axis.set_title(args.title or args.signal_name, pad=8)
    axis.text(
        0.03,
        0.97,
        f"Pearson R = {pearson_r:.3f}\n"
        f"n = {len(delta_chip):,}",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.5},
    )
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(False)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.24, right=0.96, bottom=0.21, top=0.89)
    fig.savefig(output_pdf, format="pdf")
    plt.close(fig)
    print(f"Wrote figure: {output_pdf}")


def validate_args(args: argparse.Namespace) -> None:
    if args.pseudocount <= 0:
        raise ValueError("--pseudocount must be greater than zero.")
    if args.max_plot_points < 100:
        raise ValueError("--max-plot-points must be at least 100.")
    if args.scatter_size <= 0:
        raise ValueError("--scatter-size must be greater than zero.")
    if not 0 < args.scatter_alpha <= 1:
        raise ValueError("--scatter-alpha must be in (0, 1].")
    if args.density_color_rate <= 0:
        raise ValueError("--density-color-rate must be greater than zero.")


def main() -> None:
    args = parse_args()
    validate_args(args)

    cell_a_npz = Path(args.cell_a_npz).expanduser().resolve()
    cell_b_npz = Path(args.cell_b_npz).expanduser().resolve()
    cell_a_bigwig = Path(args.cell_a_bigwig).expanduser().resolve()
    cell_b_bigwig = Path(args.cell_b_bigwig).expanduser().resolve()
    output_pdf = Path(args.output_pdf).expanduser().resolve()

    for path in (cell_a_npz, cell_b_npz, cell_a_bigwig, cell_b_bigwig):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_pdf.suffix.lower() != ".pdf":
        raise ValueError("--output-pdf must end in .pdf")

    cell_a = load_evaluation(cell_a_npz, args.cell_a_label)
    cell_b = load_evaluation(cell_b_npz, args.cell_b_label)
    cell_a, cell_b = match_evaluations(
        cell_a,
        cell_b,
        args.cell_a_label,
        args.cell_b_label,
        args.allow_partial_match,
    )

    chip_mean_a = read_mean_bigwig_signal(
        cell_a_bigwig,
        cell_a["chrom"],
        cell_a["start"],
        cell_a["end"],
        args.expected_assembly,
        f"{args.cell_a_label} {args.signal_name}",
    )
    chip_mean_b = read_mean_bigwig_signal(
        cell_b_bigwig,
        cell_b["chrom"],
        cell_b["start"],
        cell_b["end"],
        args.expected_assembly,
        f"{args.cell_b_label} {args.signal_name}",
    )

    chip_log_a = np.log1p(chip_mean_a)
    chip_log_b = np.log1p(chip_mean_b)
    chip_z_a, mean_a, std_a = standardize(
        chip_log_a, f"{args.cell_a_label} {args.signal_name}"
    )
    chip_z_b, mean_b, std_b = standardize(
        chip_log_b, f"{args.cell_b_label} {args.signal_name}"
    )

    residual_a = np.log(cell_a["observed"] + args.pseudocount) - np.log(
        cell_a["predicted"] + args.pseudocount
    )
    residual_b = np.log(cell_b["observed"] + args.pseudocount) - np.log(
        cell_b["predicted"] + args.pseudocount
    )
    delta_residual = residual_a - residual_b
    delta_chip = chip_z_a - chip_z_b

    finite = np.isfinite(delta_residual) & np.isfinite(delta_chip)
    if np.sum(finite) < 3:
        raise ValueError("Fewer than three finite matched observations remain.")
    if not np.all(finite):
        print(f"Dropping {np.sum(~finite):,} non-finite matched rows.")
        cell_a = subset_arrays(cell_a, np.flatnonzero(finite))
        cell_b = subset_arrays(cell_b, np.flatnonzero(finite))
        residual_a = residual_a[finite]
        residual_b = residual_b[finite]
        chip_mean_a = chip_mean_a[finite]
        chip_mean_b = chip_mean_b[finite]
        chip_log_a = chip_log_a[finite]
        chip_log_b = chip_log_b[finite]
        chip_z_a = chip_z_a[finite]
        chip_z_b = chip_z_b[finite]
        delta_residual = delta_residual[finite]
        delta_chip = delta_chip[finite]

    pearson_result = pearsonr(delta_chip, delta_residual)
    pearson_r = float(pearson_result.statistic)
    if not math.isfinite(pearson_r):
        raise ValueError("Pearson correlation is undefined.")

    print(
        f"{args.cell_a_label} log1p ChIP mean/std: {mean_a:.6g}/{std_a:.6g}\n"
        f"{args.cell_b_label} log1p ChIP mean/std: {mean_b:.6g}/{std_b:.6g}\n"
        f"Pearson R={pearson_r:.6g}, n={len(delta_chip):,}"
    )

    if args.output_tsv:
        write_values_tsv(
            Path(args.output_tsv).expanduser().resolve(),
            cell_a,
            cell_b,
            residual_a,
            residual_b,
            chip_mean_a,
            chip_mean_b,
            chip_log_a,
            chip_log_b,
            chip_z_a,
            chip_z_b,
        )

    plot_difference(
        delta_chip,
        delta_residual,
        pearson_r,
        output_pdf,
        args,
    )


if __name__ == "__main__":
    main()
