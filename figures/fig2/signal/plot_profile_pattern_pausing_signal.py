#!/usr/bin/env python3
"""Plot non-template G4Hunter and pausing around a TF-MoDISco pattern.

Generic over both TF-MoDISco tasks -- pick the pattern with CLI args:

    # profile head patterns (only pos_patterns exist)
    python plot_profile_pattern_pausing_signal.py --task profile --pattern-index 6
    # count head patterns (pos_patterns and neg_patterns)
    python plot_profile_pattern_pausing_signal.py --task count --sign pos --pattern-index 3
    python plot_profile_pattern_pausing_signal.py --task count --sign neg --pattern-index 5

TF-MoDISco ``example_idx`` values index the examples supplied to modisco-lite,
not the complete test set.  This script therefore maps them through
``<task>_selected_dataset_indices.npy`` before reading ``test_profiles.npz``.

The TF-MoDISco input was the centered 1,000-bp crop of the 2,114-bp PauseNet
input, which is the same interval as the model's 1,000-bp profile output.
Seqlet start/end coordinates can consequently be used directly on the stored
observed and predicted profiles.

By default, profiles remain in their stored transcription-oriented direction.
The ``is_revcomp`` flag records how TF-MoDISco aligned the sequence logo and is
not used to flip pausing profiles unless ``--orientation motif`` is requested.

For every selected example, the script also restores the centered 1,000-bp
non-template DNA sequence used by TF-MoDISco.  Signed G4Hunter scores are
calculated in a centered sliding window at every position (G-rich positive,
C-rich negative) and averaged across seqlets after motif-center alignment.

Default outputs (this folder):
  profile_pattern_<N>_observed_predicted_signal.pdf        (task=profile)
  count_<pos|neg>_pattern_<N>_observed_predicted_signal.pdf (task=count)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
RES = Path(
    "/mnt/HDD8TB/houruiyan/pausing_site/3_model_explanation/TFMoDISco/"
    "results/hek293t_netseq"
)
DEFAULT_NPZ = Path(
    "/mnt/HDD8TB/houruiyan/pausing_site/2_train_model/evaluate/"
    "hek293t_netseq/test_profiles.npz"
)
DEFAULT_SEQUENCE_CODES = Path(
    "/mnt/HDD8TB/houruiyan/pausing_site/data/HEK293T_NETseq/"
    "dataset/test/sequence_codes.npy"
)

G4_COLOR = "#008837"
OBSERVED_COLOR = "#B2182B"
PREDICTED_COLOR = "#2C7FB8"
FIGURE_SIZE = (10, 4)
PAUSING_YMAX = 1.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Center non-template G4Hunter scores and observed/predicted "
            "PauseNet profiles on TF-MoDISco seqlets."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--task",
        choices=("profile", "count"),
        default="profile",
        help="Which TF-MoDISco task the pattern comes from.",
    )
    parser.add_argument(
        "--sign",
        choices=("pos", "neg"),
        default="pos",
        help=(
            "pos_patterns or neg_patterns; only used for --task count "
            "(profile only has pos_patterns)."
        ),
    )
    parser.add_argument(
        "--pattern-index",
        type=int,
        required=True,
        help="TF-MoDISco pattern index, e.g. --pattern-index 6.",
    )
    parser.add_argument(
        "--tfmodisco-h5",
        type=Path,
        default=None,
        help="Override patterns h5 (default: RES/<task>/<task>_tfmodisco_patterns.h5).",
    )
    parser.add_argument(
        "--pattern-group",
        default=None,
        help=(
            "Override HDF5 group containing the pattern and its seqlets "
            "(default: <pos|neg>_patterns/pattern_<N> from --sign/--pattern-index)."
        ),
    )
    parser.add_argument(
        "--selected-dataset-indices",
        type=Path,
        default=None,
        help=(
            "Override dataset-row mapping saved by run_tfmodisco.py "
            "(default: RES/<task>/<task>_selected_dataset_indices.npy)."
        ),
    )
    parser.add_argument("--profiles-npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument(
        "--sequence-codes",
        type=Path,
        default=DEFAULT_SEQUENCE_CODES,
        help="Transcription-oriented PauseNet sequence_codes.npy.",
    )
    parser.add_argument(
        "--g4-window-size",
        type=int,
        default=25,
        help="Odd G4Hunter sliding-window size used for positional scores.",
    )
    parser.add_argument(
        "--flank",
        type=int,
        default=500,
        help="Bases displayed on each side of the seqlet center.",
    )
    parser.add_argument(
        "--orientation",
        choices=("transcription", "motif"),
        default="transcription",
        help=(
            "Keep the stored transcription direction, or reverse profiles for "
            "seqlets that TF-MoDISco aligned as reverse complements."
        ),
    )
    parser.add_argument(
        "--smooth-bp",
        type=int,
        default=1,
        help="Centered moving-average width; 1 draws the unsmoothed mean.",
    )
    parser.add_argument(
        "--min-coverage-fraction",
        type=float,
        default=0.25,
        help=(
            "Mask relative positions covered by less than this fraction of "
            "seqlets. Partial coverage occurs because stored profiles are 1 kb."
        ),
    )
    parser.add_argument(
        "--max-seqlets",
        type=int,
        default=None,
        help="Optional deterministic testing limit; uses the first N seqlets.",
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=None,
        help=(
            "Output PDF path (default: profile_pattern_<N>_observed_predicted_signal.pdf "
            "or count_<sign>_pattern_<N>_observed_predicted_signal.pdf in this folder)."
        ),
    )
    return parser.parse_args()


def resolve_task_inputs(args: argparse.Namespace) -> str:
    """Fill in task-dependent defaults; return a human-readable pattern label."""
    task_dir = RES / args.task
    if args.tfmodisco_h5 is None:
        args.tfmodisco_h5 = task_dir / f"{args.task}_tfmodisco_patterns.h5"
    if args.selected_dataset_indices is None:
        args.selected_dataset_indices = (
            task_dir / f"{args.task}_selected_dataset_indices.npy"
        )
    if args.pattern_group is None:
        group = "pos_patterns" if args.sign == "pos" else "neg_patterns"
        args.pattern_group = f"{group}/pattern_{args.pattern_index}"
    if args.task == "profile":
        label = f"profile pattern {args.pattern_index}"
        default_pdf = HERE / (
            f"profile_pattern_{args.pattern_index}_observed_predicted_signal.pdf"
        )
    else:
        label = f"count {args.sign} pattern {args.pattern_index}"
        default_pdf = HERE / (
            f"count_{args.sign}_pattern_{args.pattern_index}_observed_predicted_signal.pdf"
        )
    if args.output_pdf is None:
        args.output_pdf = default_pdf
    return label


def check_inputs(args: argparse.Namespace) -> None:
    for path in (
        args.tfmodisco_h5,
        args.selected_dataset_indices,
        args.profiles_npz,
        args.sequence_codes,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.flank <= 0:
        raise ValueError("--flank must be positive")
    if args.smooth_bp <= 0:
        raise ValueError("--smooth-bp must be positive")
    if args.g4_window_size <= 0 or args.g4_window_size % 2 == 0:
        raise ValueError("--g4-window-size must be a positive odd integer")
    if not 0 <= args.min_coverage_fraction <= 1:
        raise ValueError("--min-coverage-fraction must be between 0 and 1")
    if args.max_seqlets is not None and args.max_seqlets <= 0:
        raise ValueError("--max-seqlets must be positive")
    if args.output_pdf.suffix.lower() != ".pdf":
        raise ValueError("--output-pdf must end with .pdf")


def load_seqlet_mapping(
    h5_path: Path,
    pattern_group: str,
    selected_indices_path: Path,
    max_seqlets: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected_dataset_indices = np.asarray(
        np.load(selected_indices_path), dtype=np.int64
    )
    if selected_dataset_indices.ndim != 1:
        raise ValueError(
            "Selected dataset indices must be one-dimensional; received "
            f"{selected_dataset_indices.shape}"
        )

    with h5py.File(h5_path, "r") as h5:
        if pattern_group not in h5:
            parent = pattern_group.rstrip("/").rsplit("/", 1)[0]
            hint = ""
            if parent in h5:
                available = sorted(
                    k for k in h5[parent].keys() if k.startswith("pattern_")
                )
                hint = (
                    f"\navailable groups under {parent}: "
                    f"{', '.join(available) if available else '(none)'}"
                )
            else:
                hint = f"\navailable top-level groups: {list(h5.keys())}"
            raise KeyError(f"Missing HDF5 group: {pattern_group}{hint}")
        pattern = h5[pattern_group]
        if "seqlets" not in pattern:
            raise KeyError(f"{pattern_group} does not contain seqlets")
        seqlets = pattern["seqlets"]
        required = ("example_idx", "start", "end", "is_revcomp")
        missing = [key for key in required if key not in seqlets]
        if missing:
            raise KeyError(f"Missing seqlet arrays: {', '.join(missing)}")

        example_idx = np.asarray(seqlets["example_idx"], dtype=np.int64)
        starts = np.asarray(seqlets["start"], dtype=np.int64)
        ends = np.asarray(seqlets["end"], dtype=np.int64)
        is_revcomp = np.asarray(seqlets["is_revcomp"], dtype=bool)

    n_seqlets = len(example_idx)
    if not (len(starts) == len(ends) == len(is_revcomp) == n_seqlets):
        raise ValueError("TF-MoDISco seqlet arrays have inconsistent lengths")
    if n_seqlets == 0:
        raise ValueError(f"No seqlets found in {pattern_group}")
    if np.any(ends <= starts):
        raise ValueError("Seqlet end coordinates must be greater than starts")
    if np.any(example_idx < 0) or np.any(example_idx >= len(selected_dataset_indices)):
        raise IndexError(
            "A seqlet example_idx falls outside profile_selected_dataset_indices.npy"
        )

    if max_seqlets is not None:
        use = slice(0, min(max_seqlets, n_seqlets))
        example_idx = example_idx[use]
        starts = starts[use]
        ends = ends[use]
        is_revcomp = is_revcomp[use]

    dataset_indices = selected_dataset_indices[example_idx]
    centers = (starts + ends) // 2
    return dataset_indices, centers, starts, ends, is_revcomp


def load_profiles(
    npz_path: Path,
    dataset_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as data:
        required = (
            "observed_profiles",
            "predicted_profiles",
            "predicted_counts",
            "profile_masks",
        )
        missing = [key for key in required if key not in data.files]
        if missing:
            raise KeyError(f"NPZ is missing arrays: {', '.join(missing)}")

        n_examples = int(data["observed_profiles"].shape[0])
        if np.any(dataset_indices < 0) or np.any(dataset_indices >= n_examples):
            raise IndexError("Mapped dataset index falls outside test_profiles.npz")

        observed = np.asarray(
            data["observed_profiles"][dataset_indices], dtype=np.float64
        )
        predicted_probability = np.asarray(
            data["predicted_profiles"][dataset_indices], dtype=np.float64
        )
        predicted_counts = np.asarray(
            data["predicted_counts"][dataset_indices], dtype=np.float64
        )
        profile_masks = np.asarray(
            data["profile_masks"][dataset_indices], dtype=bool
        )

    if observed.ndim != 2 or predicted_probability.shape != observed.shape:
        raise ValueError(
            "Observed and predicted profiles must have matching (N,L) shapes; "
            f"received {observed.shape} and {predicted_probability.shape}"
        )
    if predicted_counts.shape != (len(observed),):
        raise ValueError("predicted_counts must have one value per profile")
    if not np.all(profile_masks):
        bad = int(np.sum(~profile_masks))
        raise ValueError(f"Pattern includes {bad} profiles masked during evaluation")
    if (
        not np.all(np.isfinite(observed))
        or not np.all(np.isfinite(predicted_probability))
        or not np.all(np.isfinite(predicted_counts))
    ):
        raise ValueError("Observed or predicted profiles contain NaN/Inf")
    if np.any(observed < 0) or np.any(predicted_probability < 0):
        raise ValueError("Observed or predicted profiles contain negative values")

    # PauseNet stores a normalized predicted profile and a separate total count.
    # Multiplication restores predicted counts per base, matching observed units.
    predicted = predicted_probability * predicted_counts[:, None]
    return observed, predicted, profile_masks


def load_non_template_sequence_codes(
    sequence_codes_path: Path,
    dataset_indices: np.ndarray,
    output_length: int,
) -> np.ndarray:
    """Load the same centered, transcription-oriented DNA used by TF-MoDISco."""
    sequence_codes = np.load(sequence_codes_path, mmap_mode="r")
    if sequence_codes.ndim != 2:
        raise ValueError(
            "sequence_codes.npy must have shape (N,L); received "
            f"{sequence_codes.shape}"
        )
    if np.any(dataset_indices < 0) or np.any(dataset_indices >= len(sequence_codes)):
        raise IndexError("Mapped dataset index falls outside sequence_codes.npy")
    input_length = int(sequence_codes.shape[1])
    if output_length > input_length:
        raise ValueError(
            f"Profile length {output_length} exceeds sequence length {input_length}"
        )
    crop_start = (input_length - output_length) // 2
    crop_end = crop_start + output_length
    cropped = np.asarray(
        sequence_codes[dataset_indices, crop_start:crop_end], dtype=np.uint8
    )
    if cropped.shape != (len(dataset_indices), output_length):
        raise ValueError(
            "Unexpected cropped sequence shape: "
            f"{cropped.shape}; expected {(len(dataset_indices), output_length)}"
        )
    if np.any(cropped > 3):
        bad = np.unique(cropped[cropped > 3]).tolist()
        raise ValueError(f"Sequence codes must be A/C/G/T=0/1/2/3; found {bad}")
    return cropped


def g4hunter_base_scores(sequence_codes: np.ndarray) -> np.ndarray:
    """Return signed G4Hunter per-base scores for A/C/G/T=0/1/2/3 codes."""
    codes = np.asarray(sequence_codes, dtype=np.uint8)
    if codes.ndim != 2:
        raise ValueError(f"Expected sequence codes (N,L), received {codes.shape}")
    scores = np.zeros(codes.shape, dtype=np.int8)
    for row in range(codes.shape[0]):
        position = 0
        while position < codes.shape[1]:
            base = int(codes[row, position])
            # C=1 receives a negative score; G=2 receives a positive score.
            if base not in (1, 2):
                position += 1
                continue
            end = position + 1
            while end < codes.shape[1] and int(codes[row, end]) == base:
                end += 1
            magnitude = min(end - position, 4)
            scores[row, position:end] = magnitude if base == 2 else -magnitude
            position = end
    return scores


def positional_g4hunter_scores(
    sequence_codes: np.ndarray,
    window_size: int,
) -> np.ndarray:
    """Assign each full G4Hunter window score to its center nucleotide."""
    if window_size > sequence_codes.shape[1]:
        raise ValueError("--g4-window-size exceeds the cropped sequence length")
    if window_size % 2 == 0:
        raise ValueError("G4Hunter positional windows require an odd size")
    base_scores = g4hunter_base_scores(sequence_codes).astype(np.float64)
    positional = np.full(base_scores.shape, np.nan, dtype=np.float64)
    half_window = window_size // 2
    kernel = np.ones(window_size, dtype=np.float64) / float(window_size)
    for row in range(base_scores.shape[0]):
        valid_scores = np.convolve(base_scores[row], kernel, mode="valid")
        positional[row, half_window : half_window + len(valid_scores)] = valid_scores
    return positional


def align_and_average(
    observed: np.ndarray,
    predicted: np.ndarray,
    g4_scores: np.ndarray,
    centers: np.ndarray,
    is_revcomp: np.ndarray,
    flank: int,
    orientation: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    n_seqlets, profile_length = observed.shape
    if g4_scores.shape != observed.shape:
        raise ValueError(
            f"G4/profile shape mismatch: {g4_scores.shape} vs {observed.shape}"
        )
    if len(centers) != n_seqlets or len(is_revcomp) != n_seqlets:
        raise ValueError("Profile and seqlet counts do not match")
    if np.any(centers < 0) or np.any(centers >= profile_length):
        raise IndexError(
            "A seqlet center falls outside the stored 1,000-bp output profile"
        )

    relative = np.arange(-flank, flank + 1, dtype=np.int64)
    observed_sum = np.zeros(len(relative), dtype=np.float64)
    predicted_sum = np.zeros(len(relative), dtype=np.float64)
    g4_sum = np.zeros(len(relative), dtype=np.float64)
    coverage = np.zeros(len(relative), dtype=np.int64)
    g4_coverage = np.zeros(len(relative), dtype=np.int64)

    for row, center in enumerate(centers):
        if orientation == "motif" and is_revcomp[row]:
            source_positions = center - relative
        else:
            source_positions = center + relative
        valid = (source_positions >= 0) & (source_positions < profile_length)
        source = source_positions[valid]
        observed_sum[valid] += observed[row, source]
        predicted_sum[valid] += predicted[row, source]
        coverage[valid] += 1
        g4_valid_values = np.isfinite(g4_scores[row, source])
        if np.any(g4_valid_values):
            destination = np.flatnonzero(valid)[g4_valid_values]
            g4_sum[destination] += g4_scores[row, source[g4_valid_values]]
            g4_coverage[destination] += 1

    if not np.all(coverage > 0):
        zero_positions = int(np.sum(coverage == 0))
        raise ValueError(
            f"No seqlet coverage at {zero_positions} relative positions; reduce --flank"
        )
    observed_mean = observed_sum / coverage
    predicted_mean = predicted_sum / coverage
    if not np.all(g4_coverage > 0):
        zero_positions = int(np.sum(g4_coverage == 0))
        raise ValueError(
            f"No valid G4Hunter windows at {zero_positions} relative positions"
        )
    g4_mean = g4_sum / g4_coverage
    return (
        relative,
        observed_mean,
        predicted_mean,
        g4_mean,
        coverage,
        g4_coverage,
    )


def moving_average(values: np.ndarray, width: int) -> np.ndarray:
    if width == 1:
        return values.copy()
    kernel = np.ones(width, dtype=np.float64) / float(width)
    left = width // 2
    right = width - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def configure_style() -> None:
    """Apply the project's final-figure-small settings."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 1.0,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "legend.frameon": False,
        }
    )


def draw_figure(
    relative: np.ndarray,
    observed: np.ndarray,
    predicted: np.ndarray,
    g4_scores: np.ndarray,
    coverage: np.ndarray,
    g4_coverage: np.ndarray,
    n_seqlets: int,
    pattern_name: str,
    min_coverage_fraction: float,
    smooth_bp: int,
    output_pdf: Path,
) -> None:
    minimum_coverage = max(1, int(np.ceil(n_seqlets * min_coverage_fraction)))
    keep = coverage >= minimum_coverage
    if not np.any(keep):
        raise ValueError(
            "No positions satisfy --min-coverage-fraction; lower the threshold"
        )

    observed_plot = moving_average(observed, smooth_bp)
    predicted_plot = moving_average(predicted, smooth_bp)
    g4_plot = moving_average(g4_scores, smooth_bp)
    observed_plot[~keep] = np.nan
    predicted_plot[~keep] = np.nan
    g4_keep = g4_coverage >= minimum_coverage
    g4_plot[~g4_keep] = np.nan

    configure_style()
    fig, g4_axis = plt.subplots(figsize=FIGURE_SIZE)
    pausing_axis = g4_axis.twinx()
    g4_line = g4_axis.plot(
        relative,
        g4_plot,
        color=G4_COLOR,
        linewidth=1.35,
        label="G4Hunter score",
        zorder=2,
    )[0]
    observed_line = pausing_axis.plot(
        relative,
        observed_plot,
        color=OBSERVED_COLOR,
        linewidth=1.05,
        label="Observed NET-seq",
        zorder=3,
    )[0]
    predicted_line = pausing_axis.plot(
        relative,
        predicted_plot,
        color=PREDICTED_COLOR,
        linewidth=1.35,
        label="PauseNet predicted",
        zorder=2,
    )[0]
    g4_axis.axvline(
        0, color="#666666", linestyle="--", linewidth=0.8, zorder=1
    )

    g4_axis.set_xlim(int(relative[0]), int(relative[-1]))
    finite_g4 = g4_plot[np.isfinite(g4_plot)]
    g4_min = float(np.min(finite_g4))
    g4_max = float(np.max(finite_g4))
    g4_span = max(g4_max - min(0.0, g4_min), 0.1)
    g4_axis.set_ylim(min(0.0, g4_min) - 0.04 * g4_span, g4_max + 0.08 * g4_span)
    pausing_axis.set_ylim(0, PAUSING_YMAX)

    g4_axis.set_xlabel("Position relative to motif center (bp)", fontsize=14)
    g4_axis.set_ylabel("Mean G4Hunter score", fontsize=14)
    pausing_axis.set_ylabel("Mean pausing signal", fontsize=14)
    g4_axis.set_title(pattern_name.replace("_", " "), fontsize=14, pad=7)
    g4_axis.tick_params(axis="both", labelsize=12)
    pausing_axis.tick_params(axis="y", labelsize=12)
    g4_axis.spines["top"].set_visible(False)
    g4_axis.spines["right"].set_visible(False)
    pausing_axis.spines["top"].set_visible(False)
    pausing_axis.legend(
        handles=[g4_line, observed_line, predicted_line],
        fontsize=12,
        frameon=False,
        loc="upper right",
    )
    g4_axis.text(
        0.03,
        0.95,
        f"n = {n_seqlets:,} seqlets",
        transform=g4_axis.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.18, right=0.82, bottom=0.17, top=0.90)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, format="pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    pattern_label = resolve_task_inputs(args)
    check_inputs(args)
    dataset_indices, centers, starts, ends, is_revcomp = load_seqlet_mapping(
        args.tfmodisco_h5,
        args.pattern_group,
        args.selected_dataset_indices,
        args.max_seqlets,
    )
    observed, predicted, _ = load_profiles(args.profiles_npz, dataset_indices)
    sequence_codes = load_non_template_sequence_codes(
        args.sequence_codes,
        dataset_indices,
        observed.shape[1],
    )
    g4_scores = positional_g4hunter_scores(
        sequence_codes,
        args.g4_window_size,
    )
    (
        relative,
        observed_mean,
        predicted_mean,
        g4_mean,
        coverage,
        g4_coverage,
    ) = align_and_average(
        observed,
        predicted,
        g4_scores,
        centers,
        is_revcomp,
        args.flank,
        args.orientation,
    )
    draw_figure(
        relative,
        observed_mean,
        predicted_mean,
        g4_mean,
        coverage,
        g4_coverage,
        len(dataset_indices),
        pattern_label,
        args.min_coverage_fraction,
        args.smooth_bp,
        args.output_pdf,
    )

    print(f"Pattern: {pattern_label} ({args.pattern_group})")
    print(f"Seqlets: {len(dataset_indices):,}")
    print(f"Unique test examples: {len(np.unique(dataset_indices)):,}")
    print(
        "Seqlet center range in stored profile: "
        f"{int(centers.min())}..{int(centers.max())} bp"
    )
    print(
        "Seqlet length range: "
        f"{int((ends - starts).min())}..{int((ends - starts).max())} bp"
    )
    print(
        "Reverse-complement aligned by TF-MoDISco: "
        f"{int(is_revcomp.sum()):,}/{len(is_revcomp):,}"
    )
    print(
        "Per-position coverage range: "
        f"{int(coverage.min()):,}..{int(coverage.max()):,} seqlets"
    )
    print(
        "Per-position G4 coverage range: "
        f"{int(g4_coverage.min()):,}..{int(g4_coverage.max()):,} seqlets"
    )
    print(
        "Mean G4Hunter range: "
        f"{float(np.min(g4_mean)):.4f}..{float(np.max(g4_mean)):.4f}"
    )
    pausing_data_max = max(
        float(np.max(observed_mean)), float(np.max(predicted_mean))
    )
    print(f"Right y-axis limits: 0..{PAUSING_YMAX:g}")
    print(f"Unclipped pausing-curve maximum: {pausing_data_max:.4f}")
    print(f"Saved: {args.output_pdf}")


if __name__ == "__main__":
    main()
