#!/usr/bin/env python3
"""Export strand-aware G4Hunter scores for every profile pattern-6 seqlet.

TF-MoDISco stores seqlet sequences in motif-aligned orientation.  For seqlets
with ``is_revcomp=True`` this script reverse-complements the stored sequence
back to the original PauseNet input orientation, which is treated here as the
5'-to-3' transcription-oriented non-template/coding strand.

The output contains both non-template-strand and complementary/template-strand
scores.  This prevents a G-rich reverse complement from being accidentally
reported as a non-template-strand G4 signal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DEFAULT_H5 = Path(
    "/mnt/HDD8TB/houruiyan/pausing_site/3_model_explanation/TFMoDISco/"
    "results/hek293t_netseq/profile/profile_tfmodisco_patterns.h5"
)
DEFAULT_SELECTED_INDICES = Path(
    "/mnt/HDD8TB/houruiyan/pausing_site/3_model_explanation/TFMoDISco/"
    "results/hek293t_netseq/profile/profile_selected_dataset_indices.npy"
)
DEFAULT_OUTPUT = HERE / "profile_pattern_6_all_seqlets_g4hunter_scores.tsv"
BASES = np.asarray(list("ACGT"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate G4Hunter scores for all TF-MoDISco profile pattern-6 "
            "seqlets and export a TSV table."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tfmodisco-h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--pattern-group", default="pos_patterns/pattern_6")
    parser.add_argument(
        "--selected-dataset-indices",
        type=Path,
        default=DEFAULT_SELECTED_INDICES,
        help="Maps TF-MoDISco example_idx to the original test-dataset row.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=25,
        help="G4Hunter sliding-window length in nucleotides.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.2,
        help="Positive G4Hunter threshold used for Boolean calls.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=HERE / "profile_pattern_6_g4hunter_summary.tsv",
    )
    parser.add_argument(
        "--parameters-output",
        type=Path,
        default=HERE / "profile_pattern_6_g4hunter_parameters.json",
    )
    return parser.parse_args()


def reverse_complement_onehot(onehot: np.ndarray) -> np.ndarray:
    """Reverse-complement an (L,4) A/C/G/T one-hot matrix."""
    return onehot[::-1, ::-1]


def reverse_complement_sequence(sequence: str) -> str:
    table = str.maketrans("ACGTN", "TGCAN")
    return sequence.translate(table)[::-1]


def onehot_to_sequence(onehot: np.ndarray) -> str:
    onehot = np.asarray(onehot, dtype=np.float64)
    if onehot.ndim != 2 or onehot.shape[1] != 4:
        raise ValueError(f"Expected (L,4) A/C/G/T matrix, received {onehot.shape}")
    if not np.all(np.isfinite(onehot)):
        raise ValueError("Seqlet sequence contains NaN or Inf")
    if np.min(onehot) < -1e-6:
        raise ValueError("Seqlet sequence contains negative values")

    row_sums = onehot.sum(axis=1)
    sequence = BASES[np.argmax(onehot, axis=1)].astype("<U1")
    sequence[row_sums < 0.5] = "N"
    return "".join(sequence.tolist())


def g4hunter_base_scores(sequence: str) -> np.ndarray:
    """Assign signed G4Hunter scores to bases in G/C runs (capped at 4)."""
    sequence = sequence.upper()
    scores = np.zeros(len(sequence), dtype=np.int8)
    index = 0
    while index < len(sequence):
        base = sequence[index]
        if base not in {"G", "C"}:
            index += 1
            continue
        end = index + 1
        while end < len(sequence) and sequence[end] == base:
            end += 1
        magnitude = min(end - index, 4)
        scores[index:end] = magnitude if base == "G" else -magnitude
        index = end
    return scores


def score_sequence(sequence: str, window_size: int, threshold: float) -> dict:
    if not 1 <= window_size <= len(sequence):
        raise ValueError(
            f"window size must be between 1 and {len(sequence)}, got {window_size}"
        )
    base_scores = g4hunter_base_scores(sequence).astype(np.float64)
    window_scores = np.convolve(
        base_scores,
        np.ones(window_size, dtype=np.float64) / window_size,
        mode="valid",
    )
    max_index = int(np.argmax(window_scores))
    min_index = int(np.argmin(window_scores))
    abs_index = int(np.argmax(np.abs(window_scores)))
    max_score = float(window_scores[max_index])
    min_score = float(window_scores[min_index])
    max_abs_signed = float(window_scores[abs_index])
    return {
        "max_score": max_score,
        "max_window_start_0based": max_index,
        "max_window_end_0based_exclusive": max_index + window_size,
        "max_window_sequence_5to3": sequence[
            max_index : max_index + window_size
        ],
        "min_score": min_score,
        "max_abs_signed_score": max_abs_signed,
        "max_abs_score": abs(max_abs_signed),
        "mean_window_score": float(np.mean(window_scores)),
        "n_windows_ge_threshold": int(np.sum(window_scores >= threshold)),
        "g4_positive": bool(max_score >= threshold),
    }


def prefixed(prefix: str, values: dict) -> dict:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def load_seqlets(
    h5_path: Path,
    pattern_group: str,
    selected_indices_path: Path,
) -> list[dict]:
    if not h5_path.is_file():
        raise FileNotFoundError(h5_path)
    if not selected_indices_path.is_file():
        raise FileNotFoundError(selected_indices_path)

    selected_indices = np.asarray(np.load(selected_indices_path), dtype=np.int64)
    if selected_indices.ndim != 1:
        raise ValueError(
            "selected dataset indices must be one-dimensional; received "
            f"{selected_indices.shape}"
        )

    with h5py.File(h5_path, "r") as h5:
        if pattern_group not in h5:
            raise KeyError(f"Missing HDF5 group: {pattern_group}")
        group = h5[pattern_group]
        if "seqlets" not in group:
            raise KeyError(f"{pattern_group} does not contain seqlets")
        seqlets = group["seqlets"]
        required = ("sequence", "example_idx", "start", "end", "is_revcomp")
        missing = [name for name in required if name not in seqlets]
        if missing:
            raise KeyError(f"Missing seqlet datasets: {', '.join(missing)}")

        sequences = np.asarray(seqlets["sequence"], dtype=np.float64)
        example_idx = np.asarray(seqlets["example_idx"], dtype=np.int64)
        starts = np.asarray(seqlets["start"], dtype=np.int64)
        ends = np.asarray(seqlets["end"], dtype=np.int64)
        is_revcomp = np.asarray(seqlets["is_revcomp"], dtype=bool)

    n_seqlets = int(sequences.shape[0])
    if sequences.ndim != 3 or sequences.shape[2] != 4:
        raise ValueError(f"Expected seqlet sequences (N,L,4), got {sequences.shape}")
    if not all(len(x) == n_seqlets for x in (example_idx, starts, ends, is_revcomp)):
        raise ValueError("Seqlet arrays have inconsistent lengths")
    if np.any(example_idx < 0) or np.any(example_idx >= len(selected_indices)):
        raise IndexError("A seqlet example_idx is outside selected dataset indices")

    records = []
    for seqlet_index in range(n_seqlets):
        aligned_onehot = sequences[seqlet_index]
        non_template_onehot = (
            reverse_complement_onehot(aligned_onehot)
            if is_revcomp[seqlet_index]
            else aligned_onehot
        )
        aligned_sequence = onehot_to_sequence(aligned_onehot)
        non_template_sequence = onehot_to_sequence(non_template_onehot)
        records.append(
            {
                "pattern_group": pattern_group,
                "seqlet_index": seqlet_index,
                "example_idx": int(example_idx[seqlet_index]),
                "dataset_index": int(selected_indices[example_idx[seqlet_index]]),
                "start_0based": int(starts[seqlet_index]),
                "end_0based_exclusive": int(ends[seqlet_index]),
                "center_0based": int((starts[seqlet_index] + ends[seqlet_index]) // 2),
                "is_revcomp_in_tfmodisco": bool(is_revcomp[seqlet_index]),
                "tfmodisco_aligned_sequence_5to3": aligned_sequence,
                "non_template_sequence_5to3": non_template_sequence,
                "template_complement_sequence_5to3": reverse_complement_sequence(
                    non_template_sequence
                ),
            }
        )
    return records


def build_summary(table: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    for strand in ("non_template", "template_complement"):
        scores = table[f"{strand}_max_score"].to_numpy(dtype=float)
        rows.append(
            {
                "strand": strand,
                "n_seqlets": len(scores),
                "mean_max_g4hunter_score": float(np.mean(scores)),
                "median_max_g4hunter_score": float(np.median(scores)),
                "q1_max_g4hunter_score": float(np.quantile(scores, 0.25)),
                "q3_max_g4hunter_score": float(np.quantile(scores, 0.75)),
                "minimum_max_g4hunter_score": float(np.min(scores)),
                "maximum_max_g4hunter_score": float(np.max(scores)),
                "threshold": threshold,
                "n_seqlets_ge_threshold": int(np.sum(scores >= threshold)),
                "fraction_seqlets_ge_threshold": float(np.mean(scores >= threshold)),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    if args.window_size <= 0:
        raise ValueError("--window-size must be positive")
    if args.threshold < 0:
        raise ValueError("--threshold must be non-negative")

    records = load_seqlets(
        args.tfmodisco_h5.resolve(),
        args.pattern_group,
        args.selected_dataset_indices.resolve(),
    )
    if not records:
        raise ValueError(f"No seqlets found in {args.pattern_group}")

    seqlet_length = len(records[0]["non_template_sequence_5to3"])
    if args.window_size > seqlet_length:
        raise ValueError(
            f"--window-size {args.window_size} exceeds seqlet length {seqlet_length}"
        )

    output_records = []
    for record in records:
        non_template = record["non_template_sequence_5to3"]
        template = record["template_complement_sequence_5to3"]
        output_records.append(
            {
                **record,
                **prefixed(
                    "non_template",
                    score_sequence(non_template, args.window_size, args.threshold),
                ),
                **prefixed(
                    "template_complement",
                    score_sequence(template, args.window_size, args.threshold),
                ),
            }
        )

    table = pd.DataFrame(output_records)
    summary = build_summary(table, args.threshold)
    for path in (args.output, args.summary_output, args.parameters_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, sep="\t", index=False, float_format="%.6g")
    summary.to_csv(
        args.summary_output, sep="\t", index=False, float_format="%.6g"
    )

    parameters = {
        "tfmodisco_h5": str(args.tfmodisco_h5.resolve()),
        "pattern_group": args.pattern_group,
        "selected_dataset_indices": str(args.selected_dataset_indices.resolve()),
        "n_seqlets": len(table),
        "seqlet_length_nt": seqlet_length,
        "window_size_nt": args.window_size,
        "threshold": args.threshold,
        "algorithm": "G4Hunter core signed G/C-run scoring",
        "score_definition": "maximum signed mean score across sliding windows",
        "non_template_orientation": (
            "TF-MoDISco is_revcomp alignment undone; PauseNet input orientation"
        ),
        "template_complement_orientation": (
            "reverse complement of restored non-template sequence, written 5to3"
        ),
    }
    args.parameters_output.write_text(json.dumps(parameters, indent=2) + "\n")

    print(f"Pattern: {args.pattern_group}")
    print(f"Seqlets scored: {len(table):,}")
    print(f"TF-MoDISco reverse-complemented: {int(table['is_revcomp_in_tfmodisco'].sum()):,}")
    print(f"Per-seqlet table: {args.output.resolve()}")
    print(f"Summary: {args.summary_output.resolve()}")
    print(f"Parameters: {args.parameters_output.resolve()}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
