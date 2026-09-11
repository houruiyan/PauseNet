#!/usr/bin/env python3
"""PauseNet in-silico disruption of seqlet G4 propensity.

For every TF-MoDISco seqlet in one positive pattern, this script permutes only
the bases inside that seqlet. Each permutation preserves the exact A/C/G/T
counts (and therefore GC content) and the candidate with the lowest maximum
positive G4Hunter score on the stored non-template/coding strand is retained.

The original and G4-disrupted 2114-bp sequences are evaluated with a trained
PauseNet checkpoint. Predicted profiles are aligned on each seqlet centre and
aggregated. This is inference with an already trained model; the model is not
retrained.

Important: shuffling changes primary-sequence grammar as well as G4 propensity.
The experiment measures PauseNet sensitivity to composition-preserving sequence
perturbations selected to reduce G4Hunter score; it is not a structure-only
causal intervention.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import re
import sys
from pathlib import Path
from typing import Any

import h5py
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path("/mnt/HDD8TB/houruiyan/pausing_site")
HERE = Path(__file__).resolve().parent
PAUSENET_REPO = ROOT / "PauseNet"
DEFAULT_MODEL = ROOT / "2_train_model/train/hek293t_netseq/best_model.pt"
DEFAULT_TEST_DIR = ROOT / "data/HEK293T_NETseq/dataset/test"
TFMODISCO_ROOT = ROOT / "3_model_explanation/TFMoDISco/results/hek293t_netseq"

SOURCE_INPUTS = {
    "count": {
        "h5": TFMODISCO_ROOT / "count/count_tfmodisco_patterns.h5",
        "bridge": TFMODISCO_ROOT / "count/count_selected_dataset_indices.npy",
    },
    "profile": {
        "h5": TFMODISCO_ROOT / "profile/profile_tfmodisco_patterns.h5",
        "bridge": TFMODISCO_ROOT / "profile/profile_selected_dataset_indices.npy",
    },
}

INPUT_LENGTH = 2114
OUTPUT_LENGTH = 1000
CROP_START = (INPUT_LENGTH - OUTPUT_LENGTH) // 2
CODE_TO_BASE = np.asarray(list("ACGT"))
PATTERN_RE = re.compile(r"^(count|profile)_pattern_(\d+)$")
SAFE_LABEL = re.compile(r"[^A-Za-z0-9_.-]+")

COLOR_WT = "#555555"
COLOR_MUTANT = "#0072B2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    input_group = parser.add_argument_group("TF-MoDISco seqlets")
    input_group.add_argument(
        "--pattern",
        default=None,
        help="pattern name such as profile_pattern_6 or count_pattern_11",
    )
    input_group.add_argument(
        "--source", choices=("count", "profile"), default="profile"
    )
    input_group.add_argument("--pattern-index", type=int, default=None)
    input_group.add_argument("--tfmodisco-h5", type=Path, default=None)
    input_group.add_argument("--bridge", type=Path, default=None)
    input_group.add_argument(
        "--seqlet-index",
        type=int,
        action="append",
        default=None,
        help="select one seqlet row; repeat to select several (default: all)",
    )
    input_group.add_argument("--max-seqlets", type=int, default=None)

    mutation_group = parser.add_argument_group("G4-disrupting shuffle search")
    mutation_group.add_argument(
        "--g4-window",
        type=int,
        default=25,
        help="sliding-window length for G4Hunter scoring",
    )
    mutation_group.add_argument(
        "--g4-threshold",
        type=float,
        default=1.2,
        help="reporting threshold for a G4-positive seqlet",
    )
    mutation_group.add_argument("--n-shuffles", type=int, default=1000)
    mutation_group.add_argument(
        "--min-delta-g4",
        type=float,
        default=0.0,
        help="minimum required reduction in maximum G4Hunter score",
    )
    mutation_group.add_argument(
        "--target-g4-score",
        type=float,
        default=None,
        help="optional early stop once mutant score is <= this value",
    )
    mutation_group.add_argument("--seed", type=int, default=20260820)
    mutation_group.add_argument(
        "--jobs", type=int, default=min(16, os.cpu_count() or 1)
    )
    mutation_group.add_argument(
        "--failure-policy",
        choices=("skip", "error"),
        default="skip",
        help="handling when no shuffle sufficiently lowers G4Hunter score",
    )

    model_group = parser.add_argument_group("PauseNet inference")
    model_group.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    model_group.add_argument("--pausenet-repo", type=Path, default=PAUSENET_REPO)
    model_group.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    model_group.add_argument("--device", default="auto")
    model_group.add_argument("--batch-size", type=int, default=128)

    plot_group = parser.add_argument_group("plot and output")
    plot_group.add_argument("--plot-left", type=int, default=-500)
    plot_group.add_argument("--plot-right", type=int, default=500)
    plot_group.add_argument("--output-prefix", type=Path, default=None)
    plot_group.add_argument(
        "--write-png",
        action="store_true",
        help="also export PNG; PDF is the default server-side figure",
    )
    return parser.parse_args()


def normalize_pattern_args(args: argparse.Namespace) -> None:
    if args.pattern:
        match = PATTERN_RE.fullmatch(args.pattern)
        if match is None:
            raise ValueError(
                "--pattern must look like profile_pattern_6 or count_pattern_11"
            )
        source = match.group(1)
        pattern_index = int(match.group(2))
        if args.pattern_index is not None and args.pattern_index != pattern_index:
            raise ValueError("--pattern conflicts with --pattern-index")
        args.source = source
        args.pattern_index = pattern_index
    elif args.pattern_index is None:
        args.pattern_index = 6


def resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path]:
    defaults = SOURCE_INPUTS[args.source]
    return (
        Path(args.tfmodisco_h5 or defaults["h5"]),
        Path(args.bridge or defaults["bridge"]),
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.pattern_index is None or args.pattern_index < 0:
        raise ValueError("--pattern-index must be nonnegative")
    if args.max_seqlets is not None and args.max_seqlets < 1:
        raise ValueError("--max-seqlets must be positive")
    if args.n_shuffles < 1:
        raise ValueError("--n-shuffles must be positive")
    if args.jobs < 1 or args.batch_size < 1:
        raise ValueError("--jobs and --batch-size must be positive")
    if args.g4_window < 1 or args.g4_window > 50:
        raise ValueError("--g4-window must be between 1 and 50")
    if args.g4_threshold < 0 or args.min_delta_g4 < 0:
        raise ValueError("G4 threshold and minimum delta must be nonnegative")
    if args.plot_right <= args.plot_left:
        raise ValueError("--plot-right must exceed --plot-left")

    h5_path, bridge_path = resolve_inputs(args)
    required = [
        h5_path,
        bridge_path,
        args.model,
        args.pausenet_repo / "pausenet/evaluate.py",
        args.test_dir / "sequence_codes.npy",
        args.test_dir / "manifest.tsv",
    ]
    missing = [str(path) for path in required if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("Missing input file(s): " + ", ".join(missing))


def load_seqlets(args: argparse.Namespace) -> list[dict[str, Any]]:
    h5_path, bridge_path = resolve_inputs(args)
    bridge = np.asarray(np.load(bridge_path), dtype=np.int64)
    if bridge.ndim != 1:
        raise ValueError(f"Bridge must be one-dimensional: {bridge_path}")

    group_path = f"pos_patterns/pattern_{args.pattern_index}/seqlets"
    with h5py.File(h5_path, "r") as handle:
        if group_path not in handle:
            raise KeyError(f"{h5_path} does not contain {group_path}")
        group = handle[group_path]
        example_idx = np.asarray(group["example_idx"], dtype=np.int64)
        starts = np.asarray(group["start"], dtype=np.int64)
        ends = np.asarray(group["end"], dtype=np.int64)
        revcomp = np.asarray(group["is_revcomp"], dtype=bool)

    if np.any(example_idx < 0) or np.any(example_idx >= len(bridge)):
        raise IndexError("A seqlet example_idx falls outside the bridge")
    rows = [
        {
            "seqlet_index": int(i),
            "example_idx": int(example_idx[i]),
            "dataset_index": int(bridge[example_idx[i]]),
            "seqlet_start": int(starts[i]),
            "seqlet_end": int(ends[i]),
            "is_revcomp": bool(revcomp[i]),
        }
        for i in range(len(example_idx))
    ]

    requested = set(args.seqlet_index or [])
    if requested:
        rows = [row for row in rows if row["seqlet_index"] in requested]
        found = {row["seqlet_index"] for row in rows}
        missing = sorted(requested.difference(found))
        if missing:
            raise ValueError(f"Seqlet indices not found: {missing}")
    if args.max_seqlets is not None:
        rows = rows[: args.max_seqlets]
    if not rows:
        raise ValueError("No seqlets remain after filtering")
    for row in rows:
        start, end = row["seqlet_start"], row["seqlet_end"]
        if start < 0 or end > OUTPUT_LENGTH or end <= start:
            raise ValueError(
                f"Invalid crop coordinates for seqlet {row['seqlet_index']}: "
                f"[{start}, {end})"
            )
        if end - start < args.g4_window:
            raise ValueError(
                f"Seqlet {row['seqlet_index']} is shorter than --g4-window"
            )
    return rows


def load_manifest(path: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {
            "array_index",
            "chrom",
            "output_start",
            "output_end",
            "strand",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
        for row in reader:
            records[int(row["array_index"])] = {
                "chrom": row["chrom"],
                "output_start": int(row["output_start"]),
                "output_end": int(row["output_end"]),
                "strand": row["strand"],
                "gene_id": row.get("gene_id", ""),
                "gene_name": row.get("gene_name", ""),
                "transcript_id": row.get("transcript_id", ""),
                "region_type": row.get("region_type", ""),
            }
    return records


def genomic_seqlet_coordinates(
    manifest_row: dict[str, Any], start_crop: int, end_crop: int
) -> tuple[int, int]:
    if manifest_row["strand"] == "+":
        return (
            manifest_row["output_start"] + start_crop,
            manifest_row["output_start"] + end_crop,
        )
    return (
        manifest_row["output_end"] - end_crop,
        manifest_row["output_end"] - start_crop,
    )


def codes_to_sequence(codes: np.ndarray) -> str:
    codes = np.asarray(codes, dtype=np.int64)
    if np.any((codes < 0) | (codes > 3)):
        raise ValueError("Sequence contains non-ACGT code(s)")
    return "".join(CODE_TO_BASE[codes].tolist())


def g4hunter_base_scores(codes: np.ndarray) -> np.ndarray:
    """Return signed G4Hunter scores: G runs positive, C runs negative."""
    codes = np.asarray(codes, dtype=np.int64)
    scores = np.zeros(len(codes), dtype=np.int8)
    index = 0
    while index < len(codes):
        code = int(codes[index])
        if code not in (1, 2):  # C=1, G=2
            index += 1
            continue
        end = index + 1
        while end < len(codes) and int(codes[end]) == code:
            end += 1
        magnitude = min(end - index, 4)
        scores[index:end] = magnitude if code == 2 else -magnitude
        index = end
    return scores


def g4hunter_window_scores(codes: np.ndarray, window: int) -> np.ndarray:
    base_scores = g4hunter_base_scores(codes).astype(np.float64)
    cumulative = np.concatenate(([0.0], np.cumsum(base_scores)))
    return (cumulative[window:] - cumulative[:-window]) / float(window)


def g4hunter_summary(codes: np.ndarray, window: int, threshold: float) -> dict:
    scores = g4hunter_window_scores(codes, window)
    max_index = int(np.argmax(scores))
    max_score = float(scores[max_index])
    return {
        "max_score": max_score,
        "max_window_start": max_index,
        "max_window_end": max_index + window,
        "max_window_sequence": codes_to_sequence(
            codes[max_index : max_index + window]
        ),
        "n_windows_ge_threshold": int(np.sum(scores >= threshold)),
        "g4_positive": bool(max_score >= threshold),
    }


def disrupt_one_seqlet(job: dict[str, Any]) -> dict[str, Any]:
    wildtype_codes = np.asarray(job["seqlet_codes"], dtype=np.int64)
    window = int(job["g4_window"])
    threshold = float(job["g4_threshold"])
    wildtype = g4hunter_summary(wildtype_codes, window, threshold)

    seed_sequence = np.random.SeedSequence(
        [int(job["seed"]), int(job["seqlet_index"]), int(job["dataset_index"])]
    )
    rng = np.random.default_rng(seed_sequence)
    best_codes: np.ndarray | None = None
    best_summary: dict[str, Any] | None = None
    best_score = np.inf
    evaluated = 0
    for _ in range(int(job["n_shuffles"])):
        shuffled = rng.permutation(wildtype_codes)
        if np.array_equal(shuffled, wildtype_codes):
            continue
        candidate = g4hunter_summary(shuffled, window, threshold)
        evaluated += 1
        if candidate["max_score"] < best_score:
            best_score = float(candidate["max_score"])
            best_codes = shuffled.copy()
            best_summary = candidate
        target = job["target_g4_score"]
        if target is not None and best_score <= float(target):
            break

    required_score = wildtype["max_score"] - float(job["min_delta_g4"])
    success = best_codes is not None and best_score < required_score
    status = "success" if success else "no_g4_score_decrease"

    if best_codes is not None:
        if not np.array_equal(
            np.bincount(wildtype_codes, minlength=4),
            np.bincount(best_codes, minlength=4),
        ):
            raise AssertionError("A/C/G/T counts changed during permutation")

    gc_count = int(np.count_nonzero(np.isin(wildtype_codes, (1, 2))))
    return {
        **{key: value for key, value in job.items() if key != "seqlet_codes"},
        "status": status,
        "wildtype_codes": wildtype_codes,
        "mutant_codes": best_codes,
        "wildtype_sequence": codes_to_sequence(wildtype_codes),
        "mutant_sequence": codes_to_sequence(best_codes) if best_codes is not None else "",
        "wildtype_g4_score": float(wildtype["max_score"]),
        "mutant_g4_score": float(best_score),
        "delta_g4_score": float(best_score - wildtype["max_score"]),
        "wildtype_max_window_start": int(wildtype["max_window_start"]),
        "wildtype_max_window_sequence": wildtype["max_window_sequence"],
        "mutant_max_window_start": (
            int(best_summary["max_window_start"]) if best_summary else -1
        ),
        "mutant_max_window_sequence": (
            best_summary["max_window_sequence"] if best_summary else ""
        ),
        "wildtype_g4_positive": bool(wildtype["g4_positive"]),
        "mutant_g4_positive": (
            bool(best_summary["g4_positive"]) if best_summary else False
        ),
        "wildtype_n_windows_ge_threshold": int(
            wildtype["n_windows_ge_threshold"]
        ),
        "mutant_n_windows_ge_threshold": (
            int(best_summary["n_windows_ge_threshold"]) if best_summary else 0
        ),
        "gc_count": gc_count,
        "gc_fraction": gc_count / len(wildtype_codes),
        "shuffles_evaluated": evaluated,
    }


def resolve_device(requested: str) -> torch.device:
    normalized = requested.strip().lower()
    if normalized == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        print(f"WARNING: {requested} unavailable; falling back to CPU", flush=True)
        return torch.device("cpu")
    return device


def predict_profiles(
    model: torch.nn.Module,
    sequences: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    profiles: list[np.ndarray] = []
    counts: list[np.ndarray] = []
    for start in range(0, len(sequences), batch_size):
        batch = torch.from_numpy(sequences[start : start + batch_size]).long().to(device)
        with torch.no_grad():
            logits, predicted_log1p_counts = model(batch)
            probabilities = torch.softmax(logits.float(), dim=-1)
            predicted_counts = torch.expm1(predicted_log1p_counts.float()).clamp_min(0)
            tracks = probabilities * predicted_counts[:, None]
        profiles.append(tracks.cpu().numpy().astype(np.float32))
        counts.append(predicted_counts.cpu().numpy().astype(np.float32))
    return np.concatenate(profiles), np.concatenate(counts)


def align_profiles(
    profiles: np.ndarray,
    centers_crop: np.ndarray,
    plot_left: int,
    plot_right: int,
) -> tuple[np.ndarray, np.ndarray]:
    relative = np.arange(plot_left, plot_right, dtype=np.int64)
    aligned = np.full((len(profiles), len(relative)), np.nan, dtype=np.float32)
    for row, center in enumerate(centers_crop):
        source_positions = int(center) + relative
        valid = (source_positions >= 0) & (source_positions < OUTPUT_LENGTH)
        aligned[row, valid] = profiles[row, source_positions[valid]]
    return relative, aligned


def mean_and_sem(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count = np.isfinite(matrix).sum(axis=0)
    mean = np.nanmean(matrix, axis=0)
    standard_deviation = np.nanstd(matrix, axis=0, ddof=0)
    sem = np.divide(
        standard_deviation,
        np.sqrt(count),
        out=np.zeros_like(standard_deviation),
        where=count > 0,
    )
    return mean, sem


def configure_style() -> None:
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
            "axes.titlesize": 12,
        }
    )


def draw_figure(
    x: np.ndarray,
    wildtype_mean: np.ndarray,
    wildtype_sem: np.ndarray,
    mutant_mean: np.ndarray,
    mutant_sem: np.ndarray,
    label: str,
    n_seqlets: int,
    delta_scores: np.ndarray,
    output_pdf: Path,
    output_png: Path | None,
) -> None:
    configure_style()
    fig, axis = plt.subplots(figsize=(5, 4))
    axis.fill_between(
        x,
        np.maximum(0, wildtype_mean - wildtype_sem),
        wildtype_mean + wildtype_sem,
        color=COLOR_WT,
        alpha=0.16,
        linewidth=0,
    )
    axis.fill_between(
        x,
        np.maximum(0, mutant_mean - mutant_sem),
        mutant_mean + mutant_sem,
        color=COLOR_MUTANT,
        alpha=0.16,
        linewidth=0,
    )
    axis.plot(x, wildtype_mean, color=COLOR_WT, linewidth=1.4, label="Original")
    axis.plot(
        x,
        mutant_mean,
        color=COLOR_MUTANT,
        linewidth=1.4,
        label="G4-disrupted",
    )
    axis.axvline(0, color="#777777", linewidth=0.8, linestyle=(0, (3, 2)))
    axis.set_xlim(float(x[0]), float(x[-1]))
    axis.set_ylim(bottom=0)
    axis.set_xlabel("Position relative to seqlet center (bp)")
    axis.set_ylabel("Mean predicted pausing signal")
    axis.set_title(label.replace("_", " "), pad=6)
    axis.text(
        0.03,
        0.96,
        f"n = {n_seqlets:,}\nmedian ΔG4 = {np.median(delta_scores):+.2f}",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color="#222222",
    )
    axis.legend(frameon=False, loc="upper right")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(False)
    fig.tight_layout()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    if output_png is not None:
        fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "status",
        "seqlet_index",
        "example_idx",
        "dataset_index",
        "chrom",
        "strand",
        "seqlet_genomic_start_hg19",
        "seqlet_genomic_end_hg19",
        "seqlet_start_crop",
        "seqlet_end_crop",
        "seqlet_center_crop",
        "is_revcomp",
        "gene_name",
        "gene_id",
        "transcript_id",
        "region_type",
        "seqlet_length",
        "gc_count",
        "gc_fraction",
        "wildtype_sequence",
        "mutant_sequence",
        "wildtype_g4hunter_score",
        "mutant_g4hunter_score",
        "delta_g4hunter_score",
        "wildtype_g4_positive",
        "mutant_g4_positive",
        "wildtype_n_windows_ge_threshold",
        "mutant_n_windows_ge_threshold",
        "wildtype_max_window_start_0based",
        "wildtype_max_window_sequence",
        "mutant_max_window_start_0based",
        "mutant_max_window_sequence",
        "shuffles_evaluated",
        "predicted_count_original",
        "predicted_count_mutant",
        "predicted_count_ratio_mutant_over_original",
        "seqlet_window_signal_original",
        "seqlet_window_signal_mutant",
        "seqlet_window_signal_ratio_mutant_over_original",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> int:
    args = parse_args()
    normalize_pattern_args(args)
    validate_args(args)
    label = f"{args.source}_pattern_{args.pattern_index}"
    safe_label = SAFE_LABEL.sub("_", label).strip("_")
    output_prefix = (
        args.output_prefix.resolve()
        if args.output_prefix is not None
        else (HERE / f"{safe_label}_g4_knockout").resolve()
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    seqlets = load_seqlets(args)
    sequence_codes = np.load(args.test_dir / "sequence_codes.npy", mmap_mode="r")
    if sequence_codes.ndim != 2 or sequence_codes.shape[1] != INPUT_LENGTH:
        raise ValueError(
            f"Expected sequence_codes (N,{INPUT_LENGTH}), got {sequence_codes.shape}"
        )
    manifest = load_manifest(args.test_dir / "manifest.tsv")
    missing_manifest = sorted(
        {row["dataset_index"] for row in seqlets}.difference(manifest)
    )
    if missing_manifest:
        raise KeyError(f"Dataset indices missing from manifest: {missing_manifest[:10]}")
    if max(row["dataset_index"] for row in seqlets) >= len(sequence_codes):
        raise IndexError("A seqlet dataset_index exceeds sequence_codes.npy")

    print(f"Input: {len(seqlets):,} seqlets from {label}", flush=True)
    jobs: list[dict[str, Any]] = []
    for row in seqlets:
        start_input = CROP_START + row["seqlet_start"]
        end_input = CROP_START + row["seqlet_end"]
        jobs.append(
            {
                **row,
                "seqlet_codes": np.asarray(
                    sequence_codes[row["dataset_index"], start_input:end_input],
                    dtype=np.int64,
                ),
                "g4_window": args.g4_window,
                "g4_threshold": args.g4_threshold,
                "n_shuffles": args.n_shuffles,
                "min_delta_g4": args.min_delta_g4,
                "target_g4_score": args.target_g4_score,
                "seed": args.seed,
            }
        )

    if args.jobs == 1:
        mutation_results = [disrupt_one_seqlet(job) for job in jobs]
    else:
        context = mp.get_context("fork")
        with context.Pool(processes=args.jobs) as pool:
            mutation_results = list(
                pool.imap(disrupt_one_seqlet, jobs, chunksize=4)
            )

    successful = [row for row in mutation_results if row["status"] == "success"]
    failed = [row for row in mutation_results if row["status"] != "success"]
    print(
        f"G4 disruption: {len(successful):,} successful, {len(failed):,} failed",
        flush=True,
    )
    if failed and args.failure_policy == "error":
        indices = ", ".join(str(row["seqlet_index"]) for row in failed[:10])
        raise RuntimeError(f"No acceptable G4-lowering shuffle for: {indices}")
    if not successful:
        raise RuntimeError(
            "No seqlet obtained a lower G4Hunter score. Increase --n-shuffles "
            "or reduce --min-delta-g4."
        )

    wildtype_full: list[np.ndarray] = []
    mutant_full: list[np.ndarray] = []
    centers_crop: list[int] = []
    for result in successful:
        original = np.asarray(
            sequence_codes[result["dataset_index"]], dtype=np.int64
        ).copy()
        mutant = original.copy()
        start_input = CROP_START + result["seqlet_start"]
        end_input = CROP_START + result["seqlet_end"]
        mutant[start_input:end_input] = result["mutant_codes"]
        wildtype_full.append(original)
        mutant_full.append(mutant)
        centers_crop.append(
            (result["seqlet_start"] + result["seqlet_end"]) // 2
        )

    device = resolve_device(args.device)
    sys.path.insert(0, str(args.pausenet_repo))
    from pausenet.evaluate import load_model  # noqa: E402

    model, checkpoint_config = load_model(args.model, device)
    wildtype_array = np.stack(wildtype_full).astype(np.int64)
    mutant_array = np.stack(mutant_full).astype(np.int64)
    wildtype_profiles, wildtype_counts = predict_profiles(
        model, wildtype_array, device, args.batch_size
    )
    mutant_profiles, mutant_counts = predict_profiles(
        model, mutant_array, device, args.batch_size
    )

    centers_array = np.asarray(centers_crop, dtype=np.int64)
    x, wildtype_aligned = align_profiles(
        wildtype_profiles, centers_array, args.plot_left, args.plot_right
    )
    _, mutant_aligned = align_profiles(
        mutant_profiles, centers_array, args.plot_left, args.plot_right
    )
    wildtype_mean, wildtype_sem = mean_and_sem(wildtype_aligned)
    mutant_mean, mutant_sem = mean_and_sem(mutant_aligned)
    delta_scores = np.asarray(
        [row["delta_g4_score"] for row in successful], dtype=np.float64
    )

    output_pdf = output_prefix.with_suffix(".pdf")
    output_png = output_prefix.with_suffix(".png") if args.write_png else None
    output_summary = output_prefix.parent / f"{output_prefix.name}_seqlets.tsv"
    output_profiles = output_prefix.parent / f"{output_prefix.name}_profiles.npz"
    output_parameters = output_prefix.parent / f"{output_prefix.name}_parameters.json"
    draw_figure(
        x,
        wildtype_mean,
        wildtype_sem,
        mutant_mean,
        mutant_sem,
        label,
        len(successful),
        delta_scores,
        output_pdf,
        output_png,
    )

    success_position = {
        row["seqlet_index"]: position for position, row in enumerate(successful)
    }
    summary_rows: list[dict[str, Any]] = []
    for result in mutation_results:
        manifest_row = manifest[result["dataset_index"]]
        genomic_start, genomic_end = genomic_seqlet_coordinates(
            manifest_row, result["seqlet_start"], result["seqlet_end"]
        )
        row = {
            "status": result["status"],
            "seqlet_index": result["seqlet_index"],
            "example_idx": result["example_idx"],
            "dataset_index": result["dataset_index"],
            "chrom": manifest_row["chrom"],
            "strand": manifest_row["strand"],
            "seqlet_genomic_start_hg19": genomic_start,
            "seqlet_genomic_end_hg19": genomic_end,
            "seqlet_start_crop": result["seqlet_start"],
            "seqlet_end_crop": result["seqlet_end"],
            "seqlet_center_crop": (
                result["seqlet_start"] + result["seqlet_end"]
            )
            // 2,
            "is_revcomp": result["is_revcomp"],
            "gene_name": manifest_row["gene_name"],
            "gene_id": manifest_row["gene_id"],
            "transcript_id": manifest_row["transcript_id"],
            "region_type": manifest_row["region_type"],
            "seqlet_length": result["seqlet_end"] - result["seqlet_start"],
            "gc_count": result["gc_count"],
            "gc_fraction": result["gc_fraction"],
            "wildtype_sequence": result["wildtype_sequence"],
            "mutant_sequence": result["mutant_sequence"],
            "wildtype_g4hunter_score": result["wildtype_g4_score"],
            "mutant_g4hunter_score": result["mutant_g4_score"],
            "delta_g4hunter_score": result["delta_g4_score"],
            "wildtype_g4_positive": result["wildtype_g4_positive"],
            "mutant_g4_positive": result["mutant_g4_positive"],
            "wildtype_n_windows_ge_threshold": result[
                "wildtype_n_windows_ge_threshold"
            ],
            "mutant_n_windows_ge_threshold": result[
                "mutant_n_windows_ge_threshold"
            ],
            "wildtype_max_window_start_0based": result[
                "wildtype_max_window_start"
            ],
            "wildtype_max_window_sequence": result[
                "wildtype_max_window_sequence"
            ],
            "mutant_max_window_start_0based": result[
                "mutant_max_window_start"
            ],
            "mutant_max_window_sequence": result[
                "mutant_max_window_sequence"
            ],
            "shuffles_evaluated": result["shuffles_evaluated"],
        }
        if result["status"] == "success":
            position = success_position[result["seqlet_index"]]
            start, end = result["seqlet_start"], result["seqlet_end"]
            wt_window = float(wildtype_profiles[position, start:end].sum())
            mut_window = float(mutant_profiles[position, start:end].sum())
            row.update(
                {
                    "predicted_count_original": float(wildtype_counts[position]),
                    "predicted_count_mutant": float(mutant_counts[position]),
                    "predicted_count_ratio_mutant_over_original": float(
                        mutant_counts[position]
                        / max(wildtype_counts[position], 1e-12)
                    ),
                    "seqlet_window_signal_original": wt_window,
                    "seqlet_window_signal_mutant": mut_window,
                    "seqlet_window_signal_ratio_mutant_over_original": float(
                        mut_window / max(wt_window, 1e-12)
                    ),
                }
            )
        summary_rows.append(row)
    write_summary(output_summary, summary_rows)

    np.savez_compressed(
        output_profiles,
        relative_position_bp=x,
        wildtype_profiles_aligned=wildtype_aligned,
        mutant_profiles_aligned=mutant_aligned,
        wildtype_mean=wildtype_mean,
        wildtype_sem=wildtype_sem,
        mutant_mean=mutant_mean,
        mutant_sem=mutant_sem,
        wildtype_predicted_counts=wildtype_counts,
        mutant_predicted_counts=mutant_counts,
        seqlet_indices=np.asarray(
            [row["seqlet_index"] for row in successful], dtype=np.int64
        ),
        dataset_indices=np.asarray(
            [row["dataset_index"] for row in successful], dtype=np.int64
        ),
        wildtype_g4hunter_score=np.asarray(
            [row["wildtype_g4_score"] for row in successful], dtype=np.float64
        ),
        mutant_g4hunter_score=np.asarray(
            [row["mutant_g4_score"] for row in successful], dtype=np.float64
        ),
    )

    h5_path, bridge_path = resolve_inputs(args)
    parameters = {
        "label": label,
        "source": args.source,
        "pattern_index": args.pattern_index,
        "n_seqlets_input": len(seqlets),
        "n_seqlets_successful": len(successful),
        "n_seqlets_failed": len(failed),
        "mutation": {
            "scope": "only bases inside each TF-MoDISco seqlet",
            "composition_constraint": "exact A/C/G/T counts preserved",
            "objective": "minimize maximum positive G4Hunter sliding-window score",
            "g4_window_nt": args.g4_window,
            "g4_positive_threshold": args.g4_threshold,
            "n_shuffles": args.n_shuffles,
            "min_delta_g4": args.min_delta_g4,
            "target_g4_score": args.target_g4_score,
            "seed": args.seed,
        },
        "strand": {
            "sequence_orientation": "PauseNet stored transcription/non-template orientation",
            "is_revcomp_used_to_flip": False,
        },
        "prediction": {
            "checkpoint": str(args.model.resolve()),
            "device": str(device),
            "formula": "softmax(profile_logits) * expm1(predicted_log1p_count)",
            "reverse_complement_average": False,
            "checkpoint_model_config": checkpoint_config.get("model", {}),
        },
        "plot": {
            "left_bp": args.plot_left,
            "right_bp_exclusive": args.plot_right,
            "center": "TF-MoDISco seqlet center",
            "bands": "standard error of the mean",
        },
        "inputs": {
            "tfmodisco_h5": str(h5_path.resolve()),
            "bridge": str(bridge_path.resolve()),
            "sequence_codes": str((args.test_dir / "sequence_codes.npy").resolve()),
            "manifest": str((args.test_dir / "manifest.tsv").resolve()),
        },
        "outputs": {
            "pdf": str(output_pdf),
            "summary_tsv": str(output_summary),
            "profiles_npz": str(output_profiles),
            "png": str(output_png) if output_png else None,
        },
        "interpretation_warning": (
            "The shuffle also changes primary-sequence motif grammar; prediction "
            "differences are not uniquely attributable to G4 structure."
        ),
    }
    output_parameters.write_text(
        json.dumps(parameters, indent=2) + "\n", encoding="utf-8"
    )

    count_ratio = mutant_counts / np.maximum(wildtype_counts, 1e-12)
    print(
        f"G4 score delta: median={np.median(delta_scores):+.3f}, "
        f"range=[{np.min(delta_scores):+.3f}, {np.max(delta_scores):+.3f}]",
        flush=True,
    )
    print(
        f"Predicted count ratio mutant/original: median={np.median(count_ratio):.4f}",
        flush=True,
    )
    print(f"PDF: {output_pdf}")
    print(f"Seqlet summary: {output_summary}")
    print(f"Profiles: {output_profiles}")
    print(f"Parameters: {output_parameters}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
