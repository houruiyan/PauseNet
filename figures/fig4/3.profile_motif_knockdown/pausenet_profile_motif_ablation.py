#!/usr/bin/env python3
"""In-silico core ablation for selected PauseNet profile TF-MoDISco motifs.

For every seqlet in the selected positive profile patterns, the contribution-
supported motif core is replaced by N in a copy of its 2,114-bp PauseNet input
sequence. The trained HEK293T NET-seq model is evaluated on the original and
ablated sequences without retraining. Predicted 1,000-bp profiles are kept in
the stored transcription orientation and aligned on the TF-MoDISco seqlet
centre before aggregation.

The default patterns are profile_pattern_{18,0,13,12,27}. One 5 x 4 inch,
editable-font PDF is written per pattern, together with per-seqlet source data,
mean/SEM profile arrays, an aggregate summary table, and a JSON audit record.

Interpretation: this tests model sensitivity to the motif sequence. Replacing
the motif by N is an in-silico intervention and is not direct experimental
evidence that the named motif or cognate TF causally changes pausing.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path("/mnt/HDD8TB/houruiyan/pausing_site")
HERE = Path(__file__).resolve().parent
PAUSENET_REPO = ROOT / "PauseNet"
DEFAULT_MODEL = ROOT / "2_train_model/train/hek293t_netseq/best_model.pt"
DEFAULT_TEST_DIR = ROOT / "data/HEK293T_NETseq/dataset/test"
DEFAULT_H5 = (
    ROOT
    / "3_model_explanation/TFMoDISco/results/hek293t_netseq/profile/"
    "profile_tfmodisco_patterns.h5"
)
DEFAULT_BRIDGE = (
    ROOT
    / "3_model_explanation/TFMoDISco/results/hek293t_netseq/profile/"
    "profile_selected_dataset_indices.npy"
)
DEFAULT_CORE_TABLE = (
    ROOT
    / "4_plot_figure/fig3/1.TF_atlas_profile/data/"
    "profile_pattern_pwm_cwm_summary.tsv"
)
DEFAULT_PATTERNS = (
    "profile_pattern_18",
    "profile_pattern_0",
    "profile_pattern_13",
    "profile_pattern_12",
    "profile_pattern_27",
)

INPUT_LENGTH = 2114
OUTPUT_LENGTH = 1000
CROP_START = (INPUT_LENGTH - OUTPUT_LENGTH) // 2
N_CODE = 4
CODE_TO_BASE = np.asarray(list("ACGTN"))
PATTERN_RE = re.compile(r"^(?:profile_)?pattern_(\d+)$")

COLOR_ORIGINAL = "#555555"
COLOR_ABLATED = "#0072B2"


@dataclass(frozen=True)
class RunningProfileStats:
    total: np.ndarray
    total_squared: np.ndarray
    n: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--patterns", nargs="+", default=list(DEFAULT_PATTERNS))
    parser.add_argument("--tfmodisco-h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--bridge", type=Path, default=DEFAULT_BRIDGE)
    parser.add_argument("--core-table", type=Path, default=DEFAULT_CORE_TABLE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--pausenet-repo", type=Path, default=PAUSENET_REPO)
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--plot-left", type=int, default=-500)
    parser.add_argument("--plot-right", type=int, default=500)
    parser.add_argument(
        "--max-seqlets",
        type=int,
        default=None,
        help="Process only the first N seqlets per pattern (for smoke tests).",
    )
    parser.add_argument(
        "--ablation-scope",
        choices=("core", "seqlet"),
        default="core",
        help="Mask the contribution-supported logo core or the complete 50-bp seqlet.",
    )
    parser.add_argument(
        "--write-png",
        action="store_true",
        help="Also write 300-dpi PNG files; PDF is the default server output.",
    )
    return parser.parse_args()


def normalize_pattern(value: str) -> tuple[str, int]:
    match = PATTERN_RE.fullmatch(value.strip())
    if match is None:
        raise ValueError(
            f"Unrecognized pattern {value!r}; use pattern_18 or profile_pattern_18"
        )
    index = int(match.group(1))
    return f"profile_pattern_{index}", index


def validate_args(args: argparse.Namespace) -> list[tuple[str, int]]:
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.max_seqlets is not None and args.max_seqlets < 1:
        raise ValueError("--max-seqlets must be positive")
    if args.plot_right <= args.plot_left:
        raise ValueError("--plot-right must exceed --plot-left")
    if args.plot_left < -OUTPUT_LENGTH or args.plot_right > OUTPUT_LENGTH:
        raise ValueError("Plot limits are outside the 1,000-bp output")

    patterns: list[tuple[str, int]] = []
    seen: set[str] = set()
    for raw in args.patterns:
        normalized = normalize_pattern(raw)
        if normalized[0] not in seen:
            patterns.append(normalized)
            seen.add(normalized[0])

    required = [
        args.tfmodisco_h5,
        args.bridge,
        args.core_table,
        args.model,
        args.pausenet_repo / "pausenet/evaluate.py",
        args.test_dir / "sequence_codes.npy",
        args.test_dir / "manifest.tsv",
    ]
    missing = [str(path) for path in required if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("Missing input file(s): " + ", ".join(missing))
    return patterns


def resolve_device(requested: str) -> torch.device:
    normalized = requested.strip().lower()
    if normalized == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        print(f"WARNING: {requested} unavailable; falling back to CPU", flush=True)
        return torch.device("cpu")
    return device


def load_manifest(path: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"array_index", "chrom", "output_start", "output_end", "strand"}
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
                "sample_id": row.get("sample_id", ""),
            }
    return records


def genomic_interval(
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


def decode_codes(codes: np.ndarray) -> str:
    values = np.asarray(codes, dtype=np.int64)
    safe = np.where((values >= 0) & (values < len(CODE_TO_BASE)), values, N_CODE)
    return "".join(CODE_TO_BASE[safe].tolist())


def load_core_definitions(path: Path) -> dict[str, dict[str, int]]:
    frame = pd.read_csv(path, sep="\t")
    required = {
        "pattern",
        "full_length",
        "display_start_0based",
        "display_end_exclusive",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Core table is missing columns: {sorted(missing)}")
    definitions: dict[str, dict[str, int]] = {}
    for row in frame.itertuples(index=False):
        definitions[str(row.pattern)] = {
            "full_length": int(row.full_length),
            "core_start": int(row.display_start_0based),
            "core_end": int(row.display_end_exclusive),
        }
    return definitions


def load_seqlets(
    h5_path: Path,
    bridge: np.ndarray,
    pattern_name: str,
    pattern_index: int,
    core_definition: dict[str, int],
    manifest: dict[int, dict[str, Any]],
    ablation_scope: str,
    max_seqlets: int | None,
) -> list[dict[str, Any]]:
    h5_pattern = f"pattern_{pattern_index}"
    group_path = f"pos_patterns/{h5_pattern}/seqlets"
    with h5py.File(h5_path, "r") as handle:
        if group_path not in handle:
            raise KeyError(f"{h5_path} does not contain {group_path}")
        group = handle[group_path]
        example_idx = np.asarray(group["example_idx"], dtype=np.int64)
        starts = np.asarray(group["start"], dtype=np.int64)
        ends = np.asarray(group["end"], dtype=np.int64)
        revcomp = np.asarray(group["is_revcomp"], dtype=bool)

    if np.any(example_idx < 0) or np.any(example_idx >= len(bridge)):
        raise IndexError(f"{pattern_name}: seqlet example_idx outside bridge")
    if not (len(example_idx) == len(starts) == len(ends) == len(revcomp)):
        raise ValueError(f"{pattern_name}: inconsistent seqlet array lengths")

    full_length = int(core_definition["full_length"])
    aligned_core_start = int(core_definition["core_start"])
    aligned_core_end = int(core_definition["core_end"])
    if not (0 <= aligned_core_start < aligned_core_end <= full_length):
        raise ValueError(f"{pattern_name}: invalid core definition {core_definition}")

    rows: list[dict[str, Any]] = []
    n_to_take = len(example_idx) if max_seqlets is None else min(max_seqlets, len(example_idx))
    for seqlet_index in range(n_to_take):
        start = int(starts[seqlet_index])
        end = int(ends[seqlet_index])
        if start < 0 or end > OUTPUT_LENGTH or end <= start:
            raise ValueError(
                f"{pattern_name} seqlet {seqlet_index}: invalid [{start}, {end})"
            )
        if end - start != full_length:
            raise ValueError(
                f"{pattern_name} seqlet {seqlet_index}: length {end-start}, "
                f"expected {full_length}"
            )

        if ablation_scope == "seqlet":
            local_start, local_end = 0, full_length
        elif bool(revcomp[seqlet_index]):
            # The core table is in motif orientation. Mirror it back into the
            # stored transcription-oriented input for reverse-complement seqlets.
            local_start = full_length - aligned_core_end
            local_end = full_length - aligned_core_start
        else:
            local_start, local_end = aligned_core_start, aligned_core_end

        dataset_index = int(bridge[int(example_idx[seqlet_index])])
        if dataset_index not in manifest:
            raise KeyError(f"Dataset index {dataset_index} is missing from manifest")
        rows.append(
            {
                "pattern": pattern_name,
                "pattern_index": pattern_index,
                "seqlet_index": seqlet_index,
                "example_idx": int(example_idx[seqlet_index]),
                "dataset_index": dataset_index,
                "seqlet_start_crop": start,
                "seqlet_end_crop": end,
                "seqlet_center_crop": (start + end) // 2,
                "is_revcomp": bool(revcomp[seqlet_index]),
                "ablation_start_crop": start + local_start,
                "ablation_end_crop": start + local_end,
                "core_start_aligned_0based": aligned_core_start,
                "core_end_aligned_exclusive": aligned_core_end,
            }
        )
    if not rows:
        raise ValueError(f"{pattern_name}: no seqlets selected")
    return rows


def predict_profiles(
    model: torch.nn.Module,
    sequences: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    batch = torch.from_numpy(np.asarray(sequences, dtype=np.int64)).long().to(device)
    with torch.no_grad():
        logits, predicted_log1p_counts = model(batch)
        log1p_counts = predicted_log1p_counts.float().reshape(-1)
        counts = torch.expm1(log1p_counts).clamp_min(0)
        probabilities = torch.softmax(logits.float(), dim=-1)
        profiles = probabilities * counts[:, None]
    return (
        profiles.cpu().numpy().astype(np.float32),
        counts.cpu().numpy().astype(np.float32),
        log1p_counts.cpu().numpy().astype(np.float32),
    )


def new_running_stats(width: int) -> RunningProfileStats:
    return RunningProfileStats(
        total=np.zeros(width, dtype=np.float64),
        total_squared=np.zeros(width, dtype=np.float64),
        n=np.zeros(width, dtype=np.int64),
    )


def update_running_stats(
    stats: RunningProfileStats,
    profiles: np.ndarray,
    centers_crop: np.ndarray,
    relative_positions: np.ndarray,
) -> None:
    for profile, center in zip(profiles, centers_crop):
        source = int(center) + relative_positions
        valid = (source >= 0) & (source < OUTPUT_LENGTH)
        values = np.asarray(profile[source[valid]], dtype=np.float64)
        stats.total[valid] += values
        stats.total_squared[valid] += values * values
        stats.n[valid] += 1


def finalize_stats(stats: RunningProfileStats) -> tuple[np.ndarray, np.ndarray]:
    mean = np.divide(
        stats.total,
        stats.n,
        out=np.full_like(stats.total, np.nan),
        where=stats.n > 0,
    )
    second_moment = np.divide(
        stats.total_squared,
        stats.n,
        out=np.full_like(stats.total_squared, np.nan),
        where=stats.n > 0,
    )
    variance = np.maximum(second_moment - mean * mean, 0.0)
    sem = np.divide(
        np.sqrt(variance),
        np.sqrt(stats.n),
        out=np.full_like(mean, np.nan),
        where=stats.n > 0,
    )
    return mean, sem


def chunks(values: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


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
    original_mean: np.ndarray,
    original_sem: np.ndarray,
    ablated_mean: np.ndarray,
    ablated_sem: np.ndarray,
    pattern: str,
    n_seqlets: int,
    median_delta_log1p: float,
    scope: str,
    output_pdf: Path,
    output_png: Path | None,
) -> None:
    configure_style()
    fig, axis = plt.subplots(figsize=(5, 4))
    axis.fill_between(
        x,
        np.maximum(0, original_mean - original_sem),
        original_mean + original_sem,
        color=COLOR_ORIGINAL,
        alpha=0.16,
        linewidth=0,
    )
    axis.fill_between(
        x,
        np.maximum(0, ablated_mean - ablated_sem),
        ablated_mean + ablated_sem,
        color=COLOR_ABLATED,
        alpha=0.16,
        linewidth=0,
    )
    axis.plot(x, original_mean, color=COLOR_ORIGINAL, lw=1.4, label="Original")
    axis.plot(x, ablated_mean, color=COLOR_ABLATED, lw=1.4, label="Motif-ablated")
    axis.axvline(0, color="#777777", lw=0.8, linestyle=(0, (3, 2)))
    axis.set_xlim(float(x[0]), float(x[-1]))
    axis.set_ylim(bottom=0)
    axis.set_xlabel(
        "Position relative to seqlet center (bp)\nTranscription direction: 5′ → 3′"
    )
    axis.set_ylabel("Mean predicted NET-seq signal")
    axis.set_title(pattern.replace("_", " "), pad=6)
    axis.text(
        0.03,
        0.96,
        f"n = {n_seqlets:,}\n{scope} → N\nmedian Δlog1p(count) = {median_delta_log1p:+.3f}",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
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


def process_pattern(
    pattern_name: str,
    pattern_index: int,
    seqlets: list[dict[str, Any]],
    sequence_codes: np.ndarray,
    manifest: dict[int, dict[str, Any]],
    model: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    relative_positions = np.arange(args.plot_left, args.plot_right, dtype=np.int64)
    original_stats = new_running_stats(len(relative_positions))
    ablated_stats = new_running_stats(len(relative_positions))
    source_rows: list[dict[str, Any]] = []

    for batch_number, batch_rows in enumerate(chunks(seqlets, args.batch_size), start=1):
        dataset_indices = np.asarray(
            [row["dataset_index"] for row in batch_rows], dtype=np.int64
        )
        original = np.asarray(sequence_codes[dataset_indices], dtype=np.int64)
        ablated = original.copy()
        for local_index, row in enumerate(batch_rows):
            start_input = CROP_START + int(row["ablation_start_crop"])
            end_input = CROP_START + int(row["ablation_end_crop"])
            ablated[local_index, start_input:end_input] = N_CODE

        original_profile, original_count, original_log1p = predict_profiles(
            model, original, device
        )
        ablated_profile, ablated_count, ablated_log1p = predict_profiles(
            model, ablated, device
        )
        centers = np.asarray(
            [row["seqlet_center_crop"] for row in batch_rows], dtype=np.int64
        )
        update_running_stats(
            original_stats, original_profile, centers, relative_positions
        )
        update_running_stats(
            ablated_stats, ablated_profile, centers, relative_positions
        )

        for position, row in enumerate(batch_rows):
            manifest_row = manifest[int(row["dataset_index"])]
            seqlet_start = int(row["seqlet_start_crop"])
            seqlet_end = int(row["seqlet_end_crop"])
            core_start = int(row["ablation_start_crop"])
            core_end = int(row["ablation_end_crop"])
            seqlet_genomic_start, seqlet_genomic_end = genomic_interval(
                manifest_row, seqlet_start, seqlet_end
            )
            core_genomic_start, core_genomic_end = genomic_interval(
                manifest_row, core_start, core_end
            )
            original_core = original[
                position,
                CROP_START + core_start : CROP_START + core_end,
            ]
            ablated_core = ablated[
                position,
                CROP_START + core_start : CROP_START + core_end,
            ]
            original_core_signal = float(
                original_profile[position, core_start:core_end].sum()
            )
            ablated_core_signal = float(
                ablated_profile[position, core_start:core_end].sum()
            )
            source_rows.append(
                {
                    **row,
                    "chrom": manifest_row["chrom"],
                    "gene_strand": manifest_row["strand"],
                    "gene_id": manifest_row["gene_id"],
                    "gene_name": manifest_row["gene_name"],
                    "transcript_id": manifest_row["transcript_id"],
                    "region_type": manifest_row["region_type"],
                    "sample_id": manifest_row["sample_id"],
                    "seqlet_genomic_start_hg19": seqlet_genomic_start,
                    "seqlet_genomic_end_hg19": seqlet_genomic_end,
                    "ablation_genomic_start_hg19": core_genomic_start,
                    "ablation_genomic_end_hg19": core_genomic_end,
                    "original_ablation_sequence": decode_codes(original_core),
                    "ablated_sequence": decode_codes(ablated_core),
                    "original_predicted_count": float(original_count[position]),
                    "ablated_predicted_count": float(ablated_count[position]),
                    "count_ratio_ablated_over_original": float(
                        ablated_count[position] / max(original_count[position], 1e-12)
                    ),
                    "original_predicted_log1p_count": float(original_log1p[position]),
                    "ablated_predicted_log1p_count": float(ablated_log1p[position]),
                    "delta_log1p_count_original_minus_ablated": float(
                        original_log1p[position] - ablated_log1p[position]
                    ),
                    "original_ablation_interval_signal": original_core_signal,
                    "ablated_ablation_interval_signal": ablated_core_signal,
                    "ablation_interval_signal_ratio_ablated_over_original": float(
                        ablated_core_signal / max(original_core_signal, 1e-12)
                    ),
                }
            )

        if batch_number % 10 == 0 or batch_number * args.batch_size >= len(seqlets):
            done = min(batch_number * args.batch_size, len(seqlets))
            print(f"  {pattern_name}: {done:,}/{len(seqlets):,}", flush=True)

    original_mean, original_sem = finalize_stats(original_stats)
    ablated_mean, ablated_sem = finalize_stats(ablated_stats)
    source = pd.DataFrame(source_rows)
    delta = source["delta_log1p_count_original_minus_ablated"].to_numpy(float)
    ratios = source["count_ratio_ablated_over_original"].to_numpy(float)

    pattern_dir = args.output_dir / pattern_name
    pattern_dir.mkdir(parents=True, exist_ok=True)
    prefix = pattern_dir / f"{pattern_name}_{args.ablation_scope}_N_ablation"
    source.to_csv(prefix.with_name(prefix.name + "_seqlets.tsv"), sep="\t", index=False)
    np.savez_compressed(
        prefix.with_name(prefix.name + "_profiles.npz"),
        relative_position_bp=relative_positions,
        original_mean=original_mean,
        original_sem=original_sem,
        ablated_mean=ablated_mean,
        ablated_sem=ablated_sem,
        n_per_position=original_stats.n,
    )
    draw_figure(
        relative_positions,
        original_mean,
        original_sem,
        ablated_mean,
        ablated_sem,
        pattern_name,
        len(source),
        float(np.median(delta)),
        args.ablation_scope,
        prefix.with_suffix(".pdf"),
        prefix.with_suffix(".png") if args.write_png else None,
    )
    return {
        "pattern": pattern_name,
        "pattern_index": pattern_index,
        "n_seqlets": int(len(source)),
        "n_unique_model_windows": int(source["dataset_index"].nunique()),
        "ablation_scope": args.ablation_scope,
        "median_delta_log1p_count_original_minus_ablated": float(np.median(delta)),
        "mean_delta_log1p_count_original_minus_ablated": float(np.mean(delta)),
        "fraction_delta_log1p_count_above_zero": float(np.mean(delta > 0)),
        "median_count_ratio_ablated_over_original": float(np.median(ratios)),
        "mean_count_ratio_ablated_over_original": float(np.mean(ratios)),
        "pdf": str(prefix.with_suffix(".pdf")),
        "seqlet_source_data": str(prefix.with_name(prefix.name + "_seqlets.tsv")),
        "profile_source_data": str(prefix.with_name(prefix.name + "_profiles.npz")),
    }


def main() -> int:
    args = parse_args()
    patterns = validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bridge = np.asarray(np.load(args.bridge), dtype=np.int64)
    if bridge.ndim != 1:
        raise ValueError(f"Bridge must be one-dimensional: {args.bridge}")
    sequence_codes = np.load(args.test_dir / "sequence_codes.npy", mmap_mode="r")
    if sequence_codes.ndim != 2 or sequence_codes.shape[1] != INPUT_LENGTH:
        raise ValueError(
            f"Expected sequence_codes (N,{INPUT_LENGTH}), got {sequence_codes.shape}"
        )
    manifest = load_manifest(args.test_dir / "manifest.tsv")
    core_definitions = load_core_definitions(args.core_table)

    device = resolve_device(args.device)
    sys.path.insert(0, str(args.pausenet_repo))
    from pausenet.evaluate import load_model  # noqa: E402

    model, checkpoint_config = load_model(args.model, device)
    summaries: list[dict[str, Any]] = []
    for pattern_name, pattern_index in patterns:
        short_name = f"pattern_{pattern_index}"
        if short_name not in core_definitions:
            raise KeyError(f"{short_name} is absent from {args.core_table}")
        seqlets = load_seqlets(
            args.tfmodisco_h5,
            bridge,
            pattern_name,
            pattern_index,
            core_definitions[short_name],
            manifest,
            args.ablation_scope,
            args.max_seqlets,
        )
        if max(row["dataset_index"] for row in seqlets) >= len(sequence_codes):
            raise IndexError(f"{pattern_name}: dataset_index exceeds sequence_codes.npy")
        print(
            f"START {pattern_name}: {len(seqlets):,} seqlets; "
            f"scope={args.ablation_scope}; device={device}",
            flush=True,
        )
        result = process_pattern(
            pattern_name,
            pattern_index,
            seqlets,
            sequence_codes,
            manifest,
            model,
            device,
            args,
        )
        summaries.append(result)
        print(
            f"DONE  {pattern_name}: median Δlog1p(count)="
            f"{result['median_delta_log1p_count_original_minus_ablated']:+.4f}",
            flush=True,
        )

    summary = pd.DataFrame(summaries)
    summary_path = args.output_dir / "profile_motif_core_N_ablation.summary.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    parameters = {
        "patterns": [name for name, _ in patterns],
        "n_patterns": len(patterns),
        "max_seqlets_per_pattern": args.max_seqlets,
        "ablation": {
            "scope": args.ablation_scope,
            "replacement": "A/C/G/T/N codes in selected interval replaced by N code 4",
            "core_definition": (
                "display_start_0based:display_end_exclusive in the profile PWM/CWM "
                "summary; mirrored for is_revcomp seqlets"
            ),
            "one_intervention_per_seqlet": True,
            "other_occurrences_in_same_model_window_unchanged": True,
        },
        "orientation": {
            "profile": "stored PauseNet transcription orientation (5-prime to 3-prime)",
            "reverse_complement_average": False,
            "is_revcomp_used_for_core_coordinate_transform": True,
            "is_revcomp_used_to_flip_predicted_profile": False,
        },
        "prediction": {
            "checkpoint": str(args.model.resolve()),
            "device": str(device),
            "formula": "softmax(profile_logits) * expm1(predicted_log1p_count)",
            "checkpoint_model_config": checkpoint_config.get("model", {}),
        },
        "plot": {
            "left_bp": args.plot_left,
            "right_bp_exclusive": args.plot_right,
            "center": "TF-MoDISco seqlet center",
            "bands": "standard error of the mean",
            "smoothing": "none",
            "pdf_only_unless_write_png": True,
        },
        "inputs": {
            "tfmodisco_h5": str(args.tfmodisco_h5.resolve()),
            "bridge": str(args.bridge.resolve()),
            "core_table": str(args.core_table.resolve()),
            "sequence_codes": str((args.test_dir / "sequence_codes.npy").resolve()),
            "manifest": str((args.test_dir / "manifest.tsv").resolve()),
        },
        "outputs": summaries,
    }
    parameter_path = args.output_dir / "profile_motif_core_N_ablation.parameters.json"
    parameter_path.write_text(
        json.dumps(parameters, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n" + summary.to_string(index=False), flush=True)
    print(f"Summary: {summary_path}", flush=True)
    print(f"Parameters: {parameter_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
