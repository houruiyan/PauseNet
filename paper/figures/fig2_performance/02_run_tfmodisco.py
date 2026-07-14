#!/usr/bin/env python3
"""Discover PauseNet sequence motifs from Figure 2 attribution arrays."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from modiscolite import io, tfmodisco, util


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run modisco-lite TF-MoDISco on PauseNet hypothetical contributions."
    )
    parser.add_argument("--task", required=True, choices=["count", "profile"])
    parser.add_argument("--data-dir", required=True, help="PauseNet dataset root.")
    parser.add_argument("--contrib-dir", required=True, help="Directory written by 01_compute_deepshap.py.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--attribution-suffix",
        help="Suffix after Lx4_ in attribution filenames. Auto-detected when omitted.",
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--sample-selection",
        choices=["all", "first", "random", "top_abs_contrib"],
        default="all",
    )
    parser.add_argument("--seed", type=int, default=2026062402)
    parser.add_argument("--sliding-window-size", type=int, default=21)
    parser.add_argument("--flank-size", type=int, default=10)
    parser.add_argument("--target-seqlet-fdr", type=float, default=0.2)
    parser.add_argument("--min-passing-windows-frac", type=float, default=0.03)
    parser.add_argument("--max-passing-windows-frac", type=float, default=0.2)
    parser.add_argument("--max-seqlets-per-metacluster", type=int, default=3000)
    parser.add_argument("--min-metacluster-size", type=int, default=100)
    parser.add_argument("--n-leiden-runs", type=int, default=5)
    parser.add_argument("--nearest-neighbors-to-compute", type=int, default=150)
    parser.add_argument("--final-min-cluster-size", type=int, default=20)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def one_hot_from_codes(codes: np.ndarray) -> np.ndarray:
    codes = np.asarray(codes, dtype=np.int16)
    one_hot = np.zeros((*codes.shape, 4), dtype=np.float32)
    rows, columns = np.nonzero(codes < 4)
    one_hot[rows, columns, codes[rows, columns]] = 1.0
    return one_hot


def discover_suffix(contrib_dir: Path, task: str, split: str, suffix: str | None) -> str:
    if suffix is not None:
        return suffix
    prefix = f"{task}_contribs_full_Lx4_{split}_"
    matches = sorted(contrib_dir.glob(f"{prefix}*.npy"))
    if len(matches) != 1:
        available = ", ".join(path.name for path in matches) or "none"
        raise ValueError(
            "Could not uniquely identify attribution output. "
            f"Found: {available}. Provide --attribution-suffix explicitly."
        )
    return matches[0].stem.removeprefix(f"{task}_contribs_full_Lx4_")


def select_indices(
    projected_contribs: np.ndarray,
    max_samples: int | None,
    selection: str,
    seed: int,
) -> tuple[np.ndarray, str]:
    n_samples = len(projected_contribs)
    if selection == "all" and (max_samples is None or max_samples >= n_samples):
        return np.arange(n_samples, dtype=np.int64), "all"
    if max_samples is None:
        raise ValueError("--max-samples is required unless --sample-selection all is used")
    if max_samples > n_samples:
        max_samples = n_samples
    if selection == "first":
        return np.arange(max_samples, dtype=np.int64), "first"
    if selection == "random":
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(n_samples, size=max_samples, replace=False)), "random"
    score = np.mean(np.abs(np.asarray(projected_contribs, dtype=np.float32)), axis=1)
    indices = np.argpartition(score, -max_samples)[-max_samples:]
    selected_as = "top_abs_contrib" if selection != "all" else "top_abs_contrib_due_to_max_samples"
    return np.sort(indices.astype(np.int64)), selected_as


def pattern_summary(patterns, sign: str) -> list[dict]:
    rows: list[dict] = []
    if patterns is None:
        return rows
    for index, pattern in enumerate(patterns):
        sequence = np.asarray(pattern.sequence)
        contributions = np.asarray(pattern.contrib_scores)
        hypothetical = np.asarray(pattern.hypothetical_contribs)
        ppm = sequence / np.maximum(sequence.sum(axis=1, keepdims=True), 1e-8)
        information_content = 2.0 + np.sum(ppm * np.log2(np.maximum(ppm, 1e-8)), axis=1)
        rows.append(
            {
                "sign": sign,
                "pattern_index": index,
                "n_seqlets": len(getattr(pattern, "seqlets", [])),
                "length": int(getattr(pattern, "length", sequence.shape[0])),
                "mean_abs_contrib": float(np.mean(np.abs(contributions))),
                "sum_abs_contrib": float(np.sum(np.abs(contributions))),
                "sum_contrib": float(np.sum(contributions)),
                "mean_abs_hypothetical": float(np.mean(np.abs(hypothetical))),
                "max_information_content": float(np.max(information_content)),
                "mean_information_content": float(np.mean(information_content)),
            }
        )
    return rows


def write_summary(path: Path, rows: list[dict]) -> None:
    fields = [
        "sign",
        "pattern_index",
        "n_seqlets",
        "length",
        "mean_abs_contrib",
        "sum_abs_contrib",
        "sum_contrib",
        "mean_abs_hypothetical",
        "max_information_content",
        "mean_information_content",
    ]
    with path.open("w") as handle:
        handle.write("\t".join(fields) + "\n")
        for row in rows:
            handle.write("\t".join(str(row[field]) for field in fields) + "\n")


def main() -> None:
    args = parse_args()
    started = time.time()
    contrib_dir = Path(args.contrib_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = discover_suffix(contrib_dir, args.task, args.split, args.attribution_suffix)

    codes_path = Path(args.data_dir) / args.split / "sequence_codes.npy"
    full_path = contrib_dir / f"{args.task}_contribs_full_Lx4_{suffix}.npy"
    projected_path = contrib_dir / f"{args.task}_contribs_projected_{suffix}.npy"
    codes = np.load(codes_path, mmap_mode="r")
    full_contribs = np.load(full_path, mmap_mode="r")
    projected_contribs = np.load(projected_path, mmap_mode="r")
    if codes.shape[:2] != full_contribs.shape[:2] or codes.shape != projected_contribs.shape:
        raise ValueError(
            f"Input shape mismatch: codes={codes.shape}, full={full_contribs.shape}, "
            f"projected={projected_contribs.shape}"
        )

    indices, effective_selection = select_indices(
        projected_contribs, args.max_samples, args.sample_selection, args.seed
    )
    np.save(output_dir / "selected_indices.npy", indices)
    one_hot = one_hot_from_codes(np.asarray(codes[indices], dtype=np.uint8))
    hypothetical = np.asarray(full_contribs[indices], dtype=np.float32)
    print(
        json.dumps(
            {
                "event": "loaded_inputs",
                "n_selected_samples": int(len(indices)),
                "selection": effective_selection,
                "one_hot_shape": list(one_hot.shape),
                "hypothetical_shape": list(hypothetical.shape),
            }
        ),
        flush=True,
    )

    positive_patterns, negative_patterns = tfmodisco.TFMoDISco(
        one_hot=one_hot,
        hypothetical_contribs=hypothetical,
        sliding_window_size=args.sliding_window_size,
        flank_size=args.flank_size,
        min_metacluster_size=args.min_metacluster_size,
        max_seqlets_per_metacluster=args.max_seqlets_per_metacluster,
        target_seqlet_fdr=args.target_seqlet_fdr,
        min_passing_windows_frac=args.min_passing_windows_frac,
        max_passing_windows_frac=args.max_passing_windows_frac,
        n_leiden_runs=args.n_leiden_runs,
        nearest_neighbors_to_compute=args.nearest_neighbors_to_compute,
        final_min_cluster_size=args.final_min_cluster_size,
        verbose=args.verbose,
    )

    h5_path = output_dir / f"{args.task}_tfmodisco_patterns.h5"
    io.save_hdf5(h5_path, positive_patterns, negative_patterns, window_size=args.sliding_window_size)
    for data_type in [
        util.MemeDataType.PFM,
        util.MemeDataType.CWM,
        util.MemeDataType.hCWM,
        util.MemeDataType.CWM_PFM,
        util.MemeDataType.hCWM_PFM,
    ]:
        output_path = output_dir / f"{args.task}_{data_type.value.replace('-', '_')}.meme"
        io.write_meme_from_h5(h5_path, datatype=data_type, output_filename=output_path, is_quiet=True)

    rows = pattern_summary(positive_patterns, "positive")
    rows += pattern_summary(negative_patterns, "negative")
    write_summary(output_dir / f"{args.task}_pattern_summary.tsv", rows)
    metadata = {
        "method": "modisco-lite TF-MoDISco with full Lx4 expected-gradient contributions",
        "task": args.task,
        "attribution_suffix": suffix,
        "effective_selection": effective_selection,
        "n_selected_samples": int(len(indices)),
        "input_length": int(codes.shape[1]),
        "full_contrib_path": str(full_path),
        "h5_output": str(h5_path),
        "n_positive_patterns": 0 if positive_patterns is None else len(positive_patterns),
        "n_negative_patterns": 0 if negative_patterns is None else len(negative_patterns),
        "elapsed_seconds": time.time() - started,
        "arguments": vars(args),
    }
    (output_dir / f"{args.task}_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
