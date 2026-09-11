#!/usr/bin/env python3
"""PauseNet count-motif knockdown with a matched negative background.

The four selected positive TF-MoDISco motifs are disrupted by shuffling the
displayed motif core without replacement.  Among repeated permutations, the
sequence with the lowest de novo PWM score is retained, so every edit keeps
the exact mononucleotide composition and GC content of its edited core.

The Background group is not an experimental TF-unbound set.  It is a matched
motif-negative computational control drawn from the same DeepSHAP-selected
test universe.  Target-pattern examples are excluded, and each retained motif
example is matched without replacement to a control with the same region,
the same pseudo-core position and length, identical core GC count,
central-1-kb GC within a configurable caliper, and a low target-PWM score on
both strands.  The identical PWM-minimizing shuffle is then applied to the
control pseudo-core.

The two-panel raincloud figure shows observed NET-seq counts above and
log1p(WT predicted count) - log1p(mutant predicted count) below.  A positive
delta means that sequence disruption lowers the PauseNet-predicted count.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path("/mnt/HDD8TB/houruiyan/pausing_site")
PAUSENET_REPO = ROOT / "PauseNet"
sys.path.insert(0, str(PAUSENET_REPO))
from pausenet.model import PauseNet, PauseNetConfig  # noqa: E402

H5_PATH = (
    ROOT
    / "3_model_explanation/TFMoDISco/results/hek293t_netseq/count/"
    "count_tfmodisco_patterns.h5"
)
BRIDGE_PATH = (
    ROOT
    / "3_model_explanation/TFMoDISco/results/hek293t_netseq/count/"
    "count_selected_dataset_indices.npy"
)
SELECTED_MANIFEST_PATH = (
    ROOT
    / "3_model_explanation/DeepSHAP/hek293t_netseq/selected_manifest.tsv"
)
MODEL_PATH = ROOT / "2_train_model/train/hek293t_netseq/best_model.pt"
TEST_DIR = ROOT / "data/HEK293T_NETseq/dataset/test"
PROFILES_NPZ = ROOT / "2_train_model/evaluate/hek293t_netseq/test_profiles.npz"
OUTDIR = ROOT / "4_plot_figure/fig3/4.count_motif_knockdown"

INPUT_LEN = 2114
CROP_LEN = 1000
CROP_START = (INPUT_LEN - CROP_LEN) // 2
CROP_END = CROP_START + CROP_LEN
COMPLEMENT_CODE = np.asarray([3, 2, 1, 0], dtype=np.int64)
CODE_TO_BASE = np.asarray(list("ACGT"))
REGION_ORDER = ("TSS", "5SS", "3SS", "TES")

MOTIF_COLOR = "#D81159"
BACKGROUND_COLOR = "#9A9A9A"
ZERO_COLOR = "#E6A0B5"
BACKGROUND_LABEL = "Background"


@dataclass(frozen=True)
class MotifSpec:
    tf_name: str
    pattern_index: int
    # Zero-based, end-exclusive coordinates in the full 50-position pattern.
    core_start: int
    core_end: int

    @property
    def pattern_name(self) -> str:
        return f"pattern_{self.pattern_index}"


# Exact displayed de novo cores from the count TF atlas:
# PATZ1 positions 1-11, ZNF610 2-11, SP1 1-9, SP2 4-13.
MOTIFS = [
    MotifSpec("PATZ1", 2, 17, 28),
    MotifSpec("ZNF610", 7, 23, 33),
    MotifSpec("SP1", 8, 20, 29),
    MotifSpec("SP2", 10, 24, 34),
]
MOTIF_BY_TF = {spec.tf_name: spec for spec in MOTIFS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, default=H5_PATH)
    parser.add_argument("--bridge", type=Path, default=BRIDGE_PATH)
    parser.add_argument(
        "--selected-manifest", type=Path, default=SELECTED_MANIFEST_PATH
    )
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--profiles-npz", type=Path, default=PROFILES_NPZ)
    parser.add_argument(
        "--sequence-codes", type=Path, default=TEST_DIR / "sequence_codes.npy"
    )
    parser.add_argument("--n-shuffles", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-seqlets", type=int, default=None)
    parser.add_argument(
        "--background-gc-caliper",
        type=float,
        default=0.02,
        help="Maximum central-1-kb GC-fraction difference for matching.",
    )
    parser.add_argument(
        "--background-pwm-quantile",
        type=float,
        default=0.50,
        help=(
            "Background pseudo-core score must be below this quantile of "
            "the corresponding positive seqlet scores on both strands."
        ),
    )
    parser.add_argument(
        "--minimum-match-fraction",
        type=float,
        default=0.90,
        help="Stop if fewer than this fraction of eligible positive units match.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTDIR)
    return parser.parse_args()


def configure_style() -> None:
    mpl.rcParams.update(
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
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def normalize_pwm(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 4:
        raise ValueError(f"Expected an Lx4 PWM, got {matrix.shape}")
    row_sum = matrix.sum(axis=1, keepdims=True)
    if np.any(row_sum <= 0):
        raise ValueError("PWM contains a position with zero total probability")
    return matrix / row_sum


def reverse_complement_codes(codes: np.ndarray) -> np.ndarray:
    return COMPLEMENT_CODE[np.asarray(codes, dtype=np.int64)[::-1]]


def codes_to_string(codes: np.ndarray) -> str:
    return "".join(CODE_TO_BASE[np.asarray(codes, dtype=np.int64)].tolist())


def base_counts(codes: np.ndarray) -> np.ndarray:
    return np.bincount(np.asarray(codes, dtype=np.int64), minlength=4)


def gc_fraction(codes: np.ndarray) -> float:
    codes = np.asarray(codes, dtype=np.int64)
    return float(np.mean((codes == 1) | (codes == 2)))


def pwm_log_odds(codes: np.ndarray, pwm: np.ndarray) -> float:
    positions = np.arange(len(codes), dtype=np.int64)
    probabilities = np.clip(pwm[positions, codes], 1e-8, 1.0)
    return float(np.log2(probabilities / 0.25).sum())


def pwm_log_odds_many(codes: np.ndarray, pwm: np.ndarray) -> np.ndarray:
    codes = np.asarray(codes, dtype=np.int64)
    if codes.ndim != 2 or codes.shape[1] != len(pwm):
        raise ValueError(f"Expected (N,{len(pwm)}) codes, got {codes.shape}")
    log_odds = np.log2(np.clip(pwm, 1e-8, 1.0) / 0.25)
    scores = np.zeros(len(codes), dtype=np.float64)
    for position in range(len(pwm)):
        scores += log_odds[position, codes[:, position]]
    return scores


def max_both_strand_pwm_score(codes: np.ndarray, pwm: np.ndarray) -> np.ndarray:
    forward = pwm_log_odds_many(codes, pwm)
    reverse = pwm_log_odds_many(COMPLEMENT_CODE[codes[:, ::-1]], pwm)
    return np.maximum(forward, reverse)


def lowest_scoring_shuffle(
    oriented_codes: np.ndarray,
    pwm: np.ndarray,
    n_shuffles: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, float, bool]:
    """Return the lowest-scoring composition-preserving permutation."""
    wt = np.asarray(oriented_codes, dtype=np.int64)
    wt_score = pwm_log_odds(wt, pwm)
    best = None
    best_score = np.inf
    for _ in range(n_shuffles):
        candidate = rng.permutation(wt)
        if np.array_equal(candidate, wt):
            continue
        score = pwm_log_odds(candidate, pwm)
        if score < best_score:
            best = candidate.copy()
            best_score = score
    if best is None:
        return wt.copy(), wt_score, wt_score, True
    return best, wt_score, float(best_score), False


def extract_region_class(sample_id: pd.Series) -> pd.Series:
    region = sample_id.astype(str).str.split("|", regex=False).str[2]
    invalid = ~region.isin(REGION_ORDER)
    if invalid.any():
        bad = sorted(region.loc[invalid].dropna().unique().tolist())
        raise ValueError(f"Cannot parse region class from sample_id: {bad[:5]}")
    return region


def validate_and_load_metadata(
    manifest_path: Path,
    bridge: np.ndarray,
    n_sequences: int,
    n_observed: int,
) -> pd.DataFrame:
    selected = pd.read_csv(manifest_path, sep="\t")
    required = {
        "contribution_array_index",
        "array_index",
        "sample_id",
        "strand",
        "profile_loss_mask",
    }
    missing = required.difference(selected.columns)
    if missing:
        raise KeyError(f"Selected manifest lacks columns: {sorted(missing)}")
    if len(selected) != len(bridge):
        raise ValueError(
            f"Bridge has {len(bridge)} rows but selected manifest has {len(selected)}"
        )
    contribution_index = selected["contribution_array_index"].to_numpy(dtype=int)
    if not np.array_equal(contribution_index, np.arange(len(selected))):
        raise ValueError("selected_manifest contribution_array_index is not 0..N-1")
    dataset_index = selected["array_index"].to_numpy(dtype=int)
    if not np.array_equal(dataset_index, bridge):
        raise ValueError("selected_manifest array_index does not equal the bridge")
    if np.min(bridge) < 0 or np.max(bridge) >= min(n_sequences, n_observed):
        raise IndexError("Bridge points outside sequence or observed-count arrays")
    if not selected["profile_loss_mask"].astype(bool).all():
        raise ValueError("Selected manifest contains profile_loss_mask == 0")
    selected = selected.copy()
    selected["region_class"] = extract_region_class(selected["sample_id"])
    return selected


def core_coordinates(
    seqlet_start: int,
    seqlet_end: int,
    reverse: bool,
    spec: MotifSpec,
) -> tuple[int, int]:
    if reverse:
        start_crop = seqlet_end - spec.core_end
        end_crop = seqlet_end - spec.core_start
    else:
        start_crop = seqlet_start + spec.core_start
        end_crop = seqlet_start + spec.core_end
    if not 0 <= start_crop < end_crop <= CROP_LEN:
        raise IndexError(
            f"{spec.tf_name}: core crop coordinates {start_crop}:{end_crop} invalid"
        )
    start_input = CROP_START + start_crop
    end_input = CROP_START + end_crop
    if not 0 <= start_input < end_input <= INPUT_LEN:
        raise IndexError(
            f"{spec.tf_name}: input coordinates {start_input}:{end_input} invalid"
        )
    return start_input, end_input


def load_positive_units(
    args: argparse.Namespace,
    bridge: np.ndarray,
    sequence_codes: np.ndarray,
    observed_counts: np.ndarray,
    selected: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], set[int], pd.DataFrame]:
    metadata = selected.set_index("array_index", drop=False)
    records: list[dict[str, object]] = []
    excluded_dataset_indices: set[int] = set()
    pwm_by_tf: dict[str, np.ndarray] = {}
    input_gc_cache: dict[int, float] = {}
    raw_counts: list[dict[str, object]] = []

    with h5py.File(args.h5, "r") as handle:
        for tf_order, spec in enumerate(MOTIFS):
            group = handle[f"pos_patterns/{spec.pattern_name}"]
            pwm_full = normalize_pwm(np.asarray(group["sequence"], dtype=np.float64))
            if spec.core_end > len(pwm_full):
                raise IndexError(f"{spec.tf_name}: core exceeds pattern width")
            pwm_core = pwm_full[spec.core_start : spec.core_end]
            pwm_by_tf[spec.tf_name] = pwm_core

            seqlets = group["seqlets"]
            examples_all = np.asarray(seqlets["example_idx"], dtype=np.int64)
            starts_all = np.asarray(seqlets["start"], dtype=np.int64)
            ends_all = np.asarray(seqlets["end"], dtype=np.int64)
            reverse_all = np.asarray(seqlets["is_revcomp"], dtype=bool)
            if np.any(examples_all < 0) or np.any(examples_all >= len(bridge)):
                raise IndexError(f"{spec.tf_name}: example_idx is outside bridge")
            excluded_dataset_indices.update(bridge[examples_all].tolist())

            use = len(examples_all)
            if args.max_seqlets is not None:
                use = min(use, args.max_seqlets)
            examples = examples_all[:use]
            starts = starts_all[:use]
            ends = ends_all[:use]
            reverse_flags = reverse_all[:use]
            dataset_indices = bridge[examples]

            for seqlet_index, (example_idx, dataset_idx, start, end, reverse) in enumerate(
                zip(
                    examples,
                    dataset_indices,
                    starts,
                    ends,
                    reverse_flags,
                    strict=True,
                )
            ):
                dataset_idx = int(dataset_idx)
                if int(end) - int(start) != len(pwm_full):
                    raise ValueError(
                        f"{spec.tf_name} seqlet {seqlet_index}: expected "
                        f"width {len(pwm_full)}, got {int(end) - int(start)}"
                    )
                core_start_input, core_end_input = core_coordinates(
                    int(start), int(end), bool(reverse), spec
                )
                codes = np.asarray(sequence_codes[dataset_idx], dtype=np.int64)
                if len(codes) != INPUT_LEN:
                    raise ValueError(f"Expected {INPUT_LEN}-bp input, got {len(codes)}")
                raw_core = codes[core_start_input:core_end_input]
                oriented_core = (
                    reverse_complement_codes(raw_core) if reverse else raw_core.copy()
                )
                if len(oriented_core) != len(pwm_core):
                    raise ValueError(f"{spec.tf_name}: edited core has wrong width")
                if dataset_idx not in input_gc_cache:
                    input_gc_cache[dataset_idx] = gc_fraction(codes[CROP_START:CROP_END])
                counts = base_counts(oriented_core)
                meta = metadata.loc[dataset_idx]
                records.append(
                    {
                        "tf_name": spec.tf_name,
                        "tf_order": tf_order,
                        "pattern": f"pos_pattern_{spec.pattern_index}",
                        "pattern_index": spec.pattern_index,
                        "seqlet_index": seqlet_index,
                        "example_idx": int(example_idx),
                        "dataset_idx": dataset_idx,
                        "seqlet_start_crop": int(start),
                        "seqlet_end_crop": int(end),
                        "is_revcomp": bool(reverse),
                        "core_start_input_0based": core_start_input,
                        "core_end_input_exclusive": core_end_input,
                        "core_length": len(oriented_core),
                        "core_count_A": int(counts[0]),
                        "core_count_C": int(counts[1]),
                        "core_count_G": int(counts[2]),
                        "core_count_T": int(counts[3]),
                        "core_gc_count": int(counts[1] + counts[2]),
                        "core_wt_motif_oriented": codes_to_string(oriented_core),
                        "pwm_score_wt": pwm_log_odds(oriented_core, pwm_core),
                        "input_gc": input_gc_cache[dataset_idx],
                        "region_class": str(meta["region_class"]),
                        "strand": str(meta["strand"]),
                        "observed_count": float(observed_counts[dataset_idx]),
                    }
                )

            raw_counts.append(
                {
                    "tf_name": spec.tf_name,
                    "n_seqlets_raw": use,
                    "n_unique_examples_raw": int(len(np.unique(dataset_indices))),
                }
            )

    positive = pd.DataFrame.from_records(records)
    if positive.empty:
        raise ValueError("No positive motif units were loaded")
    positive = (
        positive.sort_values(
            ["tf_order", "dataset_idx", "pwm_score_wt"],
            ascending=[True, True, False],
        )
        .drop_duplicates(["tf_name", "dataset_idx"], keep="first")
        .sort_values(["tf_order", "seqlet_index"])
        .reset_index(drop=True)
    )
    positive["positive_row_id"] = np.arange(len(positive), dtype=int)
    positive["core_mutable"] = positive["core_wt_motif_oriented"].map(
        lambda value: len(set(value)) >= 2
    )
    immutable = positive.loc[~positive["core_mutable"]].copy()
    if not immutable.empty:
        print(
            "[positive] removing immutable cores: "
            + ", ".join(
                f"{tf}={count}"
                for tf, count in immutable.groupby("tf_name").size().items()
            ),
            flush=True,
        )
    positive = positive.loc[positive["core_mutable"]].reset_index(drop=True)
    positive["positive_row_id"] = np.arange(len(positive), dtype=int)

    diagnostic = pd.DataFrame(raw_counts)
    deduplicated = positive.groupby("tf_name", sort=False).size().rename("n_mutable_unique")
    diagnostic = diagnostic.merge(deduplicated, on="tf_name", how="left")
    return positive, pwm_by_tf, excluded_dataset_indices, diagnostic


def calculate_candidate_gc(
    sequence_codes: np.ndarray,
    dataset_indices: np.ndarray,
    batch_size: int = 1024,
) -> np.ndarray:
    values = np.empty(len(dataset_indices), dtype=np.float64)
    for start in range(0, len(dataset_indices), batch_size):
        end = min(len(dataset_indices), start + batch_size)
        codes = np.asarray(
            sequence_codes[dataset_indices[start:end], CROP_START:CROP_END],
            dtype=np.int64,
        )
        values[start:end] = np.mean((codes == 1) | (codes == 2), axis=1)
    return values


def build_candidate_table(
    selected: pd.DataFrame,
    excluded_dataset_indices: set[int],
    sequence_codes: np.ndarray,
) -> pd.DataFrame:
    candidate = selected.loc[
        ~selected["array_index"].isin(excluded_dataset_indices)
    ].copy()
    candidate = candidate.drop_duplicates("array_index").reset_index(drop=True)
    candidate_ids = candidate["array_index"].to_numpy(dtype=int)
    candidate["input_gc"] = calculate_candidate_gc(sequence_codes, candidate_ids)
    if candidate.empty:
        raise ValueError("No background candidates remain after pattern exclusion")
    print(
        f"[background] selected universe={len(selected)}, "
        f"excluded target-pattern examples={len(excluded_dataset_indices)}, "
        f"candidate pool={len(candidate)}",
        flush=True,
    )
    return candidate


def candidate_options_for_positive(
    row: pd.Series,
    candidate_by_group: dict[str, pd.DataFrame],
    sequence_codes: np.ndarray,
    pwm: np.ndarray,
    pwm_cutoff: float,
    gc_caliper: float,
) -> tuple[list[tuple[int, float, int, float]], list[tuple[int, float, int, float]]]:
    group_key = str(row["region_class"])
    pool = candidate_by_group.get(group_key)
    if pool is None or pool.empty:
        return [], []
    gc_distance = np.abs(pool["input_gc"].to_numpy(dtype=float) - float(row["input_gc"]))
    within_gc = gc_distance <= gc_caliper + 1e-12
    if not np.any(within_gc):
        return [], []

    pool = pool.loc[within_gc].copy()
    gc_distance = gc_distance[within_gc]
    candidate_ids = pool["array_index"].to_numpy(dtype=int)
    start = int(row["core_start_input_0based"])
    end = int(row["core_end_input_exclusive"])
    cores = np.asarray(sequence_codes[candidate_ids, start:end], dtype=np.int64)
    canonical = np.all((cores >= 0) & (cores < 4), axis=1)
    mutable = np.any(cores != cores[:, :1], axis=1)
    counts = np.stack([(cores == base).sum(axis=1) for base in range(4)], axis=1)
    target_counts = np.asarray(
        [
            row["core_count_A"],
            row["core_count_C"],
            row["core_count_G"],
            row["core_count_T"],
        ],
        dtype=int,
    )
    same_gc_count = (counts[:, 1] + counts[:, 2]) == int(row["core_gc_count"])
    exact_composition = np.all(counts == target_counts[None, :], axis=1)
    max_pwm_score = np.full(len(cores), np.inf, dtype=np.float64)
    if np.any(canonical):
        max_pwm_score[canonical] = max_both_strand_pwm_score(
            cores[canonical], pwm
        )
    low_pwm = max_pwm_score < pwm_cutoff
    valid = canonical & mutable & same_gc_count & low_pwm
    if not np.any(valid):
        return [], []

    l1_distance = np.abs(counts - target_counts[None, :]).sum(axis=1)
    options = [
        (
            int(candidate_ids[index]),
            float(gc_distance[index]),
            int(l1_distance[index]),
            float(max_pwm_score[index]),
        )
        for index in np.flatnonzero(valid)
    ]
    options.sort(key=lambda item: (item[2], item[1], item[0]))
    exact = [item for item in options if item[2] == 0]
    return exact, options


def match_backgrounds(
    args: argparse.Namespace,
    positive: pd.DataFrame,
    candidate: pd.DataFrame,
    sequence_codes: np.ndarray,
    pwm_by_tf: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, dict[str, float]]:
    pwm_cutoffs = {
        tf_name: float(
            np.quantile(
                positive.loc[positive["tf_name"] == tf_name, "pwm_score_wt"],
                args.background_pwm_quantile,
            )
        )
        for tf_name in MOTIF_BY_TF
    }
    candidate_by_group = {
        str(region): group.reset_index(drop=True)
        for region, group in candidate.groupby("region_class", sort=False)
    }

    option_records: list[dict[str, object]] = []
    for row in positive.itertuples(index=False):
        series = pd.Series(row._asdict())
        exact, relaxed = candidate_options_for_positive(
            row=series,
            candidate_by_group=candidate_by_group,
            sequence_codes=sequence_codes,
            pwm=pwm_by_tf[row.tf_name],
            pwm_cutoff=pwm_cutoffs[row.tf_name],
            gc_caliper=args.background_gc_caliper,
        )
        option_records.append(
            {
                "positive_row_id": int(row.positive_row_id),
                "tf_name": row.tf_name,
                "exact_options": exact,
                "relaxed_options": relaxed,
                "n_exact_options": len(exact),
                "n_relaxed_options": len(relaxed),
            }
        )

    option_records.sort(
        key=lambda item: (
            item["n_exact_options"],
            item["n_relaxed_options"],
            MOTIF_BY_TF[str(item["tf_name"])].pattern_index,
            item["positive_row_id"],
        )
    )
    used: set[int] = set()
    matches: list[dict[str, object]] = []
    unmatched: list[int] = []
    for item in option_records:
        chosen = next(
            (option for option in item["exact_options"] if option[0] not in used),
            None,
        )
        match_mode = "exact_composition"
        if chosen is None:
            chosen = next(
                (
                    option
                    for option in item["relaxed_options"]
                    if option[0] not in used
                ),
                None,
            )
            match_mode = "exact_core_gc_minimum_l1"
        if chosen is None:
            unmatched.append(int(item["positive_row_id"]))
            continue
        background_dataset_idx, gc_distance, l1_distance, max_pwm_score = chosen
        used.add(background_dataset_idx)
        matches.append(
            {
                "positive_row_id": int(item["positive_row_id"]),
                "background_dataset_idx": background_dataset_idx,
                "matched_to_tf": str(item["tf_name"]),
                "match_mode": match_mode,
                "input_gc_absolute_difference": gc_distance,
                "core_composition_l1_difference": l1_distance,
                "background_max_both_strand_pwm_score": max_pwm_score,
                "background_pwm_cutoff": pwm_cutoffs[str(item["tf_name"])],
                "n_exact_options": int(item["n_exact_options"]),
                "n_relaxed_options": int(item["n_relaxed_options"]),
            }
        )

    match_frame = pd.DataFrame.from_records(matches)
    matched_fraction = len(match_frame) / len(positive)
    print(
        f"[background] matched={len(match_frame)}/{len(positive)} "
        f"({matched_fraction:.1%}), exact="
        f"{int((match_frame['match_mode'] == 'exact_composition').sum()) if len(match_frame) else 0}, "
        f"relaxed="
        f"{int((match_frame['match_mode'] != 'exact_composition').sum()) if len(match_frame) else 0}",
        flush=True,
    )
    if unmatched:
        counts = positive.loc[
            positive["positive_row_id"].isin(unmatched), "tf_name"
        ].value_counts()
        print(
            "[background] unmatched positive units: "
            + ", ".join(f"{tf}={count}" for tf, count in counts.items()),
            flush=True,
        )
    if matched_fraction < args.minimum_match_fraction:
        raise RuntimeError(
            f"Only {matched_fraction:.1%} of positive units matched; required "
            f"{args.minimum_match_fraction:.1%}. No thresholds were relaxed."
        )
    match_frame = match_frame.sort_values(
        ["positive_row_id"]
    ).reset_index(drop=True)
    match_frame["match_id"] = [f"match_{index:04d}" for index in range(len(match_frame))]
    return match_frame, pwm_cutoffs


def load_model(model_path: Path, device: torch.device) -> PauseNet:
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_config = checkpoint.get("config", {}).get("model", {})
    model = PauseNet(PauseNetConfig(**model_config)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict_counts(
    model: PauseNet,
    codes: np.ndarray,
    device: torch.device,
    batch_size: int,
    label: str,
) -> np.ndarray:
    counts = np.zeros(len(codes), dtype=np.float64)
    started = time.time()
    with torch.no_grad():
        for start in range(0, len(codes), batch_size):
            end = min(len(codes), start + batch_size)
            batch = torch.from_numpy(codes[start:end]).long().to(device)
            _profile_logits, log1p_counts = model(batch)
            predicted = np.expm1(log1p_counts.float().cpu().numpy())
            counts[start:end] = np.maximum(predicted, 0.0)
            if end == len(codes) or start == 0:
                print(
                    f"[{label}] {end}/{len(codes)} "
                    f"({time.time() - started:.1f}s)",
                    flush=True,
                )
    return counts


def build_matched_mutations(
    args: argparse.Namespace,
    positive: pd.DataFrame,
    matches: pd.DataFrame,
    selected: pd.DataFrame,
    sequence_codes: np.ndarray,
    observed_counts: np.ndarray,
    pwm_by_tf: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    positive_by_id = positive.set_index("positive_row_id", drop=False)
    metadata = selected.set_index("array_index", drop=False)
    child_seeds = np.random.SeedSequence(args.seed).spawn(2 * len(matches))
    wt_sequences: list[np.ndarray] = []
    mutant_sequences: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    retained_matches: list[dict[str, object]] = []

    for pair_index, match in enumerate(matches.itertuples(index=False)):
        positive_row = positive_by_id.loc[int(match.positive_row_id)]
        tf_name = str(positive_row["tf_name"])
        spec = MOTIF_BY_TF[tf_name]
        pwm = pwm_by_tf[tf_name]
        positive_dataset_idx = int(positive_row["dataset_idx"])
        background_dataset_idx = int(match.background_dataset_idx)
        start = int(positive_row["core_start_input_0based"])
        end = int(positive_row["core_end_input_exclusive"])
        reverse = bool(positive_row["is_revcomp"])

        pair_sequences: list[np.ndarray] = []
        pair_mutants: list[np.ndarray] = []
        pair_rows: list[dict[str, object]] = []
        pair_failed = False
        for role_index, (role, dataset_idx) in enumerate(
            (("positive", positive_dataset_idx), ("background", background_dataset_idx))
        ):
            codes = np.asarray(sequence_codes[dataset_idx], dtype=np.int64).copy()
            raw_core = codes[start:end]
            oriented_core = (
                reverse_complement_codes(raw_core) if reverse else raw_core.copy()
            )
            rng = np.random.default_rng(child_seeds[2 * pair_index + role_index])
            mutant_oriented, wt_score, mutant_score, failed = lowest_scoring_shuffle(
                oriented_core, pwm, args.n_shuffles, rng
            )
            if failed:
                pair_failed = True
                break
            mutant_raw = (
                reverse_complement_codes(mutant_oriented)
                if reverse
                else mutant_oriented
            )
            mutant_codes = codes.copy()
            mutant_codes[start:end] = mutant_raw
            counts = base_counts(oriented_core)
            meta = metadata.loc[dataset_idx]

            is_background = role == "background"
            pair_sequences.append(codes)
            pair_mutants.append(mutant_codes)
            pair_rows.append(
                {
                    "match_id": str(match.match_id),
                    "display_group": BACKGROUND_LABEL if is_background else tf_name,
                    "tf_name": BACKGROUND_LABEL if is_background else tf_name,
                    "matched_to_tf": tf_name,
                    "class": "background" if is_background else "positive",
                    "pattern": (
                        "matched_motif_negative_background"
                        if is_background
                        else f"pos_pattern_{spec.pattern_index}"
                    ),
                    "pattern_index": -1 if is_background else spec.pattern_index,
                    "seqlet_index": -1 if is_background else int(positive_row["seqlet_index"]),
                    "example_idx": int(meta["contribution_array_index"]),
                    "dataset_idx": dataset_idx,
                    "positive_dataset_idx": positive_dataset_idx,
                    "background_dataset_idx": background_dataset_idx,
                    "is_revcomp": reverse,
                    "core_start_input_0based": start,
                    "core_end_input_exclusive": end,
                    "core_length": end - start,
                    "core_count_A": int(counts[0]),
                    "core_count_C": int(counts[1]),
                    "core_count_G": int(counts[2]),
                    "core_count_T": int(counts[3]),
                    "core_gc_count": int(counts[1] + counts[2]),
                    "core_wt_motif_oriented": codes_to_string(oriented_core),
                    "core_mut_motif_oriented": codes_to_string(mutant_oriented),
                    "pwm_score_wt": wt_score,
                    "pwm_score_mut": mutant_score,
                    "delta_pwm_score": mutant_score - wt_score,
                    "shuffle_failed": failed,
                    "input_gc": gc_fraction(codes[CROP_START:CROP_END]),
                    "region_class": str(meta["region_class"]),
                    "strand": str(meta["strand"]),
                    "match_mode": str(match.match_mode),
                    "input_gc_absolute_difference": float(
                        match.input_gc_absolute_difference
                    ),
                    "core_composition_l1_difference": int(
                        match.core_composition_l1_difference
                    ),
                    "background_max_both_strand_pwm_score": float(
                        match.background_max_both_strand_pwm_score
                    ),
                    "background_pwm_cutoff": float(match.background_pwm_cutoff),
                    "observed_count": float(observed_counts[dataset_idx]),
                }
            )

        if pair_failed:
            print(f"[mutation] skipped unshufflable {match.match_id}", flush=True)
            continue
        wt_sequences.extend(pair_sequences)
        mutant_sequences.extend(pair_mutants)
        rows.extend(pair_rows)
        retained_matches.append(match._asdict())

    if not rows:
        raise RuntimeError("No complete positive/background mutation pairs remain")
    print(
        f"[mutation] retained {len(retained_matches)} matched pairs "
        f"({2 * len(retained_matches)} sequences)",
        flush=True,
    )
    return (
        np.stack(wt_sequences),
        np.stack(mutant_sequences),
        pd.DataFrame.from_records(rows),
        pd.DataFrame.from_records(retained_matches),
    )


def add_raincloud(
    ax: plt.Axes,
    arrays: list[np.ndarray],
    positions: np.ndarray,
    colors: list[str],
    rng: np.random.Generator,
    max_points_per_group: int = 180,
) -> None:
    """Draw left-half violins, right-side raw points and narrow boxplots."""
    for values, position, color in zip(arrays, positions, colors, strict=True):
        violin = ax.violinplot(
            [values],
            positions=[position],
            widths=0.72,
            showmeans=False,
            showmedians=False,
            showextrema=False,
            bw_method=0.25,
        )
        body = violin["bodies"][0]
        vertices = body.get_paths()[0].vertices
        vertices[:, 0] = np.minimum(vertices[:, 0], position)
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_linewidth(0.7)
        body.set_alpha(0.28)

        shown = values
        if len(values) > max_points_per_group:
            shown = rng.choice(values, size=max_points_per_group, replace=False)
        jitter = position + 0.08 + rng.uniform(0.0, 0.18, size=len(shown))
        ax.scatter(
            jitter,
            shown,
            s=5.0,
            color=color,
            alpha=0.25,
            linewidths=0,
            rasterized=False,
            zorder=2,
        )

        boxes = ax.boxplot(
            [values],
            positions=[position],
            widths=0.12,
            patch_artist=True,
            showfliers=False,
            boxprops={
                "facecolor": "white",
                "edgecolor": color,
                "linewidth": 0.9,
            },
            medianprops={"color": "#202020", "linewidth": 1.1},
            whiskerprops={"color": color, "linewidth": 0.8},
            capprops={"color": color, "linewidth": 0.8},
        )
        boxes["boxes"][0].set_zorder(3)


def draw_figure(frame: pd.DataFrame, output: Path) -> None:
    configure_style()
    rng = np.random.default_rng(20260904)
    labels = [spec.tf_name for spec in MOTIFS] + [BACKGROUND_LABEL]
    positions = np.arange(1, len(labels) + 1)
    colors = [MOTIF_COLOR] * len(MOTIFS) + [BACKGROUND_COLOR]
    observed = [
        np.log10(
            frame.loc[frame["display_group"] == label, "observed_count"].to_numpy()
            + 1.0
        )
        for label in labels
    ]
    deltas = [
        frame.loc[
            frame["display_group"] == label,
            "delta_predicted_log1p_count",
        ].to_numpy()
        for label in labels
    ]
    if any(len(values) == 0 for values in observed + deltas):
        raise ValueError("At least one figure group is empty")

    fig, (ax_observed, ax_delta) = plt.subplots(
        2,
        1,
        figsize=(5, 4),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.12},
    )

    add_raincloud(ax_observed, observed, positions, colors, rng)
    ax_observed.set_ylabel("Observed NET-seq\ncounts")
    maximum_log = max(float(np.max(values)) for values in observed)
    maximum_tick = max(1, int(np.ceil(maximum_log)))
    ticks = np.arange(0, maximum_tick + 1, dtype=int)
    ax_observed.set_yticks(ticks)
    ax_observed.set_yticklabels(
        ["0"] + [rf"$10^{{{value}}}$" for value in ticks[1:]]
    )
    ax_observed.set_ylim(-0.08, maximum_tick + 0.18)

    add_raincloud(ax_delta, deltas, positions, colors, rng)
    ax_delta.axhline(0, color=ZERO_COLOR, linewidth=0.8, linestyle=(0, (2, 2)))
    ax_delta.set_ylabel("Δ predicted\nlog1p(counts)")
    all_delta = np.concatenate(deltas)
    lower = float(np.nanpercentile(all_delta, 0.5))
    upper = float(np.nanpercentile(all_delta, 99.5))
    padding = max(0.05, 0.10 * (upper - lower))
    ax_delta.set_ylim(lower - padding, upper + padding)
    ax_delta.set_xticks(positions)
    ax_delta.set_xticklabels(labels, rotation=45, ha="right")

    for ax in (ax_observed, ax_delta):
        ax.set_xlim(0.45, len(labels) + 0.55)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)
        ax.yaxis.label.set_size(10)
        ax.tick_params(axis="y", labelsize=8, length=3)
        ax.tick_params(axis="x", length=3)

    fig.subplots_adjust(left=0.17, right=0.99, top=0.97, bottom=0.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def build_outputs(
    frame: pd.DataFrame,
    retained_matches: pd.DataFrame,
    diagnostic: pd.DataFrame,
    pwm_cutoffs: dict[str, float],
    background_pwm_quantile: float,
    output_dir: Path,
) -> list[Path]:
    seqlet_output = output_dir / "count_motif_knockdown_seqlets.tsv.gz"
    summary_output = output_dir / "count_motif_knockdown_summary.tsv"
    match_output = output_dir / "count_motif_knockdown_matched_pairs.tsv.gz"
    qc_output = output_dir / "count_motif_knockdown_background_qc.tsv"
    paired_output = output_dir / "count_motif_knockdown_paired_effects.tsv"
    figure_output = output_dir / "count_motif_knockdown_raincloud.pdf"

    frame.to_csv(seqlet_output, sep="\t", index=False, compression="gzip")
    retained_matches.to_csv(match_output, sep="\t", index=False, compression="gzip")

    labels = [spec.tf_name for spec in MOTIFS] + [BACKGROUND_LABEL]
    summary = (
        frame.groupby("display_group", sort=False)
        .agg(
            n_units=("match_id", "size"),
            n_unique_examples=("dataset_idx", "nunique"),
            median_observed_count=("observed_count", "median"),
            mean_delta_predicted_log1p_count=(
                "delta_predicted_log1p_count",
                "mean",
            ),
            median_delta_predicted_log1p_count=(
                "delta_predicted_log1p_count",
                "median",
            ),
            fraction_delta_positive=(
                "delta_predicted_log1p_count",
                lambda values: float(np.mean(np.asarray(values) > 0)),
            ),
            mean_delta_pwm_score=("delta_pwm_score", "mean"),
        )
        .reindex(labels)
        .reset_index()
    )
    summary.to_csv(summary_output, sep="\t", index=False)

    motif_effect = frame.loc[
        frame["class"] == "positive",
        ["match_id", "matched_to_tf", "delta_predicted_log1p_count"],
    ].rename(columns={"delta_predicted_log1p_count": "motif_delta"})
    background_effect = frame.loc[
        frame["class"] == "background",
        ["match_id", "delta_predicted_log1p_count"],
    ].rename(columns={"delta_predicted_log1p_count": "background_delta"})
    paired = motif_effect.merge(background_effect, on="match_id", validate="one_to_one")
    paired["difference_in_differences"] = (
        paired["motif_delta"] - paired["background_delta"]
    )
    paired.to_csv(paired_output, sep="\t", index=False)

    match_qc = (
        retained_matches.groupby("matched_to_tf", sort=False)
        .agg(
            n_matched=("match_id", "size"),
            n_exact_composition=(
                "match_mode",
                lambda values: int(np.sum(values == "exact_composition")),
            ),
            n_gc_only_fallback=(
                "match_mode",
                lambda values: int(np.sum(values != "exact_composition")),
            ),
            median_input_gc_difference=("input_gc_absolute_difference", "median"),
            maximum_input_gc_difference=("input_gc_absolute_difference", "max"),
            median_background_max_pwm_score=(
                "background_max_both_strand_pwm_score",
                "median",
            ),
        )
        .reset_index()
        .rename(columns={"matched_to_tf": "tf_name"})
    )
    diagnostic = diagnostic.merge(match_qc, on="tf_name", how="left")
    diagnostic["background_pwm_score_cutoff"] = diagnostic["tf_name"].map(
        pwm_cutoffs
    )
    diagnostic["background_pwm_cutoff_quantile"] = background_pwm_quantile
    diagnostic.to_csv(qc_output, sep="\t", index=False)

    draw_figure(frame, figure_output)
    print(summary.to_string(index=False), flush=True)
    paired_summary = paired.groupby("matched_to_tf", sort=False).agg(
        n_pairs=("match_id", "size"),
        median_motif_delta=("motif_delta", "median"),
        median_background_delta=("background_delta", "median"),
        median_difference_in_differences=("difference_in_differences", "median"),
        fraction_difference_positive=(
            "difference_in_differences",
            lambda values: float(np.mean(np.asarray(values) > 0)),
        ),
    )
    print("[paired effects]\n" + paired_summary.to_string(), flush=True)
    return [
        figure_output,
        summary_output,
        seqlet_output,
        match_output,
        qc_output,
        paired_output,
    ]


def main() -> int:
    args = parse_args()
    if args.n_shuffles <= 0:
        raise ValueError("--n-shuffles must be positive")
    if args.max_seqlets is not None and args.max_seqlets <= 0:
        raise ValueError("--max-seqlets must be positive")
    if not 0 <= args.background_gc_caliper <= 1:
        raise ValueError("--background-gc-caliper must be between 0 and 1")
    if not 0 < args.background_pwm_quantile < 1:
        raise ValueError("--background-pwm-quantile must be between 0 and 1")
    if not 0 < args.minimum_match_fraction <= 1:
        raise ValueError("--minimum-match-fraction must be in (0,1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    bridge = np.asarray(np.load(args.bridge), dtype=np.int64)
    sequence_codes = np.load(args.sequence_codes, mmap_mode="r")
    with np.load(args.profiles_npz, allow_pickle=False) as profiles:
        observed_counts = np.asarray(profiles["observed_counts"], dtype=np.float64)
    selected = validate_and_load_metadata(
        args.selected_manifest,
        bridge,
        len(sequence_codes),
        len(observed_counts),
    )
    positive, pwm_by_tf, excluded, diagnostic = load_positive_units(
        args,
        bridge,
        sequence_codes,
        observed_counts,
        selected,
    )
    candidate = build_candidate_table(selected, excluded, sequence_codes)
    matches, pwm_cutoffs = match_backgrounds(
        args,
        positive,
        candidate,
        sequence_codes,
        pwm_by_tf,
    )
    wt_codes, mutant_codes, frame, retained_matches = build_matched_mutations(
        args,
        positive,
        matches,
        selected,
        sequence_codes,
        observed_counts,
        pwm_by_tf,
    )

    requested = torch.device(args.device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        print(f"[device] {args.device} unavailable; using CPU", flush=True)
        requested = torch.device("cpu")
    print(f"[device] {requested}", flush=True)
    model = load_model(args.model, requested)
    predicted_wt = predict_counts(model, wt_codes, requested, args.batch_size, "WT")
    predicted_mutant = predict_counts(
        model, mutant_codes, requested, args.batch_size, "MUT"
    )
    frame["predicted_count_wt"] = predicted_wt
    frame["predicted_count_mut"] = predicted_mutant
    frame["delta_predicted_log1p_count"] = np.log1p(predicted_wt) - np.log1p(
        predicted_mutant
    )

    outputs = build_outputs(
        frame,
        retained_matches,
        diagnostic,
        pwm_cutoffs,
        args.background_pwm_quantile,
        args.output_dir,
    )
    for output in outputs:
        print(f"[out] {output}", flush=True)
    print(f"[done] elapsed {time.time() - started:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
