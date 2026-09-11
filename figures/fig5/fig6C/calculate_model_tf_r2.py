#!/usr/bin/env python3
"""Fit one independent PauseNet-plus-TF linear model per TF bigWig.

Required inputs are deliberately generic:

    * one PauseNet evaluation NPZ;
    * OLS training chromosome(s);
    * disjoint OLS test chromosome(s);
    * one or more TF=bigWig tracks.

For each TF, the script fits:

    model only:  y ~ x_model
    TF only:     y ~ x_tf
    model + TF:  y ~ x_model + x_tf + x_model:x_tf

where:

    y       = log1p(observed_counts)
    x_model = log1p(predicted_counts)
    x_tf    = mean(log1p(max(bigWig signal, 0))) in the window

The interaction follows the ProCapNet linear-model analysis and can be removed
with --no-interaction. By default, each TF is fitted separately. Use
--fit-all-tfs to additionally fit all supplied TFs together, or combine it
with --only-all-tfs to output only the model-only and all-TF models. The
script has no cell-type pairing logic.

Example
-------
python 3.calculate_model_tf_r2.py \
  --npz /path/test_profiles.npz \
  --train-chr chr8 \
  --test-chr chr9 \
  --tf CHD2=/path/CHD2.bigWig \
  --tf SWI=/path/SWI.bigWig \
  --threads 4 \
  --output-tsv k562_tf_r2.tsv

Outputs
-------
The requested TSV
    Three rows per TF: model_only, tf_only, and model_plus_tf.
"""

from __future__ import annotations

import argparse
import csv
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pyBigWig


HG19_CHR1_LENGTH = 249_250_621
HG38_CHR1_LENGTH = 248_956_422
METRICS = ("r2_test", "r2_adj_procapnet", "r2_adj_full")


@dataclass(frozen=True)
class Windows:
    chrom: np.ndarray
    start: np.ndarray
    end: np.ndarray
    strand: np.ndarray
    observed_counts: np.ndarray
    predicted_counts: np.ndarray
    source_row: np.ndarray

    def __len__(self) -> int:
        return len(self.chrom)

    def subset(self, indices: np.ndarray) -> "Windows":
        return Windows(
            chrom=self.chrom[indices],
            start=self.start[indices],
            end=self.end[indices],
            strand=self.strand[indices],
            observed_counts=self.observed_counts[indices],
            predicted_counts=self.predicted_counts[indices],
            source_row=self.source_row[indices],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--npz", required=True, help="PauseNet profiles NPZ.")
    parser.add_argument(
        "--train-chr",
        action="append",
        required=True,
        help="OLS training chromosome(s); repeat or comma-separate.",
    )
    parser.add_argument(
        "--test-chr",
        action="append",
        required=True,
        help="Disjoint OLS test chromosome(s); repeat or comma-separate.",
    )
    parser.add_argument(
        "--tf",
        action="append",
        required=True,
        metavar="NAME=BIGWIG",
        help="TF name and bigWig. Repeat once per TF.",
    )
    parser.add_argument(
        "--observed-key",
        default="observed_counts",
        help="NPZ observed-count key (default: observed_counts).",
    )
    parser.add_argument(
        "--predicted-key",
        default="predicted_counts",
        help="NPZ predicted-count key (default: predicted_counts).",
    )
    parser.add_argument(
        "--chrom-key",
        default="manifest_chrom",
        help="NPZ chromosome key (default: manifest_chrom).",
    )
    parser.add_argument(
        "--start-key",
        default="manifest_output_start",
        help="NPZ window-start key (default: manifest_output_start).",
    )
    parser.add_argument(
        "--end-key",
        default="manifest_output_end",
        help="NPZ window-end key (default: manifest_output_end).",
    )
    parser.add_argument(
        "--strand-key",
        default="manifest_strand",
        help="Optional NPZ strand key (default: manifest_strand).",
    )
    parser.add_argument(
        "--predicted-values",
        choices=("counts", "log1p-counts"),
        default="counts",
        help="Representation in --predicted-key (default: counts).",
    )
    parser.add_argument(
        "--expected-assembly",
        choices=("hg19", "hg38", "none"),
        default="hg19",
        help="Validate bigWig chr1 length (default: hg19).",
    )
    parser.add_argument(
        "--negative-signal",
        choices=("clip", "error"),
        default="clip",
        help="Handle negative bigWig signal before log1p (default: clip).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Independent bigWig readers (default: 4).",
    )
    parser.add_argument(
        "--no-interaction",
        action="store_true",
        help="Fit an additive model without x_model:x_tf.",
    )
    parser.add_argument(
        "--fit-all-tfs",
        action="store_true",
        help=(
            "Also fit All TF only and Model + All TF using every supplied "
            "TF as predictors."
        ),
    )
    parser.add_argument(
        "--only-all-tfs",
        action="store_true",
        help=(
            "With --fit-all-tfs, omit individual-TF rows and output only "
            "model_only, all_tf_only, and model_plus_all_tf."
        ),
    )
    parser.add_argument(
        "--output-tsv",
        required=True,
        help="Output R2 TSV; .tsv is added if omitted.",
    )
    return parser.parse_args()


def parse_chromosomes(values: Sequence[str], option: str) -> list[str]:
    chromosomes: list[str] = []
    for value in values:
        for item in value.split(","):
            chromosome = item.strip()
            if not chromosome:
                continue
            if not chromosome.startswith("chr"):
                chromosome = "chr" + chromosome
            chromosomes.append(chromosome)
    if not chromosomes:
        raise ValueError(f"{option} contains no chromosomes.")
    if len(chromosomes) != len(set(chromosomes)):
        raise ValueError(f"{option} contains duplicate chromosomes.")
    return chromosomes


def parse_tf_specs(items: Sequence[str]) -> list[tuple[str, Path]]:
    tracks: list[tuple[str, Path]] = []
    names: set[str] = set()
    for item in items:
        if "=" not in item:
            raise ValueError(f"--tf must be NAME=BIGWIG; received {item!r}.")
        name, path_text = item.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError("--tf contains an empty TF name.")
        if name in names:
            raise ValueError(f"Duplicate TF name: {name}")
        path = Path(path_text).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"TF bigWig not found: {path}")
        names.add(name)
        tracks.append((name, path))
    return tracks


def required_array(
    npz: np.lib.npyio.NpzFile,
    key: str,
    path: Path,
) -> np.ndarray:
    if key not in npz.files:
        raise KeyError(
            f"{path} is missing key {key!r}; available keys: "
            + ", ".join(npz.files)
        )
    return np.asarray(npz[key])


def collapse_counts(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 0:
        raise ValueError(f"{name} has no window dimension.")
    if values.ndim > 1:
        values = values.sum(axis=tuple(range(1, values.ndim)))
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values.")
    if np.any(values < 0):
        raise ValueError(f"{name} contains negative values.")
    return values


def load_windows(path_text: str, args: argparse.Namespace) -> Windows:
    path = Path(path_text).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"NPZ not found: {path}")
    with np.load(path, allow_pickle=False) as npz:
        observed = collapse_counts(
            required_array(npz, args.observed_key, path),
            args.observed_key,
        )
        predicted = collapse_counts(
            required_array(npz, args.predicted_key, path),
            args.predicted_key,
        )
        chrom = required_array(npz, args.chrom_key, path).astype(str)
        start = required_array(npz, args.start_key, path).astype(np.int64)
        end = required_array(npz, args.end_key, path).astype(np.int64)
        if args.strand_key in npz.files:
            strand = np.asarray(npz[args.strand_key]).astype(str)
        else:
            strand = np.full(len(observed), ".", dtype="<U1")

    arrays = {
        args.predicted_key: predicted,
        args.chrom_key: chrom,
        args.start_key: start,
        args.end_key: end,
        args.strand_key: strand,
    }
    for key, values in arrays.items():
        if len(values) != len(observed):
            raise ValueError(
                f"{key!r} has {len(values):,} rows; "
                f"{args.observed_key!r} has {len(observed):,}."
            )
    if len(observed) == 0:
        raise ValueError("The NPZ contains no windows.")
    if np.any(start < 0) or np.any(end <= start):
        raise ValueError("The NPZ contains invalid genomic intervals.")
    print(f"Loaded {len(observed):,} windows from {path}")
    return Windows(
        chrom=chrom,
        start=start,
        end=end,
        strand=strand,
        observed_counts=observed,
        predicted_counts=predicted,
        source_row=np.arange(len(observed), dtype=np.int64),
    )


def split_windows(
    data: Windows,
    train_chromosomes: Sequence[str],
    test_chromosomes: Sequence[str],
) -> tuple[Windows, Windows]:
    overlap = sorted(set(train_chromosomes).intersection(test_chromosomes))
    if overlap:
        raise ValueError(
            "Training and testing chromosomes overlap: " + ", ".join(overlap)
        )
    available = set(data.chrom.astype(str))
    requested = set(train_chromosomes).union(test_chromosomes)
    missing = sorted(requested.difference(available))
    if missing:
        raise ValueError(
            "Requested chromosomes are absent from the NPZ: "
            + ", ".join(missing)
            + ". Available: "
            + ", ".join(sorted(available))
        )
    train_indices = np.flatnonzero(
        np.isin(data.chrom, train_chromosomes)
    )
    test_indices = np.flatnonzero(np.isin(data.chrom, test_chromosomes))
    if len(train_indices) == 0 or len(test_indices) == 0:
        raise ValueError("The chromosome split produced an empty subset.")
    train = data.subset(train_indices)
    test = data.subset(test_indices)
    print(f"OLS split: train={len(train):,}; test={len(test):,} windows")
    return train, test


def validate_bigwig(
    name: str,
    path: Path,
    required_chromosomes: set[str],
    expected_assembly: str,
) -> None:
    with pyBigWig.open(str(path)) as bigwig:
        chromosome_sizes = bigwig.chroms()
    missing = sorted(required_chromosomes.difference(chromosome_sizes))
    if missing:
        raise ValueError(
            f"{name}: bigWig is missing chromosomes: " + ", ".join(missing)
        )
    if expected_assembly != "none":
        expected_chr1 = {
            "hg19": HG19_CHR1_LENGTH,
            "hg38": HG38_CHR1_LENGTH,
        }[expected_assembly]
        observed_chr1 = chromosome_sizes.get("chr1")
        if observed_chr1 != expected_chr1:
            raise ValueError(
                f"{name}: expected {expected_assembly} chr1 length "
                f"{expected_chr1:,}, found {observed_chr1} in {path}."
            )


def extract_tf_values(
    task: tuple[str, Path, np.ndarray, np.ndarray, np.ndarray, str],
) -> tuple[str, np.ndarray, int]:
    name, path, chromosome, start, end, negative_mode = task
    values = np.empty(len(chromosome), dtype=np.float64)
    negative_bases = 0
    with pyBigWig.open(str(path)) as bigwig:
        sizes = bigwig.chroms()
        for i, (seqname, left, right) in enumerate(
            zip(chromosome, start, end, strict=True)
        ):
            seqname = str(seqname)
            left = int(left)
            right = int(right)
            if right > sizes[seqname]:
                raise ValueError(
                    f"{name}: {seqname}:{left}-{right} exceeds bigWig bounds."
                )
            profile = np.asarray(
                bigwig.values(seqname, left, right, numpy=True),
                dtype=np.float64,
            )
            profile = np.nan_to_num(
                profile, nan=0.0, posinf=0.0, neginf=0.0
            )
            n_negative = int(np.count_nonzero(profile < 0))
            negative_bases += n_negative
            if n_negative:
                if negative_mode == "error":
                    raise ValueError(
                        f"{name}: negative signal at "
                        f"{seqname}:{left}-{right}."
                    )
                np.maximum(profile, 0.0, out=profile)
            values[i] = np.log1p(profile).mean(dtype=np.float64)
    return name, values, negative_bases


def extract_all_tf_values(
    tf_specs: Sequence[tuple[str, Path]],
    train: Windows,
    test: Windows,
    args: argparse.Namespace,
) -> tuple[list[str], np.ndarray, np.ndarray, dict[str, int]]:
    chromosome = np.concatenate([train.chrom, test.chrom])
    start = np.concatenate([train.start, test.start])
    end = np.concatenate([train.end, test.end])
    required_chromosomes = set(chromosome.astype(str))
    tasks = []
    for name, path in tf_specs:
        validate_bigwig(
            name,
            path,
            required_chromosomes,
            args.expected_assembly,
        )
        tasks.append(
            (
                name,
                path,
                chromosome,
                start,
                end,
                args.negative_signal,
            )
        )

    workers = min(args.threads, len(tasks))
    print(
        f"Extracting {len(tasks)} TF bigWig track(s) "
        f"with {workers} reader(s) ..."
    )
    values_by_name: dict[str, np.ndarray] = {}
    negative_counts: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(extract_tf_values, task): task[0]
            for task in tasks
        }
        for future in as_completed(futures):
            name, values, negative_bases = future.result()
            values_by_name[name] = values
            negative_counts[name] = negative_bases
            suffix = (
                f"; clipped {negative_bases:,} negative bases"
                if negative_bases
                else ""
            )
            print(f"  completed {name}{suffix}")

    tf_names = [name for name, _ in tf_specs]
    matrix = np.column_stack([values_by_name[name] for name in tf_names])
    n_train = len(train)
    return tf_names, matrix[:n_train], matrix[n_train:], negative_counts


def design_matrix(
    model_values: np.ndarray | None,
    tf_values: np.ndarray | None,
    interaction: bool,
) -> tuple[np.ndarray, list[str], int]:
    if model_values is None and tf_values is None:
        raise ValueError("At least one predictor must be supplied.")
    if model_values is not None:
        model_values = np.asarray(model_values, dtype=np.float64)
        n_rows = len(model_values)
    else:
        tf_values = np.asarray(tf_values, dtype=np.float64)
        n_rows = len(tf_values)

    blocks = [np.ones((n_rows, 1), dtype=np.float64)]
    terms = ["Intercept"]
    p_main = 0
    if model_values is not None:
        blocks.append(model_values[:, None])
        terms.append("Model")
        p_main += 1
    if tf_values is not None:
        tf_values = np.asarray(tf_values, dtype=np.float64)
        if len(tf_values) != n_rows:
            raise ValueError("Model and TF predictors have different lengths.")
        blocks.append(tf_values[:, None])
        terms.append("TF")
        p_main += 1
        if interaction and model_values is not None:
            blocks.append((model_values * tf_values)[:, None])
            terms.append("Model:TF")
    matrix = np.concatenate(blocks, axis=1)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("The design matrix contains non-finite values.")
    return matrix, terms, p_main


def adjusted_r2(r2: float, n: int, p: int) -> float:
    denominator = n - p - 1
    if denominator <= 0:
        return float("nan")
    return 1.0 - (1.0 - r2) * (n - 1) / denominator


def fit_and_score(
    y_train: np.ndarray,
    y_test: np.ndarray,
    model_train: np.ndarray | None,
    model_test: np.ndarray | None,
    tf_train: np.ndarray | None,
    tf_test: np.ndarray | None,
    interaction: bool,
) -> dict:
    train_design, terms, p_main = design_matrix(
        model_train, tf_train, interaction
    )
    test_design, test_terms, _ = design_matrix(
        model_test, tf_test, interaction
    )
    if terms != test_terms:
        raise AssertionError("Training/test design terms differ.")
    coefficients, _, rank, _ = np.linalg.lstsq(
        train_design, y_train, rcond=None
    )
    fitted = test_design @ coefficients
    residual_ss = float(np.sum(np.square(y_test - fitted)))
    total_ss = float(np.sum(np.square(y_test - np.mean(y_test))))
    if total_ss == 0:
        raise ValueError("The test response is constant; R2 is undefined.")
    r2 = 1.0 - residual_ss / total_ss
    p_full = len(terms) - 1
    result = {
        "n_main_predictors": p_main,
        "n_terms_excluding_intercept": p_full,
        "matrix_rank": int(rank),
        "r2_test": r2,
        "r2_adj_procapnet": adjusted_r2(r2, len(y_test), p_main),
        "r2_adj_full": adjusted_r2(r2, len(y_test), p_full),
        "rmse_test_log1p": float(
            np.sqrt(np.mean(np.square(y_test - fitted)))
        ),
    }
    return result


def build_multi_predictor_design(
    predictors: np.ndarray,
    interaction: bool,
) -> tuple[np.ndarray, int, int]:
    predictors = np.asarray(predictors, dtype=np.float64)
    if predictors.ndim == 1:
        predictors = predictors[:, None]
    if predictors.ndim != 2 or predictors.shape[1] == 0:
        raise ValueError("Predictors must be a nonempty two-dimensional matrix.")
    if not np.all(np.isfinite(predictors)):
        raise ValueError("Predictors contain non-finite values.")

    blocks = [np.ones((len(predictors), 1)), predictors]
    p_main = predictors.shape[1]
    p_full = p_main
    if interaction and p_main > 1:
        for left, right in itertools.combinations(range(p_main), 2):
            blocks.append(
                (predictors[:, left] * predictors[:, right])[:, None]
            )
            p_full += 1
    return np.concatenate(blocks, axis=1), p_main, p_full


def fit_multi_predictor_model(
    y_train: np.ndarray,
    y_test: np.ndarray,
    predictors_train: np.ndarray,
    predictors_test: np.ndarray,
    interaction: bool,
) -> dict:
    train_design, p_main, p_full = build_multi_predictor_design(
        predictors_train, interaction
    )
    test_design, test_p_main, test_p_full = build_multi_predictor_design(
        predictors_test, interaction
    )
    if p_main != test_p_main or p_full != test_p_full:
        raise AssertionError("Training/test design terms differ.")
    coefficients, _, rank, _ = np.linalg.lstsq(
        train_design, y_train, rcond=None
    )
    fitted = test_design @ coefficients
    residual_ss = float(np.sum(np.square(y_test - fitted)))
    total_ss = float(np.sum(np.square(y_test - np.mean(y_test))))
    if total_ss == 0:
        raise ValueError("The test response is constant; R2 is undefined.")
    r2 = 1.0 - residual_ss / total_ss
    return {
        "n_main_predictors": p_main,
        "n_terms_excluding_intercept": p_full,
        "matrix_rank": int(rank),
        "r2_test": r2,
        "r2_adj_procapnet": adjusted_r2(r2, len(y_test), p_main),
        "r2_adj_full": adjusted_r2(r2, len(y_test), p_full),
        "rmse_test_log1p": float(
            np.sqrt(np.mean(np.square(y_test - fitted)))
        ),
    }


def calculate_results(
    train: Windows,
    test: Windows,
    tf_names: Sequence[str],
    tf_train: np.ndarray,
    tf_test: np.ndarray,
    tf_paths: dict[str, Path],
    train_chromosomes: Sequence[str],
    test_chromosomes: Sequence[str],
    args: argparse.Namespace,
) -> list[dict]:
    y_train = np.log1p(train.observed_counts)
    y_test = np.log1p(test.observed_counts)
    if args.predicted_values == "counts":
        model_train = np.log1p(train.predicted_counts)
        model_test = np.log1p(test.predicted_counts)
    else:
        model_train = train.predicted_counts.astype(np.float64)
        model_test = test.predicted_counts.astype(np.float64)

    baseline = fit_and_score(
        y_train,
        y_test,
        model_train,
        model_test,
        None,
        None,
        interaction=False,
    )
    print(
        f"Model only: R2={baseline['r2_test']:.4f}; "
        f"adjusted R2={baseline['r2_adj_procapnet']:.4f}"
    )
    rows: list[dict] = []
    train_text = ",".join(train_chromosomes)
    test_text = ",".join(test_chromosomes)

    def make_row(
        tf: str,
        condition: str,
        bigwig: str,
        result: dict,
    ) -> dict:
        row = {
            "tf": tf,
            "condition": condition,
            "tf_bigwig": bigwig,
            "train_chromosomes": train_text,
            "test_chromosomes": test_text,
            "n_train": len(train),
            "n_test": len(test),
            **{metric: result[metric] for metric in METRICS},
        }
        for metric in METRICS:
            row[f"delta_{metric}_vs_model"] = (
                float(result[metric]) - float(baseline[metric])
            )
        return row

    interaction = not args.no_interaction
    if args.only_all_tfs:
        rows.append(
            make_row(
                tf="ALL",
                condition="model_only",
                bigwig="-",
                result=baseline,
            )
        )

    for i, tf in enumerate(tf_names):
        if args.only_all_tfs:
            break
        rows.append(
            make_row(
                tf=tf,
                condition="model_only",
                bigwig="-",
                result=baseline,
            )
        )
        tf_only = fit_and_score(
            y_train,
            y_test,
            None,
            None,
            tf_train[:, i],
            tf_test[:, i],
            interaction=False,
        )
        rows.append(
            make_row(
                tf=tf,
                condition="tf_only",
                bigwig=str(tf_paths[tf]),
                result=tf_only,
            )
        )
        combined = fit_and_score(
            y_train,
            y_test,
            model_train,
            model_test,
            tf_train[:, i],
            tf_test[:, i],
            interaction=interaction,
        )
        rows.append(
            make_row(
                tf=tf,
                condition="model_plus_tf",
                bigwig=str(tf_paths[tf]),
                result=combined,
            )
        )
        print(
            f"{tf}: TF only R2={tf_only['r2_test']:.4f}; "
            f"model + TF R2={combined['r2_test']:.4f}; "
            f"model + TF adjusted R2="
            f"{combined['r2_adj_procapnet']:.4f}"
        )

    if args.fit_all_tfs:
        all_tf_only = fit_multi_predictor_model(
            y_train,
            y_test,
            tf_train,
            tf_test,
            interaction=interaction,
        )
        model_plus_all_tf = fit_multi_predictor_model(
            y_train,
            y_test,
            np.column_stack([model_train, tf_train]),
            np.column_stack([model_test, tf_test]),
            interaction=interaction,
        )
        all_bigwigs = ";".join(
            f"{tf}={tf_paths[tf]}" for tf in tf_names
        )
        rows.append(
            make_row(
                tf="ALL",
                condition="all_tf_only",
                bigwig=all_bigwigs,
                result=all_tf_only,
            )
        )
        rows.append(
            make_row(
                tf="ALL",
                condition="model_plus_all_tf",
                bigwig=all_bigwigs,
                result=model_plus_all_tf,
            )
        )
        print(
            f"All TF only R2={all_tf_only['r2_test']:.4f}; "
            f"model + all TF R2={model_plus_all_tf['r2_test']:.4f}; "
            f"model + all TF adjusted R2="
            f"{model_plus_all_tf['r2_adj_procapnet']:.4f}"
        )
    return rows


def write_tsv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, delimiter="\t"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: f"{value:.10g}" if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )


def main() -> None:
    args = parse_args()
    if args.threads < 1:
        raise ValueError("--threads must be at least 1.")
    if args.only_all_tfs and not args.fit_all_tfs:
        raise ValueError("--only-all-tfs requires --fit-all-tfs.")
    train_chromosomes = parse_chromosomes(
        args.train_chr, "--train-chr"
    )
    test_chromosomes = parse_chromosomes(args.test_chr, "--test-chr")
    tf_specs = parse_tf_specs(args.tf)
    if args.fit_all_tfs and len(tf_specs) < 2:
        raise ValueError("--fit-all-tfs requires at least two --tf tracks.")
    tf_paths = dict(tf_specs)

    all_windows = load_windows(args.npz, args)
    train, test = split_windows(
        all_windows, train_chromosomes, test_chromosomes
    )
    tf_names, tf_train, tf_test, _negative_counts = extract_all_tf_values(
        tf_specs, train, test, args
    )
    rows = calculate_results(
        train,
        test,
        tf_names,
        tf_train,
        tf_test,
        tf_paths,
        train_chromosomes,
        test_chromosomes,
        args,
    )

    r2_path = Path(args.output_tsv).expanduser().resolve()
    if r2_path.suffix.lower() != ".tsv":
        r2_path = r2_path.with_suffix(".tsv")
    r2_path.parent.mkdir(parents=True, exist_ok=True)
    write_tsv(r2_path, rows)

    print("\nCompleted:")
    print(f"  R2 results: {r2_path}")


if __name__ == "__main__":
    main()
