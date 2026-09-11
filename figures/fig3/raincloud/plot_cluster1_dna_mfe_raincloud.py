#!/usr/bin/env python3
"""Compute cluster-1 non-template DNA MFE and draw one raincloud PDF.

For every 50-nt TF-MoDISco seqlet, the script:

1. undoes TF-MoDISco reverse-complement alignment to restore the PauseNet
   non-template/coding DNA strand in 5'-to-3' orientation;
2. folds every overlapping 30-nt window with ViennaRNA DNA Mathews 2004
   thermodynamic parameters at 37 degrees C; and
3. uses the most negative window MFE as the seqlet-level structure score.

No dinucleotide-matched correction is performed. The default run produces
one count/profile raincloud PDF; use --save-scores to retain the per-seqlet
table as an optional TSV.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import multiprocessing as mp
import os
from pathlib import Path
import re

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import RNA


HERE = Path(__file__).resolve().parent
DEFAULT_METADATA = Path(
    "/mnt/HDD8TB/houruiyan/pausing_site/4_plot_figure/fig2/plot_tsne/"
    "hek293t_count_profile_positive_pattern_tsne_metadata.tsv"
)
DEFAULT_COUNT_H5 = Path(
    "/mnt/HDD8TB/houruiyan/pausing_site/3_model_explanation/TFMoDISco/"
    "results/hek293t_netseq/count/count_tfmodisco_patterns.h5"
)
DEFAULT_PROFILE_H5 = Path(
    "/mnt/HDD8TB/houruiyan/pausing_site/3_model_explanation/TFMoDISco/"
    "results/hek293t_netseq/profile/profile_tfmodisco_patterns.h5"
)
DEFAULT_OUTPUT = HERE / "cluster1_dna_mfe_raincloud.pdf"

BASES = np.asarray(list("ACGT"))
COLORS = {"count": "#E71D36", "profile": "#2EC4B6"}
SOURCE_LABELS = {"count": "Count patterns", "profile": "Profile patterns"}
SOURCES = ("count", "profile")

WORKER_WINDOW_SIZE = 30
WORKER_MODEL: RNA.md | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate minimum 30-nt DNA MFE for cluster seqlets and draw "
            "one publication-style raincloud PDF."
        )
    )
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--count-h5", type=Path, default=DEFAULT_COUNT_H5)
    parser.add_argument("--profile-h5", type=Path, default=DEFAULT_PROFILE_H5)
    parser.add_argument("--cluster", type=int, default=1)
    parser.add_argument(
        "--window-size",
        type=int,
        default=30,
        help="DNA folding sliding-window length (default: 30 nt).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=37.0,
        help="Folding temperature in degrees C (default: 37).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Parallel ViennaRNA workers (default: min(8, CPU count)).",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--save-scores",
        type=Path,
        default=None,
        help="Optional per-seqlet TSV output; omitted by default.",
    )
    parser.add_argument(
        "--max-seqlets-per-pattern",
        type=int,
        default=None,
        help="Optional testing limit; all seqlets are used by default.",
    )
    parser.add_argument(
        "--max-points-per-pattern",
        type=int,
        default=180,
        help=(
            "Maximum raw points displayed per pattern. All calculated "
            "seqlets are still used for violin and box summaries."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260820,
        help="Seed used only for display-point subsampling and jitter.",
    )
    return parser.parse_args()


def set_style() -> None:
    """Apply the NET-seq project's final-figure-small typography."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.figsize": (5, 4),
            "axes.linewidth": 0.9,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "text.color": "#000000",
            "axes.labelcolor": "#000000",
            "axes.titlecolor": "#000000",
            "xtick.color": "#000000",
            "ytick.color": "#000000",
            "legend.labelcolor": "#000000",
        }
    )


def pattern_number(motif_id: str) -> int:
    match = re.search(r"pattern_(\d+)$", motif_id)
    if match is None:
        raise ValueError(f"Cannot parse pattern number from {motif_id!r}")
    return int(match.group(1))


def reverse_complement_onehot(onehot: np.ndarray) -> np.ndarray:
    """Reverse-complement an (L, 4) A/C/G/T matrix."""
    return onehot[::-1, ::-1]


def onehot_to_sequence(onehot: np.ndarray) -> str:
    """Convert an (L, 4) A/C/G/T one-hot matrix to a DNA sequence."""
    onehot = np.asarray(onehot, dtype=np.float64)
    if onehot.ndim != 2 or onehot.shape[1] != 4:
        raise ValueError(f"Expected one-hot shape (L,4), received {onehot.shape}")
    if not np.all(np.isfinite(onehot)):
        raise ValueError("Seqlet one-hot matrix contains NaN or Inf")
    if np.min(onehot) < -1e-6:
        raise ValueError("Seqlet one-hot matrix contains negative values")

    row_sums = onehot.sum(axis=1)
    calls = BASES[np.argmax(onehot, axis=1)].astype("<U1")
    calls[row_sums < 0.5] = "N"
    return "".join(calls.tolist())


def read_cluster_metadata(path: Path, cluster: int) -> pd.DataFrame:
    metadata = pd.read_csv(path, sep="\t")
    required = {"motif_id", "source", "h5_group", "n_seqlets", "leiden_cluster"}
    missing = required.difference(metadata.columns)
    if missing:
        raise KeyError(f"Metadata is missing columns: {sorted(missing)}")

    selected = metadata.loc[metadata["leiden_cluster"] == cluster].copy()
    if selected.empty:
        raise ValueError(f"No motifs found for Leiden cluster {cluster}")
    if not set(selected["source"]).issubset(set(SOURCES)):
        raise ValueError("Metadata source must contain only count/profile")
    selected["pattern_number"] = selected["motif_id"].map(pattern_number)
    return selected.sort_values(["source", "pattern_number"])


def load_cluster_seqlets(
    selected: pd.DataFrame,
    count_h5: Path,
    profile_h5: Path,
    max_seqlets_per_pattern: int | None,
) -> list[tuple]:
    """Read and restore cluster seqlets to non-template DNA orientation."""
    h5_paths = {"count": count_h5, "profile": profile_h5}
    handles: dict[str, h5py.File] = {}
    records: list[tuple] = []

    try:
        for source, path in h5_paths.items():
            if not path.is_file():
                raise FileNotFoundError(path)
            handles[source] = h5py.File(path, "r")

        for row in selected.itertuples(index=False):
            handle = handles[row.source]
            if row.h5_group not in handle:
                raise KeyError(f"{h5_paths[row.source]}: missing {row.h5_group}")

            seqlet_group = handle[row.h5_group]["seqlets"]
            sequences = seqlet_group["sequence"]
            reverse_flags = np.asarray(seqlet_group["is_revcomp"], dtype=bool)
            n_seqlets = int(sequences.shape[0])

            if sequences.ndim != 3 or sequences.shape[1:] != (50, 4):
                raise ValueError(
                    f"{row.motif_id}: expected (N,50,4), got {sequences.shape}"
                )
            if len(reverse_flags) != n_seqlets:
                raise ValueError(f"{row.motif_id}: inconsistent is_revcomp length")
            if int(row.n_seqlets) != n_seqlets:
                raise ValueError(
                    f"{row.motif_id}: metadata has {row.n_seqlets} seqlets, "
                    f"but H5 has {n_seqlets}"
                )

            use_n = n_seqlets
            if max_seqlets_per_pattern is not None:
                use_n = min(use_n, max_seqlets_per_pattern)

            for seqlet_index in range(use_n):
                aligned = np.asarray(sequences[seqlet_index], dtype=np.float64)
                # TF-MoDISco stores seqlets in motif-aligned orientation.
                # Undo its reverse-complement operation. PauseNet inputs are
                # transcription-oriented, so this restores the non-template
                # (coding) DNA strand in 5'-to-3' orientation.
                non_template = (
                    reverse_complement_onehot(aligned)
                    if reverse_flags[seqlet_index]
                    else aligned
                )
                records.append(
                    (
                        row.motif_id,
                        row.source,
                        int(row.pattern_number),
                        seqlet_index,
                        bool(reverse_flags[seqlet_index]),
                        onehot_to_sequence(non_template),
                    )
                )

            print(
                f"Loaded {row.motif_id}: {use_n:,}/{n_seqlets:,} seqlets",
                flush=True,
            )
    finally:
        for handle in handles.values():
            handle.close()

    if not records:
        raise ValueError("No cluster seqlets were loaded")
    return records


def initialize_fold_worker(window_size: int, temperature: float) -> None:
    """Initialize one ViennaRNA worker with DNA thermodynamic parameters."""
    global WORKER_WINDOW_SIZE, WORKER_MODEL
    WORKER_WINDOW_SIZE = window_size
    RNA.params_load_DNA_Mathews2004()
    model = RNA.md()
    model.temperature = temperature
    model.dangles = 2
    WORKER_MODEL = model


@lru_cache(maxsize=200_000)
def dna_window_mfe(sequence: str) -> float:
    """Fold one DNA window and return its MFE in kcal/mol."""
    if WORKER_MODEL is None:
        raise RuntimeError("DNA folding worker was not initialized")
    fold_compound = RNA.fold_compound(sequence, WORKER_MODEL)
    _, mfe = fold_compound.mfe()
    return float(mfe)


def fold_seqlet(record: tuple) -> tuple:
    """Return the most negative valid-window MFE for one 50-nt seqlet."""
    sequence = record[-1]
    candidates: list[tuple[float, int, str]] = []
    for start in range(len(sequence) - WORKER_WINDOW_SIZE + 1):
        window = sequence[start : start + WORKER_WINDOW_SIZE]
        if "N" in window:
            continue
        candidates.append((dna_window_mfe(window), start, window))

    if candidates:
        mfe, best_start, best_window = min(candidates, key=lambda item: item[0])
        n_valid_windows = len(candidates)
        best_end = best_start + WORKER_WINDOW_SIZE
    else:
        mfe, best_start, best_end = np.nan, -1, -1
        best_window = ""
        n_valid_windows = 0

    return record + (
        n_valid_windows,
        float(mfe),
        best_start,
        best_end,
        best_window,
    )


def calculate_mfe(
    records: list[tuple],
    window_size: int,
    temperature: float,
    jobs: int,
) -> pd.DataFrame:
    """Fold all seqlets and return a per-seqlet result table."""
    initializer_args = (window_size, temperature)
    if jobs == 1:
        initialize_fold_worker(*initializer_args)
        folded = [fold_seqlet(record) for record in records]
    else:
        context = mp.get_context("fork")
        chunksize = max(1, len(records) // (jobs * 20))
        with context.Pool(
            processes=jobs,
            initializer=initialize_fold_worker,
            initargs=initializer_args,
        ) as pool:
            folded = list(
                pool.imap(fold_seqlet, records, chunksize=chunksize)
            )

    columns = [
        "motif_id",
        "source",
        "pattern_number",
        "seqlet_index",
        "was_revcomp_in_tfmodisco",
        "non_template_sequence_5to3",
        "n_valid_windows",
        "min_mfe_kcal_mol",
        "min_window_start_0based",
        "min_window_end_0based_exclusive",
        "min_window_sequence_5to3",
    ]
    values = pd.DataFrame(folded, columns=columns)
    values = values.loc[np.isfinite(values["min_mfe_kcal_mol"])].copy()
    if values.empty:
        raise ValueError("No valid seqlet MFE values were calculated")
    return values


def motif_order(data: pd.DataFrame, source: str) -> list[str]:
    table = (
        data.loc[data["source"] == source, ["motif_id", "pattern_number"]]
        .drop_duplicates()
        .sort_values("pattern_number")
    )
    return table["motif_id"].tolist()


def plot_raincloud(
    data: pd.DataFrame,
    output: Path,
    max_points_per_pattern: int,
    seed: int,
) -> None:
    """Draw half violins, raw seqlets and compact box summaries."""
    fig, axes = plt.subplots(1, 2, figsize=(5, 4), sharex=True)
    rng = np.random.default_rng(seed)
    global_min = float(data["min_mfe_kcal_mol"].min())
    global_max = float(data["min_mfe_kcal_mol"].max())
    padding = max(0.5, 0.04 * (global_max - global_min))

    for ax, source in zip(axes, SOURCES, strict=True):
        order = motif_order(data, source)
        if not order:
            raise ValueError(f"No {source} patterns are available for plotting")
        arrays = [
            data.loc[data["motif_id"] == motif, "min_mfe_kcal_mol"].to_numpy()
            for motif in order
        ]
        positions = np.arange(len(order), dtype=float)
        color = COLORS[source]

        violins = ax.violinplot(
            arrays,
            positions=positions,
            vert=False,
            widths=0.78,
            showmeans=False,
            showmedians=False,
            showextrema=False,
            bw_method=0.25,
        )
        for body, position in zip(
            violins["bodies"], positions, strict=True
        ):
            vertices = body.get_paths()[0].vertices
            vertices[:, 1] = np.minimum(vertices[:, 1], position)
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_linewidth(0.7)
            body.set_alpha(0.28)

        for position, values in zip(positions, arrays, strict=True):
            shown = values
            if len(values) > max_points_per_pattern:
                shown = rng.choice(
                    values, size=max_points_per_pattern, replace=False
                )
            jitter = position + 0.08 + rng.uniform(
                0.0, 0.18, size=len(shown)
            )
            ax.scatter(
                shown,
                jitter,
                s=4.5,
                color=color,
                alpha=0.25,
                linewidths=0,
                rasterized=False,
                zorder=2,
            )

        boxes = ax.boxplot(
            arrays,
            positions=positions,
            vert=False,
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
        for artist in boxes["boxes"]:
            artist.set_zorder(3)

        ax.axvline(
            -5,
            color="#7A7A7A",
            linewidth=0.8,
            linestyle=(0, (3, 2)),
            zorder=0,
        )

        ax.set_yticks(positions)
        ax.set_yticklabels([f"P{pattern_number(motif)}" for motif in order])
        ax.invert_yaxis()
        ax.set_xlim(global_min - padding, global_max + padding)
        ax.set_title(
            SOURCE_LABELS[source], color="black", fontweight="normal", pad=4
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", color="#E6E6E6", linewidth=0.55, zorder=0)
        ax.grid(axis="y", visible=False)

    fig.supxlabel("Minimum free energy (kcal/mol)", y=0.02, fontsize=14)
    fig.subplots_adjust(
        left=0.11,
        right=0.99,
        bottom=0.17,
        top=0.90,
        wspace=0.32,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if not 1 <= args.window_size <= 50:
        raise ValueError("--window-size must be between 1 and 50")
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    if args.max_seqlets_per_pattern is not None:
        if args.max_seqlets_per_pattern < 1:
            raise ValueError("--max-seqlets-per-pattern must be positive")
    if args.max_points_per_pattern < 1:
        raise ValueError("--max-points-per-pattern must be positive")

    metadata_path = args.metadata.resolve()
    count_h5 = args.count_h5.resolve()
    profile_h5 = args.profile_h5.resolve()
    output_path = args.output.resolve()
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)

    selected = read_cluster_metadata(metadata_path, args.cluster)
    records = load_cluster_seqlets(
        selected=selected,
        count_h5=count_h5,
        profile_h5=profile_h5,
        max_seqlets_per_pattern=args.max_seqlets_per_pattern,
    )
    print(
        f"Folding {len(records):,} seqlets with {args.window_size}-nt windows, "
        f"DNA Mathews 2004, {args.temperature:g} C, jobs={args.jobs}",
        flush=True,
    )
    values = calculate_mfe(
        records=records,
        window_size=args.window_size,
        temperature=args.temperature,
        jobs=args.jobs,
    )

    if args.save_scores is not None:
        scores_path = args.save_scores.resolve()
        scores_path.parent.mkdir(parents=True, exist_ok=True)
        values.to_csv(
            scores_path,
            sep="\t",
            index=False,
            float_format="%.6g",
        )
        print(f"scores: {scores_path}")

    set_style()
    plot_raincloud(
        data=values,
        output=output_path,
        max_points_per_pattern=args.max_points_per_pattern,
        seed=args.seed,
    )
    print(
        f"Finished: {len(values):,} valid seqlets from "
        f"{values['motif_id'].nunique()} patterns"
    )
    print(f"raincloud: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
