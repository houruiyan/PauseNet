#!/usr/bin/env python3
"""Plot a compact TF atlas for selected count TF-MoDISco patterns.

Columns: matched TF, de novo TF-MoDISco motif, matched JASPAR motif,
positive/negative class, number of seqlets, and seqlet region composition.
The matched JASPAR motif is reverse-complemented when TomTom reports a
negative orientation, so that it is displayed in the de novo motif direction.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import logomaker
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/mnt/HDD8TB/houruiyan/pausing_site")
COUNT_DIR = ROOT / "3_model_explanation/TFMoDISco/results/hek293t_netseq/count"
H5_PATH = COUNT_DIR / "count_tfmodisco_patterns.h5"
SELECTED_MANIFEST = ROOT / "3_model_explanation/DeepSHAP/hek293t_netseq/selected_manifest.tsv"
TOMTOM_TSV = ROOT / "3_model_explanation/TomTom/result/hek293t_netseq/count/tomtom.tsv"
JASPAR_MEME = ROOT / "3_model_explanation/TomTom/db/JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt"
OUTDIR = ROOT / "4_plot_figure/fig3/2.TF_atlas_count"

BASES = ["A", "C", "G", "T"]
BASE_COLORS = {"A": "#16A23A", "C": "#2468C5", "G": "#F39C12", "T": "#E53935"}
REGIONS = ["TSS", "5SS", "3SS", "TES"]
REGION_COLORS = {
    "TSS": "#4E7FAF",
    "5SS": "#75B7B2",
    "3SS": "#FF7F0E",
    "TES": "#4DA64A",
}


@dataclass(frozen=True)
class PatternSpec:
    query_id: str
    tf_name: str
    target_id: str

    @property
    def pattern_class(self) -> str:
        return "positive" if self.query_id.startswith("pos_") else "negative"

    @property
    def h5_branch(self) -> str:
        return "pos_patterns" if self.pattern_class == "positive" else "neg_patterns"

    @property
    def pattern_name(self) -> str:
        return self.query_id.removeprefix("pos_").removeprefix("neg_")


PATTERNS = [
    PatternSpec("pos_pattern_2", "PATZ1", "MA1961.2"),
    PatternSpec("pos_pattern_7", "ZNF610", "MA1713.2"),
    PatternSpec("pos_pattern_8", "SP1", "MA0079.5"),
    PatternSpec("pos_pattern_10", "SP2", "MA0516.3"),
]

# Inclusive, 1-based positions within the automatically selected de novo
# display window. These ranges deliberately refer to the motif as displayed,
# rather than to the full 50-position TF-MoDISco matrix.
DE_NOVO_DISPLAY_RANGES_1BASED = {
    "pos_pattern_2": (1, 11),
    "pos_pattern_7": (2, 11),
    "pos_pattern_8": (1, 9),
    "pos_pattern_10": (4, 13),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTDIR / "count_tf_atlas.pdf")
    parser.add_argument("--summary", type=Path, default=OUTDIR / "count_tf_atlas_summary.tsv")
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
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def normalize_pwm(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != 4:
        raise ValueError(f"Expected an Lx4 motif matrix, got {matrix.shape}")
    row_sum = matrix.sum(axis=1, keepdims=True)
    if np.any(row_sum <= 0):
        raise ValueError("Motif contains a position with zero total probability")
    return matrix / row_sum


def information_content(pwm: np.ndarray) -> np.ndarray:
    p = np.clip(normalize_pwm(pwm), 1e-9, 1.0)
    return np.maximum(0.0, 2.0 + np.sum(p * np.log2(p), axis=1))


def pwm_to_bits(pwm: np.ndarray) -> pd.DataFrame:
    p = normalize_pwm(pwm)
    return pd.DataFrame(p * information_content(p)[:, None], columns=BASES)


def reverse_complement_pwm(pwm: np.ndarray) -> np.ndarray:
    # Rows reverse; A,C,G,T columns become T,G,C,A.
    return normalize_pwm(pwm)[::-1, :][:, [3, 2, 1, 0]]


def crop_de_novo_pwm(pwm: np.ndarray, target_width: int) -> tuple[np.ndarray, int, int]:
    """Keep the most informative window while retaining modest motif context."""
    pwm = normalize_pwm(pwm)
    width = min(len(pwm), max(10, min(20, int(target_width) + 4)))
    if len(pwm) <= width:
        return pwm, 0, len(pwm)
    score = information_content(pwm)
    window_score = np.convolve(score, np.ones(width), mode="valid")
    start = int(np.argmax(window_score))
    return pwm[start : start + width], start, start + width


def read_meme_database(path: Path) -> dict[str, tuple[str, np.ndarray]]:
    lines = path.read_text().splitlines()
    motifs: dict[str, tuple[str, np.ndarray]] = {}
    i = 0
    while i < len(lines):
        if not lines[i].startswith("MOTIF "):
            i += 1
            continue
        fields = lines[i].split(maxsplit=2)
        motif_id = fields[1]
        motif_name = fields[2] if len(fields) > 2 else motif_id
        i += 1
        while i < len(lines) and "letter-probability matrix:" not in lines[i]:
            if lines[i].startswith("MOTIF "):
                raise ValueError(f"No probability matrix found for {motif_id}")
            i += 1
        if i >= len(lines):
            raise ValueError(f"No probability matrix found for {motif_id}")
        match = re.search(r"\bw\s*=\s*(\d+)", lines[i])
        if match is None:
            raise ValueError(f"Cannot parse motif width for {motif_id}")
        width = int(match.group(1))
        rows = []
        i += 1
        while i < len(lines) and len(rows) < width:
            if lines[i].strip():
                values = [float(value) for value in lines[i].split()[:4]]
                if len(values) != 4:
                    raise ValueError(f"Malformed row in motif {motif_id}")
                rows.append(values)
            i += 1
        if len(rows) != width:
            raise ValueError(f"Incomplete probability matrix for {motif_id}")
        motifs[motif_id] = (motif_name, normalize_pwm(np.asarray(rows)))
    return motifs


def draw_logo(ax: plt.Axes, pwm: np.ndarray) -> None:
    frame = pwm_to_bits(pwm)
    logomaker.Logo(
        frame,
        ax=ax,
        color_scheme=BASE_COLORS,
        fade_below=0,
        shade_below=0,
        width=0.92,
    )
    ax.axhline(0, color="#555555", linewidth=0.4)
    ax.set_xlim(-0.6, len(frame) - 0.4)
    ax.set_ylim(0, 2.05)
    ax.set_axis_off()


def region_counts(sample_ids: pd.Series) -> dict[str, int]:
    labels = sample_ids.astype(str).str.extract(r"\|(TSS|5SS|3SS|TES)\|", expand=False)
    if labels.isna().any():
        examples = sample_ids[labels.isna()].head(3).tolist()
        raise ValueError(f"Cannot parse region type from sample_id values: {examples}")
    counts = labels.value_counts()
    return {region: int(counts.get(region, 0)) for region in REGIONS}


def load_rows() -> list[dict[str, object]]:
    for path in (H5_PATH, SELECTED_MANIFEST, TOMTOM_TSV, JASPAR_MEME):
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = pd.read_csv(SELECTED_MANIFEST, sep="\t")
    expected = np.arange(len(manifest), dtype=int)
    observed = manifest["contribution_array_index"].to_numpy(dtype=int)
    if not np.array_equal(expected, observed):
        raise ValueError("selected_manifest.tsv is not in contribution-array order")

    tomtom = pd.read_csv(TOMTOM_TSV, sep="\t", comment="#")
    jaspar = read_meme_database(JASPAR_MEME)
    rows: list[dict[str, object]] = []

    with h5py.File(H5_PATH, "r") as handle:
        for spec in PATTERNS:
            group_path = f"{spec.h5_branch}/{spec.pattern_name}"
            if group_path not in handle:
                raise KeyError(f"Missing TF-MoDISco pattern: {group_path}")
            if spec.target_id not in jaspar:
                raise KeyError(f"Missing JASPAR motif: {spec.target_id}")

            match = tomtom[
                (tomtom["Query_ID"] == spec.query_id)
                & (tomtom["Target_ID"] == spec.target_id)
            ].sort_values(["q-value", "p-value"])
            if match.empty:
                raise KeyError(f"TomTom match is absent: {spec.query_id} -> {spec.target_id}")
            hit = match.iloc[0]
            orientation = str(hit["Orientation"])
            if orientation not in {"+", "-"}:
                raise ValueError(f"Unexpected TomTom orientation: {orientation}")

            group = handle[group_path]
            de_novo_full = normalize_pwm(np.asarray(group["sequence"], dtype=float))
            seqlet_idx = np.asarray(group["seqlets/example_idx"], dtype=np.int64)
            if np.any(seqlet_idx < 0) or np.any(seqlet_idx >= len(manifest)):
                raise IndexError(f"{spec.query_id}: example_idx outside selected_manifest.tsv")

            jaspar_name, known_pwm = jaspar[spec.target_id]
            if orientation == "-":
                known_pwm = reverse_complement_pwm(known_pwm)
            de_novo, crop_start, crop_end = crop_de_novo_pwm(de_novo_full, len(known_pwm))
            display_start_1based, display_end_1based = DE_NOVO_DISPLAY_RANGES_1BASED[
                spec.query_id
            ]
            if not (
                1 <= display_start_1based <= display_end_1based <= len(de_novo)
            ):
                raise IndexError(
                    f"{spec.query_id}: requested displayed de novo range "
                    f"{display_start_1based}-{display_end_1based} is outside "
                    f"the {len(de_novo)}-bp automatic display window"
                )
            automatic_crop_start = crop_start
            de_novo = de_novo[display_start_1based - 1 : display_end_1based]
            crop_start = automatic_crop_start + display_start_1based - 1
            crop_end = automatic_crop_start + display_end_1based
            counts = region_counts(manifest.iloc[seqlet_idx]["sample_id"])
            n_seqlets = int(len(seqlet_idx))

            rows.append(
                {
                    "query_id": spec.query_id,
                    "tf_name": spec.tf_name,
                    "jaspar_name": jaspar_name,
                    "target_id": spec.target_id,
                    "class": spec.pattern_class,
                    "orientation": orientation,
                    "q_value": float(hit["q-value"]),
                    "n_seqlets": n_seqlets,
                    "de_novo_pwm": de_novo,
                    "known_pwm": known_pwm,
                    "requested_display_start_1based": display_start_1based,
                    "requested_display_end_1based": display_end_1based,
                    "displayed_de_novo_length": int(len(de_novo)),
                    "crop_start_0based": crop_start,
                    "crop_end_exclusive": crop_end,
                    **{f"{region}_seqlets": counts[region] for region in REGIONS},
                    **{
                        f"{region}_percent": 100.0 * counts[region] / n_seqlets
                        for region in REGIONS
                    },
                }
            )
    return rows


def draw_figure(rows: list[dict[str, object]], output: Path) -> None:
    configure_style()
    n = len(rows)
    fig = plt.figure(figsize=(10.4, 4.0))
    gs = fig.add_gridspec(
        nrows=n,
        ncols=6,
        width_ratios=[1.05, 1.75, 1.75, 0.55, 1.35, 2.25],
        left=0.035,
        right=0.985,
        top=0.82,
        bottom=0.20,
        wspace=0.16,
        hspace=0.14,
    )
    headers = [
        "Most similar\nknown TF",
        "De novo motif",
        "Most similar\nknown motif",
        "Class",
        "# seqlets\n(sqrt scale)",
        "% seqlets\n(region type)",
    ]
    max_sqrt_n = max(np.sqrt(float(row["n_seqlets"])) for row in rows)

    for row_index, row in enumerate(rows):
        axes = [fig.add_subplot(gs[row_index, column]) for column in range(6)]
        if row_index == 0:
            for ax, header in zip(axes, headers):
                ax.set_title(header, fontsize=14, pad=8)

        axes[0].text(0.96, 0.5, row["tf_name"], ha="right", va="center", fontsize=12)
        axes[0].set_axis_off()
        draw_logo(axes[1], row["de_novo_pwm"])
        draw_logo(axes[2], row["known_pwm"])

        class_symbol = "+" if row["class"] == "positive" else "−"
        class_color = "#E53935" if row["class"] == "positive" else "#2468C5"
        axes[3].text(0.5, 0.5, class_symbol, color=class_color, ha="center", va="center", fontsize=16)
        axes[3].set_axis_off()

        n_seqlets = int(row["n_seqlets"])
        axes[4].set_facecolor("#F6F6F6")
        axes[4].barh([0], [np.sqrt(n_seqlets)], color="#F6B262", height=0.58)
        axes[4].text(
            np.sqrt(n_seqlets) + 0.025 * max_sqrt_n,
            0,
            f"{n_seqlets:,}",
            ha="left",
            va="center",
            fontsize=10,
        )
        axes[4].set_xlim(0, max_sqrt_n * 1.30)
        axes[4].set_ylim(-0.55, 0.55)
        axes[4].set_axis_off()

        left = 0.0
        for region in REGIONS:
            fraction = float(row[f"{region}_percent"]) / 100.0
            axes[5].barh(
                [0],
                [fraction],
                left=[left],
                height=0.58,
                color=REGION_COLORS[region],
                edgecolor="white",
                linewidth=0.6,
            )
            left += fraction
        axes[5].set_xlim(0, 1)
        axes[5].set_ylim(-0.55, 0.55)
        axes[5].set_axis_off()

    legend_handles = [
        mpl.patches.Patch(facecolor=REGION_COLORS[region], label=region)
        for region in REGIONS
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower right",
        bbox_to_anchor=(0.985, 0.025),
        ncol=4,
        frameon=False,
        handlelength=1.0,
        handletextpad=0.35,
        columnspacing=0.9,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def write_summary(rows: list[dict[str, object]], output: Path) -> None:
    records = []
    for row in rows:
        records.append(
            {
                key: value
                for key, value in row.items()
                if key not in {"de_novo_pwm", "known_pwm"}
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output, sep="\t", index=False)


def main() -> int:
    args = parse_args()
    if args.output.suffix.lower() != ".pdf":
        raise ValueError("--output must end with .pdf")
    rows = load_rows()
    draw_figure(rows, args.output)
    write_summary(rows, args.summary)
    print(f"Saved: {args.output}")
    print(f"Summary: {args.summary}")
    for row in rows:
        print(
            f"{row['query_id']} -> {row['tf_name']} ({row['target_id']}), "
            f"class={row['class']}, orientation={row['orientation']}, "
            f"q={row['q_value']:.4g}, n={row['n_seqlets']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
