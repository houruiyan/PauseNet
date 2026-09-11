#!/usr/bin/env python3
"""Six-panel seqlet heatmap around TF-MoDISco pattern motif centres (HEK293T NET-seq).

Works for both TF-MoDISco branches: --task profile (default, pattern_6 etc.)
or --task count. Rows = seqlets of the chosen pattern, columns = position
relative to motif centre (+/-400 bp). Panels:
  1) G4Hscore            - G4Hunter (Bedrat 2016), non-template strand, 25-nt
                           sliding mean; positive part clipped to [0,1] - Greens
  2) PauseNet predicted  - predicted_profiles x predicted_counts  - Blues
  3) NET-seq signal      - merged pos/neg bigWig by window strand - Reds
     (from the real NET-seq bigWigs, independent of the eval npz)
  4) GRO-seq signal      - merged pos/neg bigWig by window strand - Purples
  5) Pol II ChIP-seq     - merged all bigWig                      - Oranges
  6) PRO-seq signal      - GSM3073997 plus/minus bigWig           - YlGnBu

Coordinate bridge:
  seqlet -> example_idx -> <task>_selected_dataset_indices.npy -> dataset row
  -> manifest.tsv (chrom/input_start/input_end/strand; input window = 2114 bp,
  stored transcription-oriented; modisco crop = central 1000 bp, offset 557).
  seqlet genomic centre (+) = input_start + 557 + c ; (-) = input_end - (557 + c).

Rows sorted by total G4Hunter score across the window (descending; PauseNet
predicted total breaks ties).
Outputs: <task>_pattern<N>_seqlet_heatmap.pdf / .png.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pyBigWig

HERE = Path(__file__).resolve().parent
RES = Path("/mnt/HDD8TB/houruiyan/pausing_site/3_model_explanation/TFMoDISco/results/hek293t_netseq")
# per-task inputs: --task switches between profile/count TF-MoDISco branches
TASK_INPUTS = {
    "profile": {
        "h5": RES / "profile/profile_tfmodisco_patterns.h5",
        "bridge": RES / "profile/profile_selected_dataset_indices.npy",
    },
    "count": {
        "h5": RES / "count/count_tfmodisco_patterns.h5",
        "bridge": RES / "count/count_selected_dataset_indices.npy",
    },
}
DATA = Path("/mnt/HDD8TB/houruiyan/pausing_site/data")
MANIFEST = DATA / "HEK293T_NETseq/dataset/test/manifest.tsv"
CODES = DATA / "HEK293T_NETseq/dataset/test/sequence_codes.npy"
PROFILES = Path("/mnt/HDD8TB/houruiyan/pausing_site/2_train_model/evaluate/hek293t_netseq/test_profiles.npz")
BW_POLII = DATA / "HEK293T_POLIIseq/merge/merged/HEK293T_POLIIseq.merged.all.bw"
BW_NET_POS = DATA / "HEK293T_NETseq/merge/merged/HEK293T.merged.pos.bw"
BW_NET_NEG = DATA / "HEK293T_NETseq/merge/merged/HEK293T.merged.neg.bw"
BW_GRO_POS = DATA / "HEK293T_GROseq/merge/merged/HEK293T_GROseq.merged.pos.bw"
BW_GRO_NEG = DATA / "HEK293T_GROseq/merge/merged/HEK293T_GROseq.merged.neg.bw"
BW_PRO_POS = DATA / "HEK293T_PROseq/GSM3073997_PRO_DMSO_293T_rep1.plus.bw"
BW_PRO_NEG = DATA / "HEK293T_PROseq/GSM3073997_PRO_DMSO_293T_rep1.minus.bw"

INPUT_LEN = 2114
CROP_LEN = 1000
CROP_START = (INPUT_LEN - CROP_LEN) // 2          # 557
HALF = 400
BIN = 5
NBINS = 2 * HALF // BIN                            # 160
G4_WIN = 25

CODE_TO_BASE = np.asarray(list("ACGT"))
_RUN_RE = re.compile(r"G+|C+")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=["profile", "count"], default="profile",
                   help="Which TF-MoDISco branch to read seqlets from.")
    p.add_argument("--pattern-index", type=int, default=6)
    p.add_argument("--h5", type=Path, default=None,
                   help="Override patterns h5 (default: from --task).")
    p.add_argument("--bridge", type=Path, default=None,
                   help="Override bridge npy (default: from --task).")
    p.add_argument("--max-seqlets", type=int, default=None)
    p.add_argument("--output-prefix", type=Path, default=None)
    return p.parse_args()


def codes_to_seq(codes: np.ndarray) -> str:
    bases = np.full(codes.shape, "N", dtype="<U1")
    valid = codes < 4
    bases[valid] = CODE_TO_BASE[codes[valid].astype(np.intp)]
    return "".join(bases.tolist())


def g4hunter_track(seq: str, c_input: int) -> np.ndarray:
    """Return (2*HALF/BIN,) G4Hunter row at BIN-bp resolution, relative to centre."""
    base_score = np.zeros(len(seq), dtype=np.float64)
    for m in _RUN_RE.finditer(seq):
        rl = min(len(m.group()), 4)
        base_score[m.start():m.end()] = rl if m.group()[0] == "G" else -rl
    means = np.convolve(base_score, np.ones(G4_WIN) / G4_WIN, mode="valid")
    centers = np.arange(len(means)) + G4_WIN // 2          # input coords
    rel = centers - c_input                                # relative bp
    row = np.full(2 * HALF, np.nan)
    ok = (rel >= -HALF) & (rel < HALF)
    row[rel[ok] + HALF] = means[ok]
    return row.reshape(NBINS, BIN).mean(axis=1)            # NaN-safe: no NaNs here


def bw_track(bw, chrom: str, g: int, strand: str) -> np.ndarray:
    lo, hi = g - HALF, g + HALF
    clen = bw.chroms().get(chrom, 0)
    v = np.zeros(2 * HALF, dtype=np.float64)
    q_lo, q_hi = max(0, lo), min(clen, hi)
    if q_hi > q_lo:
        vals = np.nan_to_num(np.abs(bw.values(chrom, q_lo, q_hi, numpy=True).astype(np.float64)))
        v[q_lo - lo:q_hi - lo] = vals
    if strand == "-":
        v = v[::-1]
    return v.reshape(NBINS, BIN).mean(axis=1)


def normalize_rows(m: np.ndarray) -> np.ndarray:
    """Per-row normalization: divide each row by its own 99th percentile.

    Every row appears equally bright; keeps shape, discards intensity
    differences between rows.
    """
    q = np.nanpercentile(m, 99, axis=1, keepdims=True)
    q = np.where(q <= 0, 1.0, q)
    return np.clip(np.nan_to_num(m) / q, 0, 1)


def main() -> int:
    args = parse_args()
    h5_path = args.h5 or TASK_INPUTS[args.task]["h5"]
    bridge_path = args.bridge or TASK_INPUTS[args.task]["bridge"]
    output_prefix = args.output_prefix or (
        HERE / f"{args.task}_pattern{args.pattern_index}_seqlet_heatmap")
    output_prefix = output_prefix.resolve()
    print(f"[input] task={args.task} h5={h5_path} bridge={bridge_path}", flush=True)

    bridge = np.load(bridge_path)
    codes = np.load(CODES, mmap_mode="r")

    man = {}
    with open(MANIFEST) as f:
        header = f.readline().rstrip("\n").split("\t")
        ia = {k: header.index(k) for k in ("array_index", "chrom", "input_start", "input_end", "strand")}
        for line in f:
            p = line.rstrip("\n").split("\t")
            man[int(p[ia["array_index"]])] = (p[ia["chrom"]], int(p[ia["input_start"]]),
                                              int(p[ia["input_end"]]), p[ia["strand"]])
    print(f"manifest rows: {len(man)}", flush=True)

    with h5py.File(h5_path, "r") as h:
        g = h["pos_patterns"][f"pattern_{args.pattern_index}"]["seqlets"]
        ex = np.asarray(g["example_idx"], dtype=np.int64)
        st = np.asarray(g["start"], dtype=np.int64)
        en = np.asarray(g["end"], dtype=np.int64)
    n_all = len(ex)
    n = min(n_all, args.max_seqlets) if args.max_seqlets else n_all
    ex, st, en = ex[:n], st[:n], en[:n]
    drow = bridge[ex]
    c_crop = ((st + en) // 2).astype(np.int64)             # motif centre, 1kb crop coords
    print(f"{args.task} pattern_{args.pattern_index}: {n}/{n_all} seqlets; "
          f"centre range {c_crop.min()}..{c_crop.max()}", flush=True)

    # row order is decided AFTER the npz tracks are computed (needs G4/predicted
    # totals); bigWig extraction and plotting use the sorted arrays.

    # ---------- npz tracks (predicted / G4Hunter) ----------
    z = np.load(PROFILES)
    pred = z["predicted_profiles"][drow] * z["predicted_counts"][drow, None].astype(np.float64)
    m_pred = np.full((n, NBINS), np.nan)
    m_g4 = np.full((n, NBINS), np.nan)
    for i in range(n):
        c = int(c_crop[i])
        lo, hi = max(0, c - HALF), min(CROP_LEN, c + HALF)
        # fill at 1bp then bin
        tmp_p = np.full(2 * HALF, np.nan); tmp_p[lo - c + HALF:hi - c + HALF] = pred[i, lo:hi]
        with np.errstate(invalid="ignore"):
            m_pred[i] = np.nanmean(tmp_p.reshape(NBINS, BIN), axis=1)
        seq = codes_to_seq(np.asarray(codes[drow[i]]))
        m_g4[i] = g4hunter_track(seq, CROP_START + c)
    print(f"npz tracks done; G4H raw percentiles: "
          f"50%={np.nanpercentile(m_g4, 50):.3f} 90%={np.nanpercentile(m_g4, 90):.3f} "
          f"99%={np.nanpercentile(m_g4, 99):.3f} max={np.nanmax(m_g4):.3f}", flush=True)

    # ---------- row ordering: total G4Hunter score desc, predicted breaks ties ----
    # Each bin stores a mean; multiplying by BIN recovers the positional total.
    g4_window_total = np.nansum(np.clip(m_g4, 0, None), axis=1) * BIN
    predicted_total = np.nansum(m_pred, axis=1)
    order = np.lexsort((-predicted_total, -np.nan_to_num(g4_window_total, nan=-1)))
    drow, c_crop = drow[order], c_crop[order]
    m_pred, m_g4 = m_pred[order], m_g4[order]
    print(f"rows sorted by window G4Hunter total (desc): "
          f"top={g4_window_total[order[0]]:.1f} bottom={g4_window_total[order[-1]]:.1f}",
          flush=True)

    # ---------- bigWig tracks ----------
    mats = {}
    for key, pos_bw, neg_bw, stranded in (
        ("net", BW_NET_POS, BW_NET_NEG, True),
        ("gro", BW_GRO_POS, BW_GRO_NEG, True),
        ("polii", BW_POLII, BW_POLII, False),
        ("pro", BW_PRO_POS, BW_PRO_NEG, True),
    ):
        bw_p = pyBigWig.open(str(pos_bw))
        bw_n = pyBigWig.open(str(neg_bw)) if stranded else bw_p
        m = np.zeros((n, NBINS), dtype=np.float32)
        for i in range(n):
            chrom, i_start, i_end, strand = man[int(drow[i])]
            c_in = CROP_START + int(c_crop[i])
            g = i_start + c_in if strand == "+" else i_end - c_in
            bw = bw_p if strand == "+" else bw_n
            m[i] = bw_track(bw, chrom, g, strand)
        mats[key] = m
        print(f"track {key}: mean row-max {m.max(axis=1).mean():.2f}", flush=True)
        bw_p.close()
        if stranded:
            bw_n.close()

    # ---------- plot ----------
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    g4_plot = np.clip(m_g4, 0, 1)     # positive (G-rich) part only, like the reference 0-1 scale
    panels = [
        (g4_plot, "G4Hscore", "Greens", 0, 1),
        (normalize_rows(m_pred), "PauseNet predicted signal", "Blues", 0, 1),
        (normalize_rows(mats["net"]), "NET-seq signal", "Reds", 0, 1),
        (normalize_rows(mats["gro"]), "GRO-seq signal", "Purples", 0, 1),
        (normalize_rows(mats["polii"]), "Pol II ChIP-seq signal", "Oranges", 0, 1),
        (normalize_rows(mats["pro"]), "PRO-seq signal", "YlGnBu", 0, 1),
    ]
    fig, axes = plt.subplots(1, 6, figsize=(11.5, 5.2), sharey=True)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.90, bottom=0.13, wspace=0.42)
    for ax, (mat, title, cmap, vmin, vmax) in zip(axes, panels):
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                       origin="upper", interpolation="none",
                       extent=[-HALF, HALF, len(mat), 0])
        ax.axvline(0, color="grey", lw=0.8, ls="--")
        ax.set_xticks([-400, -200, 0, 200, 400])
        ax.tick_params(axis="x", labelsize=9, length=2)
        ax.tick_params(axis="y", labelsize=12, length=2)
        ax.set_title(title, fontsize=14)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=11, length=2)
        cb.outline.set_linewidth(0.5)
    axes[0].set_ylabel("Seqlets", fontsize=14)
    fig.supxlabel("Position relative to motif center (bp)", fontsize=15, y=0.02)
    fig.savefig(f"{output_prefix}.pdf", bbox_inches="tight")
    fig.savefig(f"{output_prefix}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {output_prefix}.pdf")
    print(f"Wrote: {output_prefix}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
