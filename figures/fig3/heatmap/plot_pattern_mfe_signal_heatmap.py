#!/usr/bin/env python3
"""Seqlet heatmap of DNA MFE and pausing signals around TF-MoDISco pattern seqlets.

The pattern is NOT hardcoded. Works for both TF-MoDISco tasks, e.g.
    python plot_pattern8_mfe_signal_heatmap.py --task profile --pattern-index 8
    python plot_pattern8_mfe_signal_heatmap.py --task count --sign pos --pattern-index 3
    python plot_pattern8_mfe_signal_heatmap.py --task count --sign neg --pattern-index 5

Rows = seqlets of the chosen pattern (TF-MoDISco, HEK293T NET-seq PauseNet),
columns = position relative to seqlet (motif) centre (+/-400 bp, 5-bp bins).

Panels:
  1) DNA MFE               - ViennaRNA fold_compound, DNA Mathews 2004 params,
                             30-nt sliding windows assigned to window centre,
                             on the transcription-oriented (non-template)
                             2114-bp input sequence. Absolute kcal/mol scale,
                             dark green = more stable.             - Greens_r
  2) PauseNet predicted    - predicted_profiles x predicted_counts  - Blues
  3) NET-seq signal        - merged pos/neg bigWig by window strand - Reds
  4) GRO-seq signal        - merged pos/neg bigWig by window strand - Purples
  5) Pol II ChIP-seq       - merged all bigWig                      - Oranges
  6) PRO-seq signal        - GSM3073997 plus/minus bigWig           - YlGnBu

Coordinate bridge (same as plot_pattern6_seqlet_heatmap.py):
  seqlet -> example_idx -> <task>_selected_dataset_indices.npy -> dataset row
  -> manifest.tsv (input window = 2114 bp, stored transcription-oriented;
  modisco crop = central 1000 bp, offset 557).
  seqlet genomic centre (+) = input_start + 557 + c ; (-) = input_end - (557 + c).

Rows sorted by mean MFE within the central +/-100 bp (ascending = most stable
structure on top; PauseNet predicted total breaks ties).

Outputs (this folder):
  pattern<N>_mfe_signal_heatmap.pdf / .png               (task=profile)
  count_<pos|neg>_pattern<N>_mfe_signal_heatmap.pdf/.png (task=count)
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from functools import lru_cache
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pyBigWig
import RNA

HERE = Path(__file__).resolve().parent
RES = Path("/mnt/HDD8TB/houruiyan/pausing_site/3_model_explanation/TFMoDISco/results/hek293t_netseq")
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

CODE_TO_BASE = np.asarray(list("ACGT"))

WORKER_WINDOW_SIZE = 30
WORKER_MD: "RNA.md | None" = None


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=["profile", "count"], default="profile",
                   help="which TF-MoDISco task the pattern comes from (default: profile)")
    p.add_argument("--sign", choices=["pos", "neg"], default="pos",
                   help="pos_patterns or neg_patterns; only used for --task count "
                        "(profile only has pos_patterns)")
    p.add_argument("--patterns-h5", type=Path, default=None,
                   help="override patterns h5 (default: RES/<task>/<task>_tfmodisco_patterns.h5)")
    p.add_argument("--bridge", type=Path, default=None,
                   help="override bridge npy (default: RES/<task>/<task>_selected_dataset_indices.npy)")
    p.add_argument("--pattern-index", type=int, required=True,
                   help="TF-MoDISco pattern index, e.g. --pattern-index 8")
    p.add_argument("--window-size", type=int, default=30, help="MFE folding window (nt)")
    p.add_argument("--temperature", type=float, default=37.0)
    p.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    p.add_argument("--max-seqlets", type=int, default=None)
    p.add_argument("--output-prefix", type=Path, default=None)
    return p.parse_args()


# ---------------- MFE folding (same logic as the p37 MFE+signal script) ----------------

def initialize_fold_worker(window_size: int, temperature: float) -> None:
    global WORKER_WINDOW_SIZE, WORKER_MD
    WORKER_WINDOW_SIZE = window_size
    RNA.params_load_DNA_Mathews2004()
    model = RNA.md()
    model.temperature = temperature
    model.dangles = 2
    WORKER_MD = model


@lru_cache(maxsize=500_000)
def dna_window_mfe(sequence: str) -> float:
    if WORKER_MD is None:
        raise RuntimeError("DNA folding worker was not initialized")
    fc = RNA.fold_compound(sequence, WORKER_MD)
    _, mfe = fc.mfe()
    return float(mfe)


def mfe_track(sequence: str) -> np.ndarray:
    """sequence covers centre-(HALF+w//2) .. centre+(HALF+w//2); return 2*HALF MFE
    values at relative positions -HALF..HALF-1 (window centres, 0.5-bp offset)."""
    w = WORKER_WINDOW_SIZE
    out = np.full(2 * HALF, np.nan, dtype=np.float64)
    for i in range(len(sequence) - w + 1):
        win = sequence[i:i + w]
        if "N" in win:
            continue
        out[i] = dna_window_mfe(win)
    return out


def fold_region(job: tuple) -> tuple:
    seqlet_index, sequence = job
    return seqlet_index, mfe_track(sequence)


def codes_to_seq(codes: np.ndarray) -> str:
    bases = np.full(codes.shape, "N", dtype="<U1")
    valid = codes < 4
    bases[valid] = CODE_TO_BASE[codes[valid].astype(np.intp)]
    return "".join(bases.tolist())


# ---------------- bigWig track (same as reference heatmap script) ----------------

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


def bin_1bp(tmp: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.nanmean(tmp.reshape(NBINS, BIN), axis=1)


def normalize_rows(m: np.ndarray) -> np.ndarray:
    """Per-row normalization: divide each row by its own 99th percentile."""
    q = np.nanpercentile(m, 99, axis=1, keepdims=True)
    q = np.where(q <= 0, 1.0, q)
    return np.clip(np.nan_to_num(m) / q, 0, 1)


def main() -> int:
    args = parse_args()

    # resolve task-dependent inputs: <task>/<task>_tfmodisco_patterns.h5 + bridge
    task_dir = RES / args.task
    patterns_h5 = args.patterns_h5 or task_dir / f"{args.task}_tfmodisco_patterns.h5"
    bridge_path = args.bridge or task_dir / f"{args.task}_selected_dataset_indices.npy"
    group_name = "pos_patterns" if args.sign == "pos" else "neg_patterns"
    if args.task == "profile":
        label = f"profile pattern_{args.pattern_index}"
        default_prefix = HERE / f"pattern{args.pattern_index}_mfe_signal_heatmap"
    else:
        label = f"count {args.sign} pattern_{args.pattern_index}"
        default_prefix = HERE / f"count_{args.sign}_pattern{args.pattern_index}_mfe_signal_heatmap"

    output_prefix = args.output_prefix or default_prefix
    output_prefix = output_prefix.resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    bridge = np.load(bridge_path)
    codes = np.load(CODES, mmap_mode="r")
    assert codes.shape[1] == INPUT_LEN, f"unexpected input length {codes.shape}"

    man = {}
    with open(MANIFEST) as f:
        header = f.readline().rstrip("\n").split("\t")
        ia = {k: header.index(k) for k in ("array_index", "chrom", "input_start", "input_end", "strand")}
        for line in f:
            p = line.rstrip("\n").split("\t")
            man[int(p[ia["array_index"]])] = (p[ia["chrom"]], int(p[ia["input_start"]]),
                                              int(p[ia["input_end"]]), p[ia["strand"]])
    print(f"manifest rows: {len(man)}", flush=True)

    with h5py.File(patterns_h5, "r") as h:
        if group_name not in h:
            raise SystemExit(
                f"ERROR: {patterns_h5} has no {group_name} group "
                f"(available: {list(h.keys())})")
        pos = h[group_name]
        key = f"pattern_{args.pattern_index}"
        if key not in pos:
            available = sorted(int(k.split("_")[1]) for k in pos.keys()
                               if k.startswith("pattern_"))
            raise SystemExit(
                f"ERROR: {group_name}/{key} not found in {patterns_h5}\n"
                f"available pattern indices in {group_name}: {available}")
        g = pos[key]["seqlets"]
        ex = np.asarray(g["example_idx"], dtype=np.int64)
        st = np.asarray(g["start"], dtype=np.int64)
        en = np.asarray(g["end"], dtype=np.int64)
    n_all = len(ex)
    n = min(n_all, args.max_seqlets) if args.max_seqlets else n_all
    ex, st, en = ex[:n], st[:n], en[:n]
    drow = bridge[ex]
    c_crop = ((st + en) // 2).astype(np.int64)             # motif centre, 1kb crop coords
    print(f"{label}: {n}/{n_all} seqlets; "
          f"centre range {c_crop.min()}..{c_crop.max()}", flush=True)

    # ---------- MFE tracks ----------
    w = args.window_size
    flank = HALF + w // 2
    region_len = 2 * HALF + w - 1
    jobs = []
    for i in range(n):
        c_in = CROP_START + int(c_crop[i])
        region = np.asarray(codes[drow[i], c_in - flank:c_in - flank + region_len])
        jobs.append((i, codes_to_seq(region)))
    if args.jobs == 1:
        initialize_fold_worker(w, args.temperature)
        folded = [fold_region(j) for j in jobs]
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=args.jobs, initializer=initialize_fold_worker,
                      initargs=(w, args.temperature)) as pool:
            folded = list(pool.imap(fold_region, jobs, chunksize=8))
    mfe_1bp = np.full((n, 2 * HALF), np.nan)
    for i, tr in folded:
        mfe_1bp[i] = tr
    m_mfe = np.full((n, NBINS), np.nan)
    for i in range(n):
        m_mfe[i] = bin_1bp(mfe_1bp[i])
    print(f"MFE done: matrix range [{np.nanmin(m_mfe):.2f}, {np.nanmax(m_mfe):.2f}] kcal/mol",
          flush=True)

    # ---------- npz signal track (predicted) ----------
    z = np.load(PROFILES)
    pred = (z["predicted_profiles"][drow].astype(np.float64)
            * z["predicted_counts"][drow].astype(np.float64)[:, None])
    m_pred = np.full((n, NBINS), np.nan)
    for i in range(n):
        c = int(c_crop[i])
        lo, hi = max(0, c - HALF), min(CROP_LEN, c + HALF)
        tmp_p = np.full(2 * HALF, np.nan); tmp_p[lo - c + HALF:hi - c + HALF] = pred[i, lo:hi]
        m_pred[i] = bin_1bp(tmp_p)
    print("npz signal track done", flush=True)

    # ---------- row ordering: central mean MFE asc, predicted total breaks ties ----
    half100 = 100 // BIN
    centre = NBINS // 2
    mfe_centre = np.nanmean(m_mfe[:, centre - half100:centre + half100], axis=1)
    predicted_total = np.nansum(m_pred, axis=1)
    order = np.lexsort((-predicted_total, np.nan_to_num(mfe_centre, nan=np.inf)))
    drow, c_crop = drow[order], c_crop[order]
    m_mfe, m_pred = m_mfe[order], m_pred[order]
    print(f"rows sorted by central (+/-100 bp) mean MFE (asc): "
          f"top={mfe_centre[order[0]]:.2f} bottom={mfe_centre[order[-1]]:.2f} kcal/mol",
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
    mfe_vmin = float(np.nanpercentile(m_mfe, 1))
    panels = [
        (m_mfe, "DNA MFE (kcal/mol)", "Greens_r", mfe_vmin, 0),
        (normalize_rows(m_pred), "PauseNet predicted signal", "Blues", 0, 1),
        (normalize_rows(mats["net"]), "NET-seq signal", "Reds", 0, 1),
        (normalize_rows(mats["gro"]), "GRO-seq signal", "Purples", 0, 1),
        (normalize_rows(mats["polii"]), "Pol II ChIP-seq signal", "Oranges", 0, 1),
        (normalize_rows(mats["pro"]), "PRO-seq signal", "YlGnBu", 0, 1),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(2.0 * len(panels) + 0.5, 5.2), sharey=True)
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
    fig.suptitle(f"{label}  (n = {n} seqlets)", fontsize=15, y=0.97)
    fig.supxlabel("Position relative to motif center (bp)", fontsize=15, y=0.02)
    fig.savefig(f"{output_prefix}.pdf", bbox_inches="tight")
    fig.savefig(f"{output_prefix}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {output_prefix}.pdf")
    print(f"Wrote: {output_prefix}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
