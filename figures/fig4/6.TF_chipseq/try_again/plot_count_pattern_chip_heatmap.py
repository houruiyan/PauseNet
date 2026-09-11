#!/usr/bin/env python3
"""Seqlet-centered ChIP-seq heatmaps for count positive patterns 2/7/8/10.

One panel per pattern; rows = seqlets centered on the seqlet midpoint,
+-500 bp window; rows sorted top-to-bottom by the matched TF's ChIP-seq
signal at the center (+-50 bp). Per-row normalization (99th percentile).

Pattern -> matched TF (from TomTom, best available local bigWig):
  P2  -> PATZ1 (q=3.3e-2)   [top hit ZNF93 has no local ChIP]
  P7  -> PATZ1 (q=4.8e-2)
  P8  -> SP1   (q=1.7e-3)
  P10 -> PATZ1 (q=9.1e-4)

Style: final-figure-small (5x4, PDF only; --preview-png for exploration).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyBigWig
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/mnt/HDD8TB/houruiyan/pausing_site")
CHIPSEQ_DIR = ROOT / "4_plot_figure/fig3/6.TF_chipseq/chipseq"
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(CHIPSEQ_DIR))

from final_figure_small import apply_style, save_pdf, FIGSIZE
from plot_seqlet_chipseq import (load_seqlet_centers, profile_matrix,
                                 TF_BW, TEST_MANIFEST)

TF_BW = dict(TF_BW)
TF_BW["ZNF610"] = ROOT / "data/TF/ZNF610/ENCFF114HMS_ZNF610_HEK293_hg19.bigWig"

# (pattern_idx, matched TF, TomTom q)
PATTERNS = [
    (2,  "PATZ1", 3.31e-02),
    (7,  "PATZ1", 4.83e-02),
    (8,  "SP1",   1.72e-03),
    (10, "PATZ1", 9.09e-04),
]

CMAP = "Reds"


def row_normalize(mat, pct=99):
    scale = np.percentile(mat, pct, axis=1, keepdims=True)
    scale[scale <= 0] = 1.0
    return np.clip(mat / scale, 0, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="count", choices=["count", "profile"])
    p.add_argument("--half-win", type=int, default=500)
    p.add_argument("--bin", type=int, default=10)
    p.add_argument("--sort-half", type=int, default=50,
                   help="central window (bp) used for sorting")
    p.add_argument("--max-seqlets", type=int, default=5000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent /
                   "count_pattern_seqlet_chip_heatmap.pdf")
    p.add_argument("--preview-png", action="store_true")
    a = p.parse_args()

    manifest = pd.read_csv(TEST_MANIFEST, sep="\t")
    rng = np.random.default_rng(a.seed)

    bw_cache = {}
    panels = []
    for pidx, tf, q in PATTERNS:
        if tf not in bw_cache:
            bw_cache[tf] = pyBigWig.open(str(TF_BW[tf]))
        bw = bw_cache[tf]
        chrom_arr, centers = load_seqlet_centers(a.task, pidx, manifest)
        if a.max_seqlets and len(centers) > a.max_seqlets:
            sel = rng.choice(len(centers), a.max_seqlets, replace=False)
            chrom_arr, centers = chrom_arr[sel], centers[sel]
        n = len(centers)
        m = profile_matrix(bw, chrom_arr, centers, a.half_win, bw.chroms())
        nb = m.shape[1] // a.bin
        m = m[:, :nb * a.bin].reshape(n, nb, a.bin).mean(axis=2)
        # sort by central signal
        c0 = nb // 2
        hs = max(1, a.sort_half // a.bin)
        order = np.argsort(-m[:, c0 - hs:c0 + hs].sum(axis=1))
        m = row_normalize(m[order])
        center_signal = m[:, c0 - hs:c0 + hs].sum(axis=1)
        panels.append((pidx, tf, q, n, m))
        frac_bound = float((m[:, c0 - hs:c0 + hs].sum(axis=1) > 0).mean())
        print(f"P{pidx} -> {tf}: n={n}, rows with center signal: "
              f"{frac_bound * 100:.0f}%", flush=True)
    for bw in bw_cache.values():
        bw.close()

    apply_style()
    fig, axes = plt.subplots(1, len(panels), figsize=FIGSIZE, sharey=True)
    fig.subplots_adjust(left=0.04, right=0.90, bottom=0.15, top=0.88,
                        wspace=0.10)
    extent = [-a.half_win, a.half_win, None, 0]
    for i, (ax, (pidx, tf, q, n, m)) in enumerate(zip(axes, panels)):
        extent = [-a.half_win, a.half_win, n, 0]
        im = ax.imshow(m, aspect="auto", cmap=CMAP, vmin=0, vmax=1,
                       extent=extent, interpolation="none")
        ax.set_title(f"P{pidx} -> {tf}\n(n={n}, q={q:.0e})", fontsize=9)
        ax.set_xticks([-a.half_win, 0, a.half_win])
        labels = ["-500", "0", "+500"]
        if i > 0:
            labels[0] = ""
        if i < len(panels) - 1:
            labels[2] = ""
        ax.set_xticklabels(labels, fontsize=8)
        ax.axvline(0, color="k", ls="--", lw=0.5)
    axes[0].set_yticks([])
    fig.text(0.5, 0.04, "Position relative to seqlet center (bp)",
             ha="center", fontsize=11)
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.73])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("row-normalized\nChIP signal", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    save_pdf(fig, a.out)
    if a.preview_png:
        fig.savefig(str(a.out.with_suffix(".png")), dpi=200)
        print("saved", a.out.with_suffix(".png"), flush=True)


if __name__ == "__main__":
    main()
