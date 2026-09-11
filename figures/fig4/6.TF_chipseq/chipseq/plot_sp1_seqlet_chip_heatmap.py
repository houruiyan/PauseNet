#!/usr/bin/env python3
"""Seqlet-centered TF ChIP-seq heatmap.

Rows = seqlets of one TF-MoDISco pattern (default: count P8, SP1 motif),
centered on the seqlet; columns = genomic position around the center.
Panels = one per TF ChIP bigWig (default SP1 / SP2 / PATZ1), sharing the row
order (sorted by the matched TF's window signal). Per-row normalization:
each row divided by its own 99th percentile (shape preserved, rows comparable).

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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from final_figure_small import apply_style, save_pdf, FIGSIZE
from plot_seqlet_chipseq import (load_seqlet_centers, profile_matrix,
                                 TF_BW, TF_COLORS, TEST_MANIFEST)

CMAP = "Reds"


def row_normalize(mat, pct=99):
    scale = np.percentile(mat, pct, axis=1, keepdims=True)
    scale[scale <= 0] = 1.0
    return np.clip(mat / scale, 0, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="count", choices=["count", "profile"])
    p.add_argument("--pattern-index", type=int, default=8)
    p.add_argument("--matched-tf", default="SP1")
    p.add_argument("--tfs", nargs="+", default=["SP1", "SP2", "PATZ1"])
    p.add_argument("--half-win", type=int, default=2000)
    p.add_argument("--bin", type=int, default=20,
                   help="bp per heatmap column")
    p.add_argument("--max-seqlets", type=int, default=5000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent /
                   "seqlet_chip_heatmap.pdf")
    p.add_argument("--preview-png", action="store_true")
    a = p.parse_args()

    manifest = pd.read_csv(TEST_MANIFEST, sep="\t")
    chrom_arr, centers = load_seqlet_centers(a.task, a.pattern_index, manifest)
    rng = np.random.default_rng(a.seed)
    if a.max_seqlets and len(centers) > a.max_seqlets:
        sel = rng.choice(len(centers), a.max_seqlets, replace=False)
        chrom_arr, centers = chrom_arr[sel], centers[sel]
    n = len(centers)
    print(f"{a.task} P{a.pattern_index}: n={n} seqlets", flush=True)

    mats = {}
    for tf in a.tfs:
        bw = pyBigWig.open(str(TF_BW[tf]))
        m = profile_matrix(bw, chrom_arr, centers, a.half_win, bw.chroms())
        bw.close()
        # bin columns: (n, L) -> (n, L//bin)
        nb = m.shape[1] // a.bin
        m = m[:, :nb * a.bin].reshape(n, nb, a.bin).mean(axis=2)
        mats[tf] = m
        print(f"  {tf}: matrix {m.shape}", flush=True)

    # sort rows by matched TF CENTRAL signal (strongest center on top)
    nb = mats[a.matched_tf].shape[1]
    c0 = nb // 2
    half_core = max(1, int(250 / a.bin))
    order = np.argsort(-mats[a.matched_tf][:, c0 - half_core:c0 + half_core].sum(axis=1))
    for tf in a.tfs:
        mats[tf] = row_normalize(mats[tf][order])

    extent = [-a.half_win, a.half_win, n, 0]
    apply_style()
    fig, axes = plt.subplots(1, len(a.tfs), figsize=FIGSIZE, sharey=True)
    if len(a.tfs) == 1:
        axes = [axes]
    fig.subplots_adjust(left=0.04, right=0.90, bottom=0.15, top=0.90,
                        wspace=0.10)
    for i, (ax, tf) in enumerate(zip(axes, a.tfs)):
        im = ax.imshow(mats[tf], aspect="auto", cmap=CMAP, vmin=0, vmax=1,
                       extent=extent, interpolation="none")
        ax.set_title(tf, fontsize=12)
        ax.set_xticks([-a.half_win, 0, a.half_win])
        labels = [f"-{a.half_win // 1000}k", "0", f"+{a.half_win // 1000}k"]
        if i > 0:
            labels[0] = ""
        if i < len(a.tfs) - 1:
            labels[2] = ""
        ax.set_xticklabels(labels, fontsize=9)
        ax.axvline(0, color="k", ls="--", lw=0.5)
    axes[0].set_ylabel(f"seqlets (n={n})", fontsize=11)
    axes[0].set_yticks([])
    fig.text(0.5, 0.04, f"Position relative to {a.task} P{a.pattern_index} "
                        f"seqlet center (bp)", ha="center", fontsize=11)
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.75])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("row-normalized\nChIP signal", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    save_pdf(fig, a.out)
    if a.preview_png:
        fig.savefig(str(a.out.with_suffix(".png")), dpi=200)
        print("saved", a.out.with_suffix(".png"), flush=True)


if __name__ == "__main__":
    main()
