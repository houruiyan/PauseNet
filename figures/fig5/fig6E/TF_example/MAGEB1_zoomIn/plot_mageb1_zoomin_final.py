#!/usr/bin/env python3
"""MAGEB1 zoom-in, final-figure-small edition.
chrX:30,265,063-30,265,363 (MAGEB1 MANE-Select TSS -200..+100, hg19).
Top: DeepSHAP profile contribution logo; bottom: K562 CTCF per-base bars.
FIMO CTCF motif hits inside the window are highlighted in light yellow.
Canvas 15x4 inch, PDF only.
"""
import sys
sys.path.insert(0, ".")
from final_figure_small import apply_style, save_pdf

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch
from matplotlib.font_manager import FontProperties
from matplotlib.transforms import Affine2D
import pyBigWig

DEEPSHAP_DIR = "/mnt/HDD8TB/houruiyan/pausing_site/3_model_explanation/DeepSHAP/k562_mnetseq_TF/run_deepshap"
CTCF_BW = "/mnt/HDD8TB/houruiyan/pausing_site/data/TF/CTCF/ENCFF979PWH_K562_hg19.bigWig"
CHROM = "chrX"
REGION_START = 30185000
Z0, Z1 = 30_265_063, 30_265_363   # 0-based half-open
TSS = 30265263                     # MAGEB1 MANE Select TSS
# FIMO CTCF hits within the window: keep only the middle one (used by model)
MOTIFS = [
    (30_265_153, 30_265_172, "CTCF motif (FIMO p=5.7e-05, minus strand)"),
]
HL = "#fdf0b8"

s0, s1 = Z0 - REGION_START, Z1 - REGION_START
L = s1 - s0
x = np.arange(Z0, Z1)

scores = np.load(f"{DEEPSHAP_DIR}/region_profile_contribution_scores.npy")[:, s0:s1]
bw = pyBigWig.open(CTCF_BW)
ctcf = np.nan_to_num(bw.values(CHROM, Z0, Z1, numpy=True), nan=0.0)
bw.close()

BASES = "ACGT"
COLORS = {"A": "#109648", "C": "#255C99", "G": "#D97706", "T": "#C6333B"}
fp = FontProperties(family="DejaVu Sans", weight="bold")
pos_h = np.clip(scores, 0, None).sum(axis=0)
neg_h = np.clip(scores, None, 0).sum(axis=0)
ymax, ymin = pos_h.max() * 1.2, neg_h.min() * 1.2

apply_style()
fig, axes = plt.subplots(2, 1, figsize=(15, 4), sharex=True,
                         gridspec_kw={"height_ratios": [2.2, 1]})

# --- top: contribution logo ---
ax = axes[0]
for i in range(L):
    vals = scores[:, i]
    order = np.argsort(np.abs(vals))
    y_pos, y_neg = 0.0, 0.0
    for j in order:
        v = vals[j]
        if abs(v) < 1e-9:
            continue
        tp = TextPath((0, 0), BASES[j], size=1, prop=fp)
        bb = tp.get_extents()
        sy = abs(v) / max(bb.height, 1e-9)
        sx = 0.9 / max(bb.width, 1e-9)
        t = (Affine2D().scale(sx, sy)
             .translate(Z0 + i + 0.05 - bb.x0 * sx,
                        (y_pos if v > 0 else y_neg) - bb.y0 * sy))
        ax.add_patch(PathPatch(tp, transform=t + ax.transData,
                               color=COLORS[BASES[j]], lw=0))
        if v > 0:
            y_pos += v
        else:
            y_neg += v
ax.set_ylim(ymin, ymax)
ax.axhline(0, color="grey", lw=0.5)
ax.axvline(TSS, color="black", ls="--", lw=1)
for m0, m1, label in MOTIFS:
    ax.axvspan(m0, m1, color=HL, zorder=0)
    if label:
        ax.text((m0 + m1) / 2, ymax * 0.97, label,
                ha="center", va="top", fontsize=12, fontweight="bold",
                color="#8a6d00")
ax.text(TSS + 3, ymax * 0.82, "TSS", fontsize=12, va="top", fontweight="bold")
ax.text(TSS - 3, ymax * 0.82, "MAGEB1", fontsize=12, va="top", ha="right",
        fontweight="bold", style="italic")
ax.set_ylabel("profile contribution\n(per letter)")

# --- bottom: CTCF per-base bars ---
ax = axes[1]
ax.bar(x, ctcf, width=1.0, color="#444444", lw=0)
ax.axvline(TSS, color="black", ls="--", lw=1)
for m0, m1, label in MOTIFS:
    ax.axvspan(m0, m1, color=HL, zorder=0)
    ax.add_patch(plt.Rectangle((m0, 0), m1 - m0, max(ctcf.max(), 1) * 0.05,
                               color="#8a6d00", lw=0, zorder=3))
    if label:
        ax.text((m0 + m1) / 2, max(ctcf.max(), 1) * 0.075, "CTCF motif",
                ha="center", va="bottom", fontsize=12, color="#8a6d00",
                fontweight="bold", zorder=3)
ax.set_ylabel("CTCF\nK562")
ax.set_xlabel("chrX")

for a in axes:
    a.set_xlim(Z0 - 0.5, Z1 - 0.5)
ticks = np.arange(Z0, Z1, 50)
axes[1].set_xticks(ticks)
axes[1].set_xticklabels([f"{t:,}" for t in ticks])

fig.tight_layout()
save_pdf(fig, "MAGEB1_zoomIn_final.pdf")
print(f"CTCF max in window: {ctcf.max():.2f}, mean: {ctcf.mean():.2f}")
print(f"contribution max: {pos_h.max():.4f}")
