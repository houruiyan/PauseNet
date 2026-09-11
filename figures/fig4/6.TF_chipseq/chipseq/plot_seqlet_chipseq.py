#!/usr/bin/env python3
"""Seqlet-centered TF ChIP-seq validation: metaprofile (fig1) + enrichment (fig2).

For each (task, pattern) whose TomTom top hit is a TF with available ChIP-seq
bigWig, seqlets are mapped to genomic coordinates
(h5 example_idx -> selected_dataset_indices bridge -> test manifest ->
genome; seqlet start/end are relative to the central 1000 bp crop, offset 557).

Fig1 (seqlet_chipseq_metaprofile.pdf): mean ChIP signal around seqlet centers
  (+-1 kb) for the matched TF, an unmatched TF (specificity control), and the
  matched TF at +10 kb shifted windows (position control).

Fig2 (seqlet_chipseq_enrichment.pdf): log2 fold enrichment of central (+-100 bp)
  signal over shifted control, per pattern, matched vs unmatched TF;
  stars = Wilcoxon signed-rank test of per-seqlet central vs shifted signal.

Style: final-figure-small (5x4, PDF only; --preview-png for exploration).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import pyBigWig
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path("/mnt/HDD8TB/houruiyan/pausing_site")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from final_figure_small import apply_style, save_pdf, FIGSIZE

INPUT_LEN = 2114
CROP_START = (INPUT_LEN - 1000) // 2          # 557

TEST_MANIFEST = ROOT / "data/HEK293T_NETseq/dataset/test/manifest.tsv"
H5 = {t: ROOT / f"3_model_explanation/TFMoDISco/results/hek293t_netseq/{t}/{t}_tfmodisco_patterns.h5"
      for t in ["count", "profile"]}
BRIDGE = {t: ROOT / f"3_model_explanation/TFMoDISco/results/hek293t_netseq/{t}/{t}_selected_dataset_indices.npy"
          for t in ["count", "profile"]}

TF_BW = {
    "SP1":   ROOT / "data/TF/SP1/ENCFF478WJL_SP1_HEK293T_hg19.bigWig",
    "SP2":   ROOT / "data/TF/SP2/ENCFF763PKH_SP2_HEK293_hg19.bigWig",
    "PATZ1": ROOT / "data/TF/PATZ1/ENCFF083UPA_PATZ1_HEK293_hg19.bigWig",
}
TF_COLORS = {"SP1": "#0072B2", "SP2": "#009E73", "PATZ1": "#D55E00"}

# (task, pattern_idx, matched TF, TomTom q)  -- only TFs with local bigWig
PAIRS = [
    ("count",   8,  "SP1",   1.72e-03),
    ("count",   10, "PATZ1", 9.09e-04),
    ("profile", 19, "PATZ1", 6.22e-03),
    ("profile", 6,  "PATZ1", 8.73e-03),
    ("profile", 37, "PATZ1", 1.65e-02),
    ("count",   5,  "PATZ1", 8.95e-02),   # marginal hit, expect weaker
    ("profile", 8,  "PATZ1", 2.86e-01),   # non-sig hit, expect flat
]
UNMATCHED = {"SP1": "PATZ1", "PATZ1": "SP2", "SP2": "SP1"}


def load_seqlet_centers(task, pattern_idx, manifest):
    with h5py.File(H5[task], "r") as f:
        g = f[f"pos_patterns/pattern_{pattern_idx}/seqlets"]
        example_idx = np.asarray(g["example_idx"], dtype=np.int64)
        starts = np.asarray(g["start"], dtype=np.int64)
        ends = np.asarray(g["end"], dtype=np.int64)
    bridge = np.load(BRIDGE[task])
    dataset_idx = bridge[example_idx]
    rows = manifest.iloc[dataset_idx]
    chrom = rows["chrom"].to_numpy()
    centers = (rows["input_start"].to_numpy(dtype=np.int64)
               + CROP_START + (starts + ends) // 2)
    return chrom, centers


def window_mean(bw, chrom, center, half, chlen):
    s, e = center - half, center + half
    cs, ce = max(s, 0), min(e, chlen)
    if ce <= cs:
        return 0.0
    v = bw.values(chrom, cs, ce, numpy=True)
    return float(np.nan_to_num(np.abs(v), nan=0.0).mean())


def profile_matrix(bw, chroms_arr, centers, half, chroms):
    L = 2 * half + 1
    mat = np.zeros((len(centers), L))
    for i, (ch, c) in enumerate(zip(chroms_arr, centers)):
        if ch not in chroms:
            continue
        s, e = c - half, c + half + 1
        cs, ce = max(s, 0), min(e, chroms[ch])
        if ce > cs:
            v = bw.values(ch, cs, ce, numpy=True)
            v = np.nan_to_num(np.abs(v), nan=0.0)
            mat[i, cs - s: cs - s + len(v)] = v
    return mat


def smooth(y, w=21):
    return np.convolve(y, np.ones(w) / w, mode="same")


def star(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--half-win", type=int, default=1000)
    p.add_argument("--core-half", type=int, default=100,
                   help="central window for enrichment")
    p.add_argument("--shift", type=int, default=10000,
                   help="position-control shift (bp)")
    p.add_argument("--max-seqlets", type=int, default=3000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--pairs", nargs="*", default=None,
                   help="e.g. count:8:SP1 profile:19:PATZ1 (overrides PAIRS)")
    p.add_argument("--out-prefix", type=Path,
                   default=Path(__file__).resolve().parent / "seqlet_chipseq")
    p.add_argument("--cache", type=Path, default=Path(__file__).resolve().parent /
                   "seqlet_chipseq_cache.npz")
    p.add_argument("--preview-png", action="store_true")
    a = p.parse_args()

    pairs = PAIRS
    if a.pairs:
        pairs = []
        for spec in a.pairs:
            t, i, tf = spec.split(":")
            q = next((q for (t2, i2, tf2, q) in PAIRS
                      if t2 == t and i2 == int(i) and tf2 == tf), np.nan)
            pairs.append((t, int(i), tf, q))

    manifest = pd.read_csv(TEST_MANIFEST, sep="\t")
    assert (manifest["array_index"].to_numpy() == np.arange(len(manifest))).all()

    bws = {}
    for tf in {tf for _, _, tf, _ in pairs} | {UNMATCHED[tf] for _, _, tf, _ in pairs}:
        bws[tf] = pyBigWig.open(str(TF_BW[tf]))
    chroms = next(iter(bws.values())).chroms()

    rng = np.random.default_rng(a.seed)
    results = {}
    for (task, pidx, tf, q) in pairs:
        key = f"{task}:P{pidx}:{tf}"
        chrom_arr, centers = load_seqlet_centers(task, pidx, manifest)
        if a.max_seqlets and len(centers) > a.max_seqlets:
            sel = rng.choice(len(centers), a.max_seqlets, replace=False)
            chrom_arr, centers = chrom_arr[sel], centers[sel]
        n = len(centers)
        print(f"{key}: n={n}", flush=True)

        prof = {}
        prof["matched"] = profile_matrix(bws[tf], chrom_arr, centers,
                                         a.half_win, chroms)
        tf_un = UNMATCHED[tf]
        prof["unmatched"] = profile_matrix(bws[tf_un], chrom_arr, centers,
                                           a.half_win, chroms)
        prof["shifted"] = profile_matrix(bws[tf], chrom_arr,
                                         centers + a.shift, a.half_win, chroms)
        # per-seqlet central signals for stats
        core = {k: m[:, a.half_win - a.core_half: a.half_win + a.core_half].mean(axis=1)
                for k, m in prof.items()}
        wstat, pval = stats.wilcoxon(core["matched"], core["shifted"])
        enr_m = np.log2((core["matched"].mean() + 1e-3) /
                        (core["shifted"].mean() + 1e-3))
        enr_u = np.log2((core["unmatched"].mean() + 1e-3) /
                        (core["shifted"].mean() + 1e-3))
        results[key] = dict(task=task, pidx=pidx, tf=tf, tf_un=tf_un, q=q, n=n,
                            prof={k: m.mean(axis=0) for k, m in prof.items()},
                            enr_m=enr_m, enr_u=enr_u, pval=pval)
        print(f"  enrich matched={enr_m:+.2f} unmatched={enr_u:+.2f} "
              f"wilcoxon p={pval:.1e}", flush=True)

    np.savez_compressed(a.cache, meta=[(k, v["enr_m"], v["enr_u"], v["pval"])
                                       for k, v in results.items()])

    # ================= Fig 1: metaprofiles (2x2, first 4 pairs) =============
    x = np.arange(-a.half_win, a.half_win + 1)
    apply_style()
    show = list(results.items())[:4]
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE)
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.12, top=0.88,
                        wspace=0.28, hspace=0.55)
    for ax, (key, r) in zip(axes.flat, show):
        ax.plot(x, smooth(r["prof"]["matched"]), color=TF_COLORS[r["tf"]],
                lw=1.5, label=f'{r["tf"]} (matched)')
        ax.plot(x, smooth(r["prof"]["unmatched"]), color="#888888", lw=1.2,
                label=f'{r["tf_un"]} (unmatched)')
        ax.plot(x, smooth(r["prof"]["shifted"]), color=TF_COLORS[r["tf"]],
                lw=1.2, ls="--", alpha=0.6, label="shifted +10 kb")
        ax.axvline(0, color="k", ls=":", lw=0.8)
        ax.set_title(f'{r["task"]} P{r["pidx"]} -> {r["tf"]} '
                     f'(q={r["q"]:.0e})', fontsize=10)
        ax.margins(x=0)
    axes[0, 0].set_ylabel("ChIP-seq signal")
    axes[1, 0].set_ylabel("ChIP-seq signal")
    for ax in axes[1]:
        ax.set_xlabel("Position (bp)")
    axes[0, 0].legend(frameon=False, fontsize=7.5, loc="upper right",
                      borderpad=0.1, labelspacing=0.15, handlelength=1.4)
    out1 = a.out_prefix.parent / (a.out_prefix.name + "_metaprofile.pdf")
    save_pdf(fig, out1)
    if a.preview_png:
        fig.savefig(str(out1.with_suffix(".png")), dpi=200)
    plt.close(fig)

    # ================= Fig 2: enrichment bars (all pairs) ==================
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(left=0.14, right=0.97, bottom=0.30, top=0.95)
    keys = list(results.keys())
    xs = np.arange(len(keys))
    w = 0.36
    for i, key in enumerate(keys):
        r = results[key]
        ax.bar(i - w / 2, r["enr_m"], width=w, color=TF_COLORS[r["tf"]],
               alpha=0.9)
        ax.bar(i + w / 2, r["enr_u"], width=w, color="#aaaaaa", alpha=0.9)
        ax.text(i - w / 2, max(r["enr_m"], 0) + 0.05, star(r["pval"]),
                ha="center", va="bottom", fontsize=10)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{results[k]['task'][:1].upper()}P{results[k]['pidx']}\n"
                        f"{results[k]['tf']}" for k in keys],
                       fontsize=9, rotation=0)
    ax.set_ylabel("log2 enrichment\n(seqlet vs +10 kb)")
    handles = [Line2D([], [], marker="s", ls="", color="#555555",
                      label="matched TF"),
               Line2D([], [], marker="s", ls="", color="#aaaaaa",
                      label="unmatched TF")]
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="upper right")
    out2 = a.out_prefix.parent / (a.out_prefix.name + "_enrichment.pdf")
    save_pdf(fig, out2)
    if a.preview_png:
        fig.savefig(str(out2.with_suffix(".png")), dpi=200)
    plt.close(fig)

    for tf in bws.values():
        tf.close()


if __name__ == "__main__":
    main()
