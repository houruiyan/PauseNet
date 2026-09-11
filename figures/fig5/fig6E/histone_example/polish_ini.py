#!/usr/bin/env python3
"""Post-process histone_tracks.ini: paired colors, clean titles, shared y per assay."""
import re
import numpy as np
import pyBigWig

INI = "histone_tracks.ini"
CHROM, START, END, NBINS = "chr12", 9_880_000, 10_080_000, 700

PAIRS = [
    # (section_k562, section_hela, bw_k, bw_h, title_prefix)
    ("K562_NETseq.hg19", "HeLaS3_NETseq.hg19",
     "./K562_NETseq.hg19.bw", "./HeLaS3_NETseq.hg19.bw", "NET-seq"),
    ("K562_H3K36me3.hg19", "HeLaS3_H3K36me3.hg19",
     "./K562_H3K36me3.hg19.bw", "./HeLaS3_H3K36me3.hg19.bw", "H3K36me3"),
    ("K562_H3K79me2.hg19", "HeLaS3_H3K79me2.hg19",
     "./K562_H3K79me2.hg19.bw", "./HeLaS3_H3K79me2.hg19.bw", "H3K79me2"),
    ("K562_H3K9me3.hg19", "HeLaS3_H3K9me3.hg19",
     "./K562_H3K9me3.hg19.bw", "./HeLaS3_H3K9me3.hg19.bw", "H3K9me3"),
]
GENE_KEYS = {
    "all_genes_transcripts": {
        "title": "",
        "labels": "true",
        "prefered_name": "gene_name",
        "fontsize": "11",
        "style": "UCSC",
        "max_labels": "60",
        "all_labels_inside": "true",
        "height": "3",
    },
    "CD69_transcript": {"title": "CD69", "fontsize": "10"},
    "KLRF1_transcript": {"title": "KLRF1", "fontsize": "10"},
}
C_K, C_H = "#2166ac", "#b2182b"


def bin_means(path):
    bw = pyBigWig.open(path)
    step = (END - START) / NBINS
    v = np.array([bw.stats(CHROM, int(START + i * step), int(START + (i + 1) * step),
                           type="mean")[0] for i in range(NBINS)], dtype=float)
    bw.close()
    return np.nan_to_num(v, nan=0.0)


def robust_ymax(path_a, path_b):
    """99.5th percentile of combined bin means, +15% headroom (spike-proof)."""
    v = np.concatenate([bin_means(path_a), bin_means(path_b)])
    nz = v[v > 0]
    ref = np.percentile(nz, 99.5) if len(nz) else v.max()
    return round(float(ref) * 1.15, 1)


text = open(INI).read()
sections = re.split(r"(?m)^(?=\[)", text)


def set_keys(block, keys):
    for k, v in keys.items():
        if re.search(rf"(?m)^{k}\s*=", block):
            block = re.sub(rf"(?m)^{k}\s*=.*$", f"{k} = {v}", block)
        else:
            lines = block.rstrip("\n").split("\n")
            lines.append(f"{k} = {v}")
            block = "\n".join(lines) + "\n"
    return block


out = []
for block in sections:
    m = re.match(r"\[(.+?)\]", block)
    if m:
        name = m.group(1)
        if name in GENE_KEYS:
            block = set_keys(block, GENE_KEYS[name])
            out.append(block)
            continue
    out.append(block)
text = "".join(out)

for sk, sh, bwk, bwh, prefix in PAIRS:
    ymax = robust_ymax(bwk, bwh)
    for sec, color, cell in ((sk, C_K, "K562"), (sh, C_H, "HeLaS3")):
        pat = re.compile(rf"(?ms)^(\[{re.escape(sec)}\].*?)(?=^\[|\Z)")
        m = pat.search(text)
        if not m:
            print(f"WARN: section {sec} not found")
            continue
        block = set_keys(m.group(1), {
            "color": color,
            "title": f"{prefix}\n    {cell}",
            "min_value": "0",
            "max_value": f"{ymax:g}",
        })
        text = text[: m.start(1)] + block + text[m.end(1):]
    print(f"{prefix}: shared max_value = {ymax:g}")

open(INI, "w").write(text)
print("INI polished:", INI)
