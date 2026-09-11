#!/usr/bin/env python3
"""Post-process tf_tracks.ini: paired colors, clean two-line titles,
shared y per assay, gene-track labels."""
import re
import numpy as np
import pyBigWig

INI = "tf_tracks.ini"
CHROM, START, END, NBINS = "chrX", 30_185_000, 30_285_000, 700

PAIRS = [
    # (section_k562, section_hela, bw_k, bw_h, title_prefix)
    ("K562_CTCF.hg19", "HeLaS3_CTCF.hg19",
     "./K562_CTCF.hg19.bw", "./HeLaS3_CTCF.hg19.bw", "CTCF"),
]
FIXED_YMAX = {"CTCF": 80}
SINGLES = [
    # (section, bw, title_prefix, cell, color)
    ("K562_NETseq.hg19", "./K562_NETseq.hg19.bw", "NET-seq", "K562", "#2166ac"),
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
    if m and m.group(1) in GENE_KEYS:
        out.append(set_keys(block, GENE_KEYS[m.group(1)]))
        continue
    out.append(block)
text = "".join(out)


def polish_section(text, sec, keys):
    pat = re.compile(rf"(?ms)^(\[{re.escape(sec)}\].*?)(?=^\[|\Z)")
    m = pat.search(text)
    if not m:
        print(f"WARN: section {sec} not found")
        return text
    return text[: m.start(1)] + set_keys(m.group(1), keys) + text[m.end(1):]


for sk, sh, bwk, bwh, prefix in PAIRS:
    ymax = FIXED_YMAX.get(prefix) or robust_ymax(bwk, bwh)
    for sec, color, cell in ((sk, C_K, "K562"), (sh, C_H, "HeLaS3")):
        text = polish_section(text, sec, {
            "color": color,
            "title": f"{prefix}\n    {cell}",
            "min_value": "0",
            "max_value": f"{ymax:g}",
        })
    print(f"{prefix}: shared max_value = {ymax:g}")

for sec, bw, prefix, cell, color in SINGLES:
    ymax = robust_ymax(bw, bw)
    text = polish_section(text, sec, {
        "color": color,
        "title": f"{prefix}\n    {cell}",
        "min_value": "0",
        "max_value": f"{ymax:g}",
    })
    print(f"{prefix} {cell}: max_value = {ymax:g}")

open(INI, "w").write(text)
print("INI polished:", INI)
