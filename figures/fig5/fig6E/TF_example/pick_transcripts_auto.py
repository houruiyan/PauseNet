#!/usr/bin/env python3
"""Pick the most representative transcript per gene.
Usage: pick_transcripts_auto.py <candidate_transcript_lines.txt> <gene_names.txt>
"""
import re
import sys

GENES = [l.strip() for l in open(sys.argv[2]) if l.strip()]

def attrs(s):
    return dict(re.findall(r'(\S+) "([^"]+)"', s))

best = {}
for line in open(sys.argv[1]):
    f = line.rstrip("\n").split("\t")
    if len(f) < 9 or f[2] != "transcript":
        continue
    a = attrs(f[8])
    g = a.get("gene_name")
    if g not in GENES:
        continue
    tags = f[8]
    if 'MANE_Select' in tags:
        prio = 0
    elif 'Ensembl_canonical' in tags:
        prio = 1
    elif 'appris_principal' in tags:
        prio = 2
    elif 'basic' in tags:
        prio = 3
    else:
        prio = 4
    length = int(f[4]) - int(f[3])
    key = (prio, -length)
    if g not in best or key < best[g][0]:
        best[g] = (key, a["transcript_id"])

for g in GENES:
    if g not in best:
        print(f"WARN: no transcript found for {g}", file=sys.stderr)
        continue
    tid = best[g][1]
    base = tid.split(".")[0]
    print(f"{g}\t{tid}\t{base}")
