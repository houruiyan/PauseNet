#!/usr/bin/env python3
"""Extract the lowest-MFE sequence(s) from the pattern_8 ranked TSV.

Reads pattern_8_lowest_mfe_seqlets.tsv (produced by
../knock_out/find_pattern8_lowest_mfe_seqlets.py), takes the top --top-n
seqlets ranked by minimum MFE (the file is already sorted ascending),
and writes their min-MFE 30-nt window sequences to a FASTA file in this
folder, together with a small info file recording their metadata.

Outputs (this folder):
  pattern_8_lowest_mfe_top{n}.fasta - min-MFE window sequences
  pattern_8_lowest_mfe_top{n}.tsv   - metadata rows of the extracted seqlets
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_TSV = HERE.parent / "knock_out" / "pattern_8_lowest_mfe_seqlets.tsv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tsv", type=Path, default=DEFAULT_TSV)
    p.add_argument("--top-n", type=int, default=1,
                   help="how many top-ranked (lowest-MFE) seqlets to extract")
    p.add_argument("--output-prefix", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    output_prefix = args.output_prefix or HERE / f"pattern_8_lowest_mfe_top{args.top_n}"

    with open(args.tsv) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        raise SystemExit(f"no rows found in {args.tsv}")

    # file is already ranked, but re-sort defensively by min MFE ascending
    rows.sort(key=lambda r: float(r["min_mfe_kcal_mol"]))
    top = rows[:args.top_n]

    fa_path = Path(f"{output_prefix}.fasta")
    with open(fa_path, "w") as f:
        for r in top:
            f.write(f">seqlet{r['seqlet_index']}_rank{r['rank']}"
                    f"_MFE{r['min_mfe_kcal_mol']}kcalmol"
                    f"_pos{int(r['min_mfe_pos_rel_center_bp']):+d}bp\n")
            f.write(r["min_mfe_window_seq"] + "\n")
    print(f"Wrote: {fa_path}", flush=True)

    tsv_path = Path(f"{output_prefix}.tsv")
    header = list(top[0].keys())
    with open(tsv_path, "w") as f:
        f.write("\t".join(header) + "\n")
        for r in top:
            f.write("\t".join(r[k] for k in header) + "\n")
    print(f"Wrote: {tsv_path}", flush=True)

    for r in top:
        print(f"rank {r['rank']}: seqlet #{r['seqlet_index']} "
              f"MFE={r['min_mfe_kcal_mol']} kcal/mol "
              f"at {int(r['min_mfe_pos_rel_center_bp']):+d} bp "
              f"seq={r['min_mfe_window_seq']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
