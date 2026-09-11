#!/usr/bin/env python3
"""Fold the extracted lowest-MFE window sequence(s) with ViennaRNA and report
dot-bracket structures.

Reads pattern_8_lowest_mfe_top{n}.fasta (produced by
extract_lowest_mfe_sequence.py) and folds every sequence twice:

1. DNA Mathews 2004 parameters - same energy model used for the MFE track
   and the seqlet ranking, so the resulting MFE should match the TSV value
   for the 30-nt window.
2. Default RNA parameters (DNA alphabet mapped T->U) - the relevant model
   if the structure acts on the nascent RNA strand.

Outputs (this folder):
  <input-stem>_dotbracket.txt - one block per sequence with both structures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import RNA

HERE = Path(__file__).resolve().parent
DEFAULT_FASTA = HERE / "pattern_8_lowest_mfe_top1.fasta"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fasta", type=Path, default=DEFAULT_FASTA)
    p.add_argument("--temperature", type=float, default=37.0)
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def read_fasta(path: Path) -> list[tuple[str, str]]:
    entries = []
    header, seq = None, []
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            if header is not None:
                entries.append((header, "".join(seq)))
            header, seq = line[1:], []
        elif line:
            seq.append(line)
    if header is not None:
        entries.append((header, "".join(seq)))
    return entries


def fold_dna(sequence: str, temperature: float) -> tuple[str, float]:
    RNA.params_load_DNA_Mathews2004()
    md = RNA.md()
    md.temperature = temperature
    md.dangles = 2
    fc = RNA.fold_compound(sequence, md)
    structure, mfe = fc.mfe()
    return structure, float(mfe)


def fold_rna(sequence: str, temperature: float) -> tuple[str, float]:
    # RNA.md() defaults to the standard RNA (Turner 2004) energy model
    md = RNA.md()
    md.temperature = temperature
    md.dangles = 2
    fc = RNA.fold_compound(sequence.replace("T", "U"), md)
    structure, mfe = fc.mfe()
    return structure, float(mfe)


def main() -> int:
    args = parse_args()
    out_path = args.output or HERE / f"{args.fasta.stem}_dotbracket.txt"

    entries = read_fasta(args.fasta)
    if not entries:
        raise SystemExit(f"no sequences found in {args.fasta}")

    blocks = []
    for header, seq in entries:
        dna_struct, dna_mfe = fold_dna(seq, args.temperature)
        rna_struct, rna_mfe = fold_rna(seq, args.temperature)
        block = (
            f">{header}\n"
            f"sequence      {seq}  ({len(seq)} nt)\n"
            f"DNA Mathews04 {dna_struct}  MFE = {dna_mfe:.2f} kcal/mol\n"
            f"RNA Turner04  {rna_struct}  MFE = {rna_mfe:.2f} kcal/mol\n"
        )
        blocks.append(block)
        print(block, flush=True)

    with open(out_path, "w") as f:
        f.write("\n".join(blocks))
    print(f"Wrote: {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
