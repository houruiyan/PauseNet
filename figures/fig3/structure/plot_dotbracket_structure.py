#!/usr/bin/env python3
"""Draw a 2D secondary-structure plot (matplotlib) from a dot-bracket file.

Reads <stem>_dotbracket.txt produced by fold_dotbracket.py, takes the DNA
Mathews 2004 structure (falling back to the RNA one), computes 2D layout
coordinates with ViennaRNA's naview algorithm (RNA.simple_xy_coordinates),
and renders:

- grey backbone bonds between consecutive bases
- red/orange bonds for base pairs (GC vs AU/AT vs GU distinguished)
- base letters on top, colored by base identity

Outputs (this folder): <input-stem>_structure.pdf / .png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import RNA

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "pattern_8_lowest_mfe_top1_dotbracket.txt"

BASE_COLORS = {"A": "#4C9E62", "U": "#C44E52", "T": "#C44E52",
               "G": "#DD8452", "C": "#4C72B0", "N": "#888888"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output-prefix", type=Path, default=None)
    return p.parse_args()


def parse_dotbracket_file(path: Path) -> tuple[str, str, str, float]:
    """Return (header, sequence, structure, mfe) using the DNA line."""
    header, seq, struct, mfe = "", None, None, None
    for line in open(path):
        line = line.rstrip("\n")
        if line.startswith(">"):
            header = line[1:]
        elif line.startswith("sequence"):
            seq = line.split()[1]
        elif line.startswith(("DNA", "RNA")) and seq is not None:
            fields = line.split()
            cand = next((t for t in fields[1:]
                         if t and set(t) <= set(".()[]{}<>")), None)
            # keep the first (DNA) structure found
            if struct is None and cand is not None:
                struct = cand
                mfe = float(fields[-2])
    if seq is None or struct is None:
        raise SystemExit(f"could not parse sequence/structure from {path}")
    return header, seq, struct, mfe


def paired_bases(struct: str) -> list[tuple[int, int]]:
    pairs, stack = [], {}
    opening = {"(": ")", "[": "]", "{": "}", "<": "<"}
    closing = {")": "(", "]": "[", "}": "{", ">": "<"}
    for i, ch in enumerate(struct):
        if ch in opening:
            stack.setdefault(ch, []).append(i)
        elif ch in closing:
            op = closing[ch]
            j = stack[op].pop()
            pairs.append((j, i))
    return pairs


def pair_kind(a: str, b: str) -> str:
    pair = {a.upper(), b.upper()}
    if pair == {"G", "C"}:
        return "GC"
    if pair in ({"A", "U"}, {"A", "T"}):
        return "AU"
    return "GU/other"


def main() -> int:
    args = parse_args()
    output_prefix = args.output_prefix or HERE / f"{args.input.stem}_structure"

    header, seq, struct, mfe = parse_dotbracket_file(args.input)
    n = len(seq)
    assert len(struct) == n, "sequence/structure length mismatch"

    xy = RNA.simple_xy_coordinates(struct)
    if hasattr(xy, "get"):            # older bindings: object with .get(i).X/.Y
        xs = np.array([xy.get(i).X for i in range(n)])
        ys = np.array([xy.get(i).Y for i in range(n)])
    else:                             # newer bindings: tuple of COORDINATE (.X/.Y)
        xs = np.array([xy[i].X for i in range(n)], dtype=float)
        ys = np.array([xy[i].Y for i in range(n)], dtype=float)

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(5, 5))

    # backbone
    for i in range(n - 1):
        ax.plot([xs[i], xs[i + 1]], [ys[i], ys[i + 1]],
                color="#BBBBBB", lw=2.2, zorder=1, solid_capstyle="round")
    # base pairs
    pair_colors = {"GC": "#B2182B", "AU": "#E08214", "GU/other": "#8073AC"}
    for i, j in paired_bases(struct):
        kind = pair_kind(seq[i], seq[j])
        ax.plot([xs[i], xs[j]], [ys[i], ys[j]],
                color=pair_colors[kind], lw=2.2, zorder=2, solid_capstyle="round")
    # base letters
    for i in range(n):
        ax.text(xs[i], ys[i], seq[i], ha="center", va="center",
                fontsize=13, fontweight="bold",
                color=BASE_COLORS.get(seq[i].upper(), "#333333"), zorder=3)

    # annotate 5'/3' ends: offset outward along the direction away from the
    # neighbouring base so the labels never overlap the structure
    for idx, tag in ((0, "5'"), (n - 1, "3'")):
        nbr = 1 if idx == 0 else n - 2
        d = np.array([xs[idx] - xs[nbr], ys[idx] - ys[nbr]], dtype=float)
        d = d / (np.hypot(*d) + 1e-9)
        ax.text(xs[idx] + d[0] * 22, ys[idx] + d[1] * 22, tag,
                ha="center", va="center", fontsize=11, color="#333333", zorder=3)

    pad = 30
    ax.set_xlim(xs.min() - pad - 22, xs.max() + pad + 22)
    ax.set_ylim(ys.min() - pad - 22, ys.max() + pad + 22)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{header}\nMFE = {mfe:.2f} kcal/mol (ViennaRNA, DNA Mathews 2004, 37°C)",
                 fontsize=11)

    fig.tight_layout()
    fig.savefig(f"{output_prefix}.pdf")
    fig.savefig(f"{output_prefix}.png", dpi=300)
    plt.close(fig)
    print(f"structure: {struct}")
    print(f"Wrote: {output_prefix}.pdf")
    print(f"Wrote: {output_prefix}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
