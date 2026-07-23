#!/usr/bin/env python3
"""Build PauseNet gene-structure anchor windows directly from a GTF file.

The output schema matches ``dataset_split_hg19.tsv``.  Coordinates are
zero-based, half-open.  One representative transcript is selected per gene,
then five 1-kb windows are generated around every retained TSS, 5'SS, 3'SS,
and TES anchor.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, TextIO


WINDOWS = {
    "TSS": [(-1000, 0), (0, 1000), (1000, 2000), (2000, 3000), (3000, 4000)],
    "5SS": [
        (-2500, -1500),
        (-1500, -500),
        (-500, 500),
        (500, 1500),
        (1500, 2500),
    ],
    "3SS": [
        (-2500, -1500),
        (-1500, -500),
        (-500, 500),
        (500, 1500),
        (1500, 2500),
    ],
    "TES": [
        (-4000, -3000),
        (-3000, -2000),
        (-2000, -1000),
        (-1000, 0),
        (0, 1000),
    ],
}

OUTPUT_COLUMNS = [
    "chrom",
    "start",
    "end",
    "sample_id",
    "score",
    "strand",
    "region_type",
    "gene_id",
    "gene_name",
    "transcript_id",
    "split",
    "anchor_type",
    "anchor_index",
    "anchor_position",
    "window_index",
    "relative_start",
    "relative_end",
    "source_array_index",
]

ATTRIBUTE_RE = re.compile(r'([A-Za-z0-9_]+) "([^"]*)";')


def open_text(path: Path) -> TextIO:
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return path.open()


def parse_attributes(text: str) -> dict[str, str]:
    return {key: value for key, value in ATTRIBUTE_RE.findall(text)}


def load_chrom_sizes(path: Path) -> dict[str, int]:
    sizes: dict[str, int] = {}
    with path.open() as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            chrom, size = line.split()[:2]
            sizes[chrom] = int(size)
    if not sizes:
        raise ValueError(f"No chromosome sizes found in {path}")
    return sizes


def load_expressed_genes(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    identifiers: set[str] = set()
    with open_text(path) as handle:
        for line in handle:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            for value in line.rstrip().split("\t")[3:]:
                if value:
                    identifiers.add(value)
                    identifiers.add(value.split(".")[0])
    if not identifiers:
        raise ValueError(f"No gene identifiers found in {path}")
    return identifiers


def load_transcripts(
    gtf_path: Path,
    protein_coding_only: bool,
    min_exons: int,
) -> dict[str, dict]:
    """Load GTF exons as zero-based, half-open genomic intervals."""
    transcripts: dict[str, dict] = {}
    with open_text(gtf_path) as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip().split("\t")
            if len(fields) < 9 or fields[2] != "exon":
                continue

            chrom, start, end, strand = fields[0], fields[3], fields[4], fields[6]
            if strand not in {"+", "-"}:
                continue
            attrs = parse_attributes(fields[8])
            transcript_id = attrs.get("transcript_id")
            gene_id = attrs.get("gene_id")
            if not transcript_id or not gene_id:
                continue

            transcript_type = (
                attrs.get("transcript_type")
                or attrs.get("transcript_biotype")
                or attrs.get("gene_type")
                or attrs.get("gene_biotype")
                or ""
            )
            if protein_coding_only and transcript_type != "protein_coding":
                continue

            transcript = transcripts.setdefault(
                transcript_id,
                {
                    "transcript_id": transcript_id,
                    "gene_id": gene_id,
                    "gene_id_base": gene_id.split(".")[0],
                    "gene_name": attrs.get("gene_name", gene_id),
                    "chrom": chrom,
                    "strand": strand,
                    "exons": [],
                },
            )
            # GTF is 1-based inclusive; convert to 0-based half-open.
            transcript["exons"].append((int(start) - 1, int(end)))

    selected: dict[str, dict] = {}
    for transcript_id, transcript in transcripts.items():
        transcript["exons"] = sorted(set(transcript["exons"]))
        if len(transcript["exons"]) >= min_exons:
            selected[transcript_id] = transcript
    return selected


def choose_representative_transcripts(
    transcripts: dict[str, dict],
    expressed_genes: set[str] | None,
) -> list[dict]:
    """Select one transcript per gene by exon bp, span, then exon count."""
    by_gene: dict[str, list[dict]] = defaultdict(list)
    for transcript in transcripts.values():
        identifiers = {
            transcript["gene_id"],
            transcript["gene_id_base"],
            transcript["gene_name"],
        }
        if expressed_genes is not None and identifiers.isdisjoint(expressed_genes):
            continue
        by_gene[transcript["gene_id_base"]].append(transcript)

    representatives: list[dict] = []
    for candidates in by_gene.values():
        def transcript_key(item: dict) -> tuple[int, int, int]:
            exon_bp = sum(end - start for start, end in item["exons"])
            span = item["exons"][-1][1] - item["exons"][0][0]
            return exon_bp, span, len(item["exons"])

        representatives.append(max(candidates, key=transcript_key))

    return sorted(
        representatives,
        key=lambda item: (item["chrom"], item["gene_id"], item["transcript_id"]),
    )


def anchor_position(
    transcript: dict,
    anchor_type: str,
    junction_index: int | None = None,
) -> int:
    """Return a zero-based genomic boundary coordinate for one anchor."""
    exons = transcript["exons"]
    strand = transcript["strand"]
    if anchor_type == "TSS":
        return exons[0][0] if strand == "+" else exons[-1][1]
    if anchor_type == "TES":
        return exons[-1][1] if strand == "+" else exons[0][0]
    if junction_index is None:
        raise ValueError(f"{anchor_type} requires a junction index")

    left_exon = exons[junction_index]
    right_exon = exons[junction_index + 1]
    if anchor_type == "5SS":
        return left_exon[1] if strand == "+" else right_exon[0]
    if anchor_type == "3SS":
        return right_exon[0] if strand == "+" else left_exon[1]
    raise ValueError(f"Unsupported anchor type: {anchor_type}")


def junction_indexes(n_junctions: int, maximum: int) -> list[int]:
    """Match np.linspace(..., dtype=int) selection without requiring NumPy."""
    if n_junctions <= 0 or maximum == 0:
        return []
    if maximum < 0 or n_junctions <= maximum:
        return list(range(n_junctions))
    if maximum == 1:
        return [0]
    return sorted(
        {(index * (n_junctions - 1)) // (maximum - 1) for index in range(maximum)}
    )


def genomic_window(
    anchor: int,
    relative_start: int,
    relative_end: int,
    strand: str,
) -> tuple[int, int]:
    if strand == "+":
        return anchor + relative_start, anchor + relative_end
    return anchor - relative_end, anchor - relative_start


def parse_chroms(value: str) -> set[str]:
    return {chrom.strip() for chrom in value.split(",") if chrom.strip()}


def split_for_chrom(
    chrom: str,
    validation_chroms: set[str],
    test_chroms: set[str],
) -> str:
    if chrom in validation_chroms:
        return "validation"
    if chrom in test_chroms:
        return "test"
    return "train"


def build_rows(
    representatives: Iterable[dict],
    chrom_sizes: dict[str, int],
    input_length: int,
    output_length: int,
    max_ss_per_gene: int,
    validation_chroms: set[str],
    test_chroms: set[str],
) -> dict[str, list[dict]]:
    if output_length != 1000:
        raise ValueError("The fixed five-window scheme requires output length 1000")
    if input_length <= output_length or (input_length - output_length) % 2:
        raise ValueError("input length must exceed output length by an even number")
    context = (input_length - output_length) // 2
    rows: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}

    for transcript in representatives:
        chrom = transcript["chrom"]
        if chrom not in chrom_sizes:
            continue
        split = split_for_chrom(chrom, validation_chroms, test_chroms)
        anchors: list[tuple[str, int | None]] = [("TSS", None), ("TES", None)]
        for junction_index in junction_indexes(
            len(transcript["exons"]) - 1,
            max_ss_per_gene,
        ):
            anchors.extend([("5SS", junction_index), ("3SS", junction_index)])

        for anchor_type, junction_index in anchors:
            anchor = anchor_position(transcript, anchor_type, junction_index)
            anchor_index = -1 if junction_index is None else junction_index
            junction_label = "NA" if junction_index is None else str(junction_index)
            for window_index, (relative_start, relative_end) in enumerate(
                WINDOWS[anchor_type]
            ):
                start, end = genomic_window(
                    anchor,
                    relative_start,
                    relative_end,
                    transcript["strand"],
                )
                input_start = start - context
                input_end = end + context
                if input_start < 0 or input_end > chrom_sizes[chrom]:
                    continue
                sample_id = (
                    f"{transcript['gene_id_base']}|{transcript['transcript_id']}|"
                    f"{anchor_type}|{junction_label}|w{window_index}"
                )
                rows[split].append(
                    {
                        "chrom": chrom,
                        "start": start,
                        "end": end,
                        "sample_id": sample_id,
                        "score": 0,
                        "strand": transcript["strand"],
                        "region_type": anchor_type,
                        "gene_id": transcript["gene_id"],
                        "gene_name": transcript["gene_name"],
                        "transcript_id": transcript["transcript_id"],
                        "split": split,
                        "anchor_type": anchor_type,
                        "anchor_index": anchor_index,
                        "anchor_position": anchor,
                        "window_index": window_index,
                        "relative_start": relative_start,
                        "relative_end": relative_end,
                    }
                )

    return rows


def write_rows(rows: dict[str, list[dict]], output_path: Path) -> dict[str, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, delimiter="\t")
        writer.writeheader()
        for split in ("train", "validation", "test"):
            split_rows = rows[split]
            split_rows.sort(
                key=lambda row: (
                    row["chrom"],
                    row["start"],
                    row["strand"],
                    row["sample_id"],
                )
            )
            for array_index, row in enumerate(split_rows):
                writer.writerow({**row, "source_array_index": array_index})
            counts[split] = len(split_rows)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gtf", required=True, type=Path)
    parser.add_argument("--chrom-sizes", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expressed-genes-bed", type=Path)
    parser.add_argument("--min-exons", type=int, default=2)
    parser.add_argument("--max-ss-per-gene", type=int, default=20)
    parser.add_argument("--input-length", type=int, default=2114)
    parser.add_argument("--output-length", type=int, default=1000)
    parser.add_argument("--validation-chroms", default="chr10")
    parser.add_argument("--test-chroms", default="chr8,chr9")
    parser.add_argument(
        "--all-biotypes",
        action="store_true",
        help="Include all transcript biotypes; default is protein-coding only.",
    )
    args = parser.parse_args()

    validation_chroms = parse_chroms(args.validation_chroms)
    test_chroms = parse_chroms(args.test_chroms)
    overlap = validation_chroms.intersection(test_chroms)
    if overlap:
        parser.error(f"Chromosomes occur in both validation and test: {sorted(overlap)}")

    transcripts = load_transcripts(
        args.gtf,
        protein_coding_only=not args.all_biotypes,
        min_exons=args.min_exons,
    )
    representatives = choose_representative_transcripts(
        transcripts,
        load_expressed_genes(args.expressed_genes_bed),
    )
    rows = build_rows(
        representatives,
        chrom_sizes=load_chrom_sizes(args.chrom_sizes),
        input_length=args.input_length,
        output_length=args.output_length,
        max_ss_per_gene=args.max_ss_per_gene,
        validation_chroms=validation_chroms,
        test_chroms=test_chroms,
    )
    counts = write_rows(rows, args.output)
    total = sum(counts.values())
    print(f"Loaded transcripts: {len(transcripts):,}")
    print(f"Representative transcripts: {len(representatives):,}")
    for split in ("train", "validation", "test"):
        print(f"{split}: {counts[split]:,}")
    print(f"total: {total:,}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
