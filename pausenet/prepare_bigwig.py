"""Build fixed gene-structure PauseNet datasets from strand-specific bigWigs.

Each expressed representative transcript contributes five adjacent 1-kb output
windows around TSS, 5'SS, 3'SS, and TES landmarks. The window intervals are
defined in transcription-oriented coordinates before their genomic sequence
and matching strand-specific signal are extracted.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from numpy.lib.format import open_memmap


BASE_TO_CODE = {"A": 0, "C": 1, "G": 2, "T": 3}
REVCOMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")
ANCHOR_TYPE_CODES = {"TSS": 0, "5SS": 1, "3SS": 2, "TES": 3}
GENE_STRUCTURE_WINDOWS = {
    "TSS": [(-1000, 0), (0, 1000), (1000, 2000), (2000, 3000), (3000, 4000)],
    "5SS": [(-2500, -1500), (-1500, -500), (-500, 500), (500, 1500), (1500, 2500)],
    "3SS": [(-2500, -1500), (-1500, -500), (-500, 500), (500, 1500), (1500, 2500)],
    "TES": [(-4000, -3000), (-3000, -2000), (-2000, -1000), (-1000, 0), (0, 1000)],
}


def require_genomics_dependencies() -> None:
    try:
        import pyBigWig  # noqa: F401
        from pyfaidx import Fasta  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "The bigWig preparation command requires optional genomics "
            'dependencies. Install them with: pip install -e ".[genomics]"'
        ) from exc


def parse_chrom_list(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def parse_attributes(text: str) -> dict[str, str]:
    return {
        key: value
        for key, value in re.findall(r'([A-Za-z0-9_]+) "([^"]*)";', text)
    }


def load_chrom_sizes(path: str | Path) -> dict[str, int]:
    sizes: dict[str, int] = {}
    with Path(path).open() as handle:
        for line in handle:
            if line.strip() and not line.startswith("#"):
                chrom, size = line.rstrip().split()[:2]
                sizes[chrom] = int(size)
    if not sizes:
        raise ValueError(f"No chromosome sizes found in {path}")
    return sizes


def load_expressed_genes(path: str | Path | None) -> set[str] | None:
    """Load gene identifiers from a BED-like expressed-gene file, if supplied."""

    if path is None:
        return None
    opener = gzip.open if str(path).endswith(".gz") else open
    expressed: set[str] = set()
    with opener(path, "rt") as handle:
        for line in handle:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            fields = line.rstrip().split("\t")
            for field in fields[3:]:
                if field:
                    expressed.add(field)
                    expressed.add(field.split(".")[0])
    if not expressed:
        raise ValueError(f"No expressed-gene identifiers found in {path}")
    return expressed


def load_transcripts(
    gtf_path: str | Path,
    protein_coding_only: bool,
    min_exons: int,
) -> dict[str, dict]:
    """Read exon coordinates from a GTF/GTF.GZ file in zero-based half-open form."""

    opener = gzip.open if str(gtf_path).endswith(".gz") else open
    transcripts: dict[str, dict] = {}
    with opener(gtf_path, "rt") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip().split("\t")
            if len(fields) < 9 or fields[2] != "exon":
                continue
            chrom, _, _, start, end, _, strand, _, attrs_text = fields
            if strand not in {"+", "-"}:
                continue
            attrs = parse_attributes(attrs_text)
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
    """Select the longest exon-span transcript for each eligible gene."""

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
    """Return a genomic zero-coordinate landmark for one transcript feature."""

    exons = transcript["exons"]
    strand = transcript["strand"]
    if anchor_type == "TSS":
        return exons[0][0] if strand == "+" else exons[-1][1]
    if anchor_type == "TES":
        return exons[-1][1] if strand == "+" else exons[0][0]
    if junction_index is None:
        raise ValueError(f"{anchor_type} requires a splice-junction index")
    left_exon = exons[junction_index]
    right_exon = exons[junction_index + 1]
    if anchor_type == "5SS":
        return left_exon[1] if strand == "+" else right_exon[0]
    if anchor_type == "3SS":
        return right_exon[0] if strand == "+" else left_exon[1]
    raise ValueError(f"Unsupported anchor type: {anchor_type}")


def gene_relative_to_genome(
    anchor: int,
    relative_start: int,
    relative_end: int,
    strand: str,
) -> tuple[int, int]:
    """Map a gene-oriented half-open interval to genomic coordinates."""

    if strand == "+":
        return anchor + relative_start, anchor + relative_end
    return anchor - relative_end, anchor - relative_start


def split_for_chrom(
    chrom: str,
    train_chroms: set[str],
    validation_chroms: set[str],
    test_chroms: set[str],
    default_split: str,
) -> str:
    if chrom in train_chroms:
        return "train"
    if chrom in validation_chroms:
        return "validation"
    if chrom in test_chroms:
        return "test"
    return default_split


def validate_split_chromosomes(
    train_chroms: set[str],
    validation_chroms: set[str],
    test_chroms: set[str],
) -> None:
    groups = {
        "train": train_chroms,
        "validation": validation_chroms,
        "test": test_chroms,
    }
    names = list(groups)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            overlap = groups[first].intersection(groups[second])
            if overlap:
                raise ValueError(
                    f"Chromosomes occur in both {first} and {second}: {sorted(overlap)}"
                )


def junction_indexes(n_junctions: int, max_ss_per_gene: int) -> list[int]:
    if n_junctions <= 0 or max_ss_per_gene == 0:
        return []
    if max_ss_per_gene < 0 or n_junctions <= max_ss_per_gene:
        return list(range(n_junctions))
    return sorted(set(np.linspace(0, n_junctions - 1, max_ss_per_gene, dtype=int).tolist()))


def build_gene_structure_samples(
    transcripts: list[dict],
    chrom_sizes: dict[str, int],
    input_length: int,
    output_length: int,
    max_ss_per_gene: int,
    train_chroms: set[str],
    validation_chroms: set[str],
    test_chroms: set[str],
    default_split: str,
    pilot_genes_per_split: int | None = None,
) -> list[dict]:
    """Expand transcript landmarks into fixed five-window PauseNet examples."""

    if output_length != 1000:
        raise ValueError("The fixed gene-structure window scheme requires --output-length 1000.")
    if input_length <= output_length or (input_length - output_length) % 2:
        raise ValueError("input_length must exceed output_length by an even number.")
    context = (input_length - output_length) // 2
    per_split_genes: Counter[str] = Counter()
    samples: list[dict] = []

    for transcript in transcripts:
        chrom = transcript["chrom"]
        chrom_size = chrom_sizes.get(chrom)
        if chrom_size is None:
            continue
        split = split_for_chrom(
            chrom,
            train_chroms=train_chroms,
            validation_chroms=validation_chroms,
            test_chroms=test_chroms,
            default_split=default_split,
        )
        if pilot_genes_per_split is not None and per_split_genes[split] >= pilot_genes_per_split:
            continue
        per_split_genes[split] += 1

        anchors: list[tuple[str, int | None]] = [("TSS", None), ("TES", None)]
        for junction_index in junction_indexes(len(transcript["exons"]) - 1, max_ss_per_gene):
            anchors.extend([("5SS", junction_index), ("3SS", junction_index)])

        for anchor_type, junction_index in anchors:
            anchor = anchor_position(transcript, anchor_type, junction_index)
            for window_index, (relative_start, relative_end) in enumerate(
                GENE_STRUCTURE_WINDOWS[anchor_type]
            ):
                output_start, output_end = gene_relative_to_genome(
                    anchor,
                    relative_start,
                    relative_end,
                    transcript["strand"],
                )
                input_start = output_start - context
                input_end = output_end + context
                if output_end - output_start != output_length:
                    raise RuntimeError("A gene-structure output window is not 1 kb.")
                if input_start < 0 or input_end > chrom_size:
                    continue
                junction_label = "NA" if junction_index is None else str(junction_index)
                sample_id = (
                    f"{transcript['gene_id_base']}|{transcript['transcript_id']}|"
                    f"{anchor_type}|{junction_label}|w{window_index}"
                )
                samples.append(
                    {
                        "sample_id": sample_id,
                        "gene_id": transcript["gene_id"],
                        "gene_name": transcript["gene_name"],
                        "transcript_id": transcript["transcript_id"],
                        "chrom": chrom,
                        "strand": transcript["strand"],
                        "split": split,
                        "anchor_type": anchor_type,
                        "anchor_index": -1 if junction_index is None else junction_index,
                        "anchor_position": anchor,
                        "window_index": window_index,
                        "relative_start": relative_start,
                        "relative_end": relative_end,
                        "output_start": output_start,
                        "output_end": output_end,
                        "input_start": input_start,
                        "input_end": input_end,
                    }
                )
    return samples


def encode_sequence(sequence: str) -> np.ndarray:
    return np.fromiter(
        (BASE_TO_CODE.get(base.upper(), 4) for base in sequence),
        dtype=np.uint8,
        count=len(sequence),
    )


def reverse_complement(sequence: str) -> str:
    return sequence.translate(REVCOMP)[::-1]


def get_profile(bigwig, chrom: str, start: int, end: int, reverse: bool) -> np.ndarray:
    values = np.asarray(bigwig.values(chrom, start, end), dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.abs(values)
    if reverse:
        values = values[::-1].copy()
    return values


def write_split(
    split: str,
    samples: list[dict],
    output_dir: Path,
    fasta,
    pos_bw,
    neg_bw,
    input_length: int,
    output_length: int,
    profile_min_count: float,
    cell_line: str,
    assay: str,
) -> dict:
    split_samples = [sample for sample in samples if sample["split"] == split]
    if not split_samples:
        raise ValueError(f"No samples assigned to the {split} split.")
    split_samples.sort(
        key=lambda item: (item["chrom"], item["output_start"], item["strand"], item["sample_id"])
    )
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    n_samples = len(split_samples)

    sequence_codes = open_memmap(
        split_dir / "sequence_codes.npy", mode="w+", dtype=np.uint8, shape=(n_samples, input_length)
    )
    profiles = open_memmap(
        split_dir / "profiles.npy", mode="w+", dtype=np.float32, shape=(n_samples, output_length)
    )
    counts = open_memmap(split_dir / "counts.npy", mode="w+", dtype=np.float32, shape=(n_samples,))
    profile_loss_mask = open_memmap(
        split_dir / "profile_loss_mask.npy", mode="w+", dtype=np.uint8, shape=(n_samples,)
    )
    anchor_type_codes = open_memmap(
        split_dir / "anchor_type_codes.npy", mode="w+", dtype=np.uint8, shape=(n_samples,)
    )
    original_strands = open_memmap(
        split_dir / "original_strands.npy", mode="w+", dtype=np.int8, shape=(n_samples,)
    )

    fieldnames = [
        "array_index",
        "sample_id",
        "gene_id",
        "gene_name",
        "transcript_id",
        "chrom",
        "input_start",
        "input_end",
        "output_start",
        "output_end",
        "original_strand",
        "orientation",
        "anchor_type",
        "anchor_index",
        "anchor_position",
        "window_index",
        "relative_start",
        "relative_end",
        "total_count",
        "profile_loss_mask",
        "target_track",
        "cell_line",
        "assay",
        "split",
    ]
    counts_by_anchor: Counter[str] = Counter()
    nonzero_by_anchor: Counter[str] = Counter()
    with (split_dir / "manifest.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for index, sample in enumerate(split_samples):
            reverse = sample["strand"] == "-"
            sequence = fasta[sample["chrom"]][sample["input_start"] : sample["input_end"]].seq
            if len(sequence) != input_length:
                raise RuntimeError(f"Unexpected input sequence length for {sample['sample_id']}")
            if reverse:
                sequence = reverse_complement(sequence)
            profile = get_profile(
                neg_bw if reverse else pos_bw,
                sample["chrom"],
                sample["output_start"],
                sample["output_end"],
                reverse,
            )
            if len(profile) != output_length:
                raise RuntimeError(f"Unexpected profile length for {sample['sample_id']}")
            total_count = float(profile.sum(dtype=np.float64))
            mask = int(total_count >= profile_min_count)

            sequence_codes[index] = encode_sequence(sequence)
            profiles[index] = profile
            counts[index] = total_count
            profile_loss_mask[index] = mask
            anchor_type_codes[index] = ANCHOR_TYPE_CODES[sample["anchor_type"]]
            original_strands[index] = 1 if sample["strand"] == "+" else -1
            counts_by_anchor[sample["anchor_type"]] += 1
            if mask:
                nonzero_by_anchor[sample["anchor_type"]] += 1
            writer.writerow(
                {
                    "array_index": index,
                    "original_strand": sample["strand"],
                    "orientation": "reverse_complement" if reverse else "forward",
                    "total_count": total_count,
                    "profile_loss_mask": mask,
                    "target_track": "neg" if reverse else "pos",
                    "cell_line": cell_line,
                    "assay": assay,
                    **{key: sample[key] for key in sample if key != "strand"},
                }
            )

    for array in (sequence_codes, profiles, counts, profile_loss_mask, anchor_type_codes, original_strands):
        array.flush()
    return {
        "samples": n_samples,
        "anchors": dict(counts_by_anchor),
        "profile_eligible": dict(nonzero_by_anchor),
        "files": {
            "manifest": str(split_dir / "manifest.tsv"),
            "sequence_codes": str(split_dir / "sequence_codes.npy"),
            "profiles": str(split_dir / "profiles.npy"),
            "counts": str(split_dir / "counts.npy"),
            "profile_loss_mask": str(split_dir / "profile_loss_mask.npy"),
            "anchor_type_codes": str(split_dir / "anchor_type_codes.npy"),
            "original_strands": str(split_dir / "original_strands.npy"),
        },
    }


def prepare_bigwig_dataset(
    pos_bw_path: str | Path,
    neg_bw_path: str | Path,
    fasta_path: str | Path,
    gtf_path: str | Path,
    chrom_sizes_path: str | Path,
    output_dir: str | Path,
    expressed_genes_bed: str | Path | None = None,
    input_length: int = 2114,
    output_length: int = 1000,
    max_ss_per_gene: int = 20,
    min_exons: int = 2,
    protein_coding_only: bool = False,
    train_chroms: Iterable[str] = (),
    validation_chroms: Iterable[str] = (),
    test_chroms: Iterable[str] = (),
    default_split: str = "train",
    pilot_genes_per_split: int | None = None,
    profile_min_count: float = 1.0,
    cell_line: str = "",
    assay: str = "",
) -> dict:
    """Write train/validation/test arrays for the fixed five-window workflow."""

    require_genomics_dependencies()
    from pyfaidx import Fasta
    import pyBigWig

    train_set = set(train_chroms)
    validation_set = set(validation_chroms)
    test_set = set(test_chroms)
    validate_split_chromosomes(train_set, validation_set, test_set)
    chrom_sizes = load_chrom_sizes(chrom_sizes_path)
    transcripts = load_transcripts(gtf_path, protein_coding_only, min_exons)
    representatives = choose_representative_transcripts(
        transcripts,
        load_expressed_genes(expressed_genes_bed),
    )
    samples = build_gene_structure_samples(
        representatives,
        chrom_sizes=chrom_sizes,
        input_length=input_length,
        output_length=output_length,
        max_ss_per_gene=max_ss_per_gene,
        train_chroms=train_set,
        validation_chroms=validation_set,
        test_chroms=test_set,
        default_split=default_split,
        pilot_genes_per_split=pilot_genes_per_split,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fasta = Fasta(str(fasta_path), sequence_always_upper=True)
    pos_bw = pyBigWig.open(str(pos_bw_path))
    neg_bw = pyBigWig.open(str(neg_bw_path))
    try:
        summary = {
            "input_length": input_length,
            "output_length": output_length,
            "window_scheme": GENE_STRUCTURE_WINDOWS,
            "protein_coding_only": protein_coding_only,
            "min_exons": min_exons,
            "max_ss_per_gene": max_ss_per_gene,
            "n_transcripts_loaded": len(transcripts),
            "n_representative_transcripts": len(representatives),
            "n_samples_total": len(samples),
            "splits": {},
        }
        for split in ("train", "validation", "test"):
            summary["splits"][split] = write_split(
                split,
                samples=samples,
                output_dir=output_dir,
                fasta=fasta,
                pos_bw=pos_bw,
                neg_bw=neg_bw,
                input_length=input_length,
                output_length=output_length,
                profile_min_count=profile_min_count,
                cell_line=cell_line,
                assay=assay,
            )
    finally:
        pos_bw.close()
        neg_bw.close()
        fasta.close()

    with (output_dir / "dataset_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def add_prepare_bigwig_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pos-bw", required=True, help="Positive-strand bigWig.")
    parser.add_argument("--neg-bw", required=True, help="Negative-strand bigWig.")
    parser.add_argument("--fasta", required=True, help="Reference genome FASTA.")
    parser.add_argument("--gtf", required=True, help="Gene annotation GTF or GTF.GZ.")
    parser.add_argument("--chrom-sizes", required=True, help="Chromosome sizes matching the FASTA.")
    parser.add_argument("--expressed-genes-bed", help="Optional BED-like file of expressed genes.")
    parser.add_argument("--output-dir", required=True, help="Output PauseNet dataset directory.")
    parser.add_argument("--input-length", type=int, default=2114)
    parser.add_argument("--output-length", type=int, default=1000)
    parser.add_argument("--max-ss-per-gene", type=int, default=20)
    parser.add_argument("--min-exons", type=int, default=2)
    parser.add_argument("--protein-coding-only", action="store_true")
    parser.add_argument("--train-chroms", default="")
    parser.add_argument("--validation-chroms", default="")
    parser.add_argument("--test-chroms", default="")
    parser.add_argument("--default-split", default="train")
    parser.add_argument("--pilot-genes-per-split", type=int)
    parser.add_argument("--profile-min-count", type=float, default=1.0)
    parser.add_argument("--cell-line", default="")
    parser.add_argument("--assay", default="")


def run_prepare_bigwig_from_args(args: argparse.Namespace) -> None:
    summary = prepare_bigwig_dataset(
        pos_bw_path=args.pos_bw,
        neg_bw_path=args.neg_bw,
        fasta_path=args.fasta,
        gtf_path=args.gtf,
        chrom_sizes_path=args.chrom_sizes,
        expressed_genes_bed=args.expressed_genes_bed,
        output_dir=args.output_dir,
        input_length=args.input_length,
        output_length=args.output_length,
        max_ss_per_gene=args.max_ss_per_gene,
        min_exons=args.min_exons,
        protein_coding_only=args.protein_coding_only,
        train_chroms=parse_chrom_list(args.train_chroms),
        validation_chroms=parse_chrom_list(args.validation_chroms),
        test_chroms=parse_chrom_list(args.test_chroms),
        default_split=args.default_split,
        pilot_genes_per_split=args.pilot_genes_per_split,
        profile_min_count=args.profile_min_count,
        cell_line=args.cell_line,
        assay=args.assay,
    )
    print("PauseNet gene-structure dataset written.")
    print(json.dumps(summary, indent=2))
