"""Create PauseNet standard datasets from strand-specific bigWig files."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


BASE_TO_CODE = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3,
}
BED_COLUMNS = [
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
    "sample_type",
]
REVCOMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def require_genomics_dependencies():
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


def first_data_line(path: Path) -> list[str]:
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#"):
                return line.split("\t")
    raise ValueError(f"No data rows found in {path}")


def looks_like_bed_without_header(fields: list[str]) -> bool:
    if len(fields) < 3:
        return False
    try:
        int(fields[1])
        int(fields[2])
        return True
    except ValueError:
        return False


def load_anchor_table(path: str | Path, bed_has_header: bool | None = None) -> pd.DataFrame:
    path = Path(path)
    if bed_has_header is None:
        bed_has_header = not looks_like_bed_without_header(first_data_line(path))

    if bed_has_header:
        anchors = pd.read_csv(path, sep="\t", comment="#")
    else:
        anchors = pd.read_csv(path, sep="\t", comment="#", header=None)
        anchors.columns = BED_COLUMNS[: anchors.shape[1]]

    required = {"chrom", "start", "end", "strand"}
    missing = required.difference(anchors.columns)
    if missing:
        raise ValueError(f"Anchor BED is missing required columns: {sorted(missing)}")

    anchors["start"] = anchors["start"].astype(int)
    anchors["end"] = anchors["end"].astype(int)
    anchors["strand"] = anchors["strand"].astype(str)
    return anchors


def encode_sequence(seq: str) -> np.ndarray:
    return np.fromiter(
        (BASE_TO_CODE.get(base.upper(), 4) for base in seq),
        dtype=np.uint8,
        count=len(seq),
    )


def reverse_complement(seq: str) -> str:
    return seq.translate(REVCOMP)[::-1]


def bigwig_values(bw, chrom: str, start: int, end: int, preserve_signed_signal: bool) -> np.ndarray:
    values = np.asarray(bw.values(chrom, start, end), dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if not preserve_signed_signal:
        values = np.abs(values)
    return values


def assign_split(
    row: pd.Series,
    split_column: str,
    train_chroms: set[str],
    validation_chroms: set[str],
    test_chroms: set[str],
    default_split: str,
) -> str:
    if split_column in row and pd.notna(row[split_column]):
        return str(row[split_column])
    chrom = str(row["chrom"])
    if chrom in train_chroms:
        return "train"
    if chrom in validation_chroms:
        return "validation"
    if chrom in test_chroms:
        return "test"
    return default_split


def sample_type_code(value) -> int:
    if pd.isna(value):
        return 0
    if isinstance(value, (int, np.integer)):
        return int(value)
    text = str(value).strip().lower()
    if text in {"0", "positive", "pos", "peak"}:
        return 0
    if text in {"1", "hard_negative", "hard-negative", "hard"}:
        return 1
    if text in {"2", "zero_negative", "zero-negative", "zero"}:
        return 2
    return 0


def save_split(output_dir: Path, split_name: str, rows: list[dict]) -> None:
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    np.save(split_dir / "sequence_codes.npy", np.stack([row["sequence_codes"] for row in rows]))
    np.save(split_dir / "profiles.npy", np.stack([row["profile"] for row in rows]).astype(np.float32))
    np.save(split_dir / "counts.npy", np.asarray([row["count"] for row in rows], dtype=np.float32))
    np.save(
        split_dir / "profile_loss_mask.npy",
        np.asarray([row["profile_loss_mask"] for row in rows], dtype=np.uint8),
    )
    np.save(
        split_dir / "sample_types.npy",
        np.asarray([row["sample_type"] for row in rows], dtype=np.uint8),
    )

    manifest_rows = []
    for row in rows:
        manifest_rows.append(
            {key: value for key, value in row.items() if key not in {"sequence_codes", "profile"}}
        )
    pd.DataFrame(manifest_rows).to_csv(split_dir / "manifest.tsv", sep="\t", index=False)


def prepare_bigwig_dataset(
    pos_bw_path: str | Path,
    neg_bw_path: str | Path,
    fasta_path: str | Path,
    anchors_bed: str | Path,
    output_dir: str | Path,
    input_length: int = 2114,
    output_length: int = 1000,
    split_column: str = "split",
    train_chroms: Iterable[str] = (),
    validation_chroms: Iterable[str] = (),
    test_chroms: Iterable[str] = (),
    default_split: str = "train",
    profile_min_count: float = 1.0,
    bed_has_header: bool | None = None,
    preserve_signed_signal: bool = False,
    cell_line: str = "",
    assay: str = "",
) -> dict[str, int]:
    require_genomics_dependencies()

    import pyBigWig
    from pyfaidx import Fasta

    if (input_length - output_length) % 2 != 0:
        raise ValueError("input_length - output_length must be even")
    context = (input_length - output_length) // 2

    anchors = load_anchor_table(anchors_bed, bed_has_header=bed_has_header)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    genome = Fasta(str(fasta_path), sequence_always_upper=True)
    pos_bw = pyBigWig.open(str(pos_bw_path))
    neg_bw = pyBigWig.open(str(neg_bw_path))

    splits: dict[str, list[dict]] = defaultdict(list)
    skipped = 0

    for row_index, row in anchors.iterrows():
        chrom = str(row["chrom"])
        strand = str(row["strand"])
        if strand not in {"+", "-"}:
            skipped += 1
            continue
        if chrom not in genome:
            skipped += 1
            continue

        anchor_center = int((int(row["start"]) + int(row["end"])) // 2)
        output_start = anchor_center - output_length // 2
        output_end = output_start + output_length
        input_start = output_start - context
        input_end = input_start + input_length

        if input_start < 0 or input_end > len(genome[chrom]):
            skipped += 1
            continue

        seq = genome[chrom][input_start:input_end].seq
        if strand == "-":
            seq = reverse_complement(seq)
        sequence_codes = encode_sequence(seq)

        bw = pos_bw if strand == "+" else neg_bw
        profile = bigwig_values(
            bw,
            chrom,
            output_start,
            output_end,
            preserve_signed_signal=preserve_signed_signal,
        )
        if len(profile) != output_length:
            skipped += 1
            continue
        if strand == "-":
            profile = profile[::-1].copy()

        count = float(profile.sum())
        split = assign_split(
            row,
            split_column=split_column,
            train_chroms=set(train_chroms),
            validation_chroms=set(validation_chroms),
            test_chroms=set(test_chroms),
            default_split=default_split,
        )

        sample_id = row.get("sample_id", row.get("name", f"{chrom}:{output_start}-{output_end}:{strand}"))
        region_type = row.get("region_type", "region")
        sample_type = sample_type_code(row.get("sample_type", 0))

        splits[split].append(
            {
                "sample_id": sample_id,
                "chrom": chrom,
                "input_start": input_start,
                "input_end": input_end,
                "output_start": output_start,
                "output_end": output_end,
                "anchor_center": anchor_center,
                "strand": strand,
                "gene_id": row.get("gene_id", ""),
                "gene_name": row.get("gene_name", ""),
                "transcript_id": row.get("transcript_id", ""),
                "region_type": region_type,
                "cell_line": cell_line,
                "assay": assay,
                "split": split,
                "sample_type": sample_type,
                "count": count,
                "profile_loss_mask": int(count >= profile_min_count),
                "sequence_codes": sequence_codes,
                "profile": profile.astype(np.float32),
            }
        )

    pos_bw.close()
    neg_bw.close()
    genome.close()

    summary = {"skipped": skipped}
    for split_name, rows in sorted(splits.items()):
        save_split(output_dir, split_name, rows)
        summary[split_name] = len(rows)
    return summary


def add_prepare_bigwig_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pos-bw", required=True, help="Positive-strand bigWig.")
    parser.add_argument("--neg-bw", required=True, help="Negative-strand bigWig.")
    parser.add_argument("--fasta", required=True, help="Reference genome FASTA.")
    parser.add_argument("--anchors-bed", required=True, help="Anchor/window BED or TSV.")
    parser.add_argument("--output-dir", required=True, help="Output PauseNet dataset directory.")
    parser.add_argument("--input-length", type=int, default=2114)
    parser.add_argument("--output-length", type=int, default=1000)
    parser.add_argument("--split-column", default="split")
    parser.add_argument("--train-chroms", default="")
    parser.add_argument("--validation-chroms", default="")
    parser.add_argument("--test-chroms", default="")
    parser.add_argument("--default-split", default="train")
    parser.add_argument("--profile-min-count", type=float, default=1.0)
    parser.add_argument("--bed-has-header", action="store_true")
    parser.add_argument("--bed-no-header", action="store_true")
    parser.add_argument(
        "--preserve-signed-signal",
        action="store_true",
        help="Keep signed bigWig values. By default PauseNet uses absolute read counts.",
    )
    parser.add_argument("--cell-line", default="")
    parser.add_argument("--assay", default="")


def run_prepare_bigwig_from_args(args: argparse.Namespace) -> None:
    if args.bed_has_header and args.bed_no_header:
        raise SystemExit("Use only one of --bed-has-header or --bed-no-header.")
    bed_has_header = None
    if args.bed_has_header:
        bed_has_header = True
    if args.bed_no_header:
        bed_has_header = False

    summary = prepare_bigwig_dataset(
        pos_bw_path=args.pos_bw,
        neg_bw_path=args.neg_bw,
        fasta_path=args.fasta,
        anchors_bed=args.anchors_bed,
        output_dir=args.output_dir,
        input_length=args.input_length,
        output_length=args.output_length,
        split_column=args.split_column,
        train_chroms=parse_chrom_list(args.train_chroms),
        validation_chroms=parse_chrom_list(args.validation_chroms),
        test_chroms=parse_chrom_list(args.test_chroms),
        default_split=args.default_split,
        profile_min_count=args.profile_min_count,
        bed_has_header=bed_has_header,
        preserve_signed_signal=args.preserve_signed_signal,
        cell_line=args.cell_line,
        assay=args.assay,
    )
    print("PauseNet dataset written.")
    for key, value in summary.items():
        print(f"{key}: {value}")

