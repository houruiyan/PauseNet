"""Create PauseNet standard datasets from strand-specific bigWig files."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap


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
]
MINIMAL_BED_COLUMNS = ["chrom", "start", "end", "strand"]
REVCOMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")
MANIFEST_COLUMNS = [
    "array_index",
    "sample_id",
    "chrom",
    "input_start",
    "input_end",
    "output_start",
    "output_end",
    "anchor_center",
    "strand",
    "gene_id",
    "gene_name",
    "transcript_id",
    "region_type",
    "cell_line",
    "assay",
    "split",
    "count",
    "profile_loss_mask",
]
ANCHOR_CORE_COLUMNS = {
    "chrom",
    "start",
    "end",
    "name",
    "sample_id",
    "score",
    "strand",
    "region_type",
    "gene_id",
    "gene_name",
    "transcript_id",
    "split",
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


def _open_text(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else path.open()


def first_data_line(path: Path) -> list[str]:
    with _open_text(path) as handle:
        for line in handle:
            line = line.rstrip("\n\r")
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


def anchor_schema(path: str | Path, bed_has_header: bool | None = None) -> tuple[list[str], bool]:
    path = Path(path)
    fields = first_data_line(path)
    if bed_has_header is None:
        bed_has_header = not looks_like_bed_without_header(fields)

    if bed_has_header:
        columns = fields
    elif len(fields) == 4:
        columns = MINIMAL_BED_COLUMNS
    elif len(fields) >= 6:
        if len(fields) > len(BED_COLUMNS):
            raise ValueError(
                "Headerless anchors may contain at most 11 BED-like columns; "
                "use a header for additional metadata."
            )
        columns = BED_COLUMNS[: len(fields)]
    else:
        raise ValueError(
            "Headerless anchors must have either four columns "
            "(chrom, start, end, strand) or a BED6-like layout."
        )

    required = {"chrom", "start", "end", "strand"}
    missing = required.difference(columns)
    if missing:
        raise ValueError(f"Anchor BED is missing required columns: {sorted(missing)}")
    return columns, bool(bed_has_header)


def iter_anchor_rows(
    path: str | Path,
    bed_has_header: bool | None = None,
) -> tuple[list[str], Iterator[dict[str, str]]]:
    path = Path(path)
    columns, has_header = anchor_schema(path, bed_has_header=bed_has_header)

    def rows() -> Iterator[dict[str, str]]:
        saw_header = False
        with _open_text(path) as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.rstrip("\n\r")
                if not line or line.startswith("#"):
                    continue
                if has_header and not saw_header:
                    saw_header = True
                    continue
                fields = line.split("\t")
                if len(fields) != len(columns):
                    raise ValueError(
                        f"Anchor row {line_number} has {len(fields)} fields; "
                        f"expected {len(columns)}."
                    )
                yield dict(zip(columns, fields, strict=True))

    return columns, rows()


def load_anchor_table(path: str | Path, bed_has_header: bool | None = None):
    """Load anchors into a pandas DataFrame for small interactive use cases."""
    import pandas as pd

    columns, rows = iter_anchor_rows(path, bed_has_header=bed_has_header)
    anchors = pd.DataFrame.from_records(rows, columns=columns)
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
    row: Mapping[str, str],
    split_column: str,
    train_chroms: set[str],
    validation_chroms: set[str],
    test_chroms: set[str],
    default_split: str,
) -> str:
    explicit_split = row.get(split_column, "").strip()
    if explicit_split:
        return explicit_split
    chrom = row["chrom"]
    if chrom in train_chroms:
        return "train"
    if chrom in validation_chroms:
        return "validation"
    if chrom in test_chroms:
        return "test"
    return default_split


def anchor_geometry(
    row: Mapping[str, str],
    chrom_sizes: Mapping[str, int],
    input_length: int,
    output_length: int,
) -> dict[str, int | str] | None:
    chrom = row.get("chrom", "")
    strand = row.get("strand", "")
    if strand not in {"+", "-"} or chrom not in chrom_sizes:
        return None
    try:
        anchor_center = (int(row["start"]) + int(row["end"])) // 2
    except (KeyError, TypeError, ValueError):
        return None

    context = (input_length - output_length) // 2
    output_start = anchor_center - output_length // 2
    output_end = output_start + output_length
    input_start = output_start - context
    input_end = input_start + input_length
    if input_start < 0 or input_end > chrom_sizes[chrom]:
        return None
    return {
        "chrom": chrom,
        "strand": strand,
        "anchor_center": anchor_center,
        "output_start": output_start,
        "output_end": output_end,
        "input_start": input_start,
        "input_end": input_end,
    }


def _sample_id(row: Mapping[str, str], geometry: Mapping[str, int | str]) -> str:
    sample_id = row.get("sample_id", "") or row.get("name", "")
    if sample_id:
        return sample_id
    return (
        f"{geometry['chrom']}:{geometry['output_start']}-"
        f"{geometry['output_end']}:{geometry['strand']}"
    )


def _extra_anchor_columns(columns: Iterable[str]) -> list[str]:
    return [
        column
        for column in columns
        if column not in ANCHOR_CORE_COLUMNS and column not in MANIFEST_COLUMNS
    ]


def _manifest_record(
    row: Mapping[str, str],
    geometry: Mapping[str, int | str],
    index: int,
    split: str,
    count: float,
    profile_min_count: float,
    cell_line: str,
    assay: str,
    extra_columns: Iterable[str],
) -> dict[str, str | int | float]:
    record: dict[str, str | int | float] = {
        "array_index": index,
        "sample_id": _sample_id(row, geometry),
        "chrom": geometry["chrom"],
        "input_start": geometry["input_start"],
        "input_end": geometry["input_end"],
        "output_start": geometry["output_start"],
        "output_end": geometry["output_end"],
        "anchor_center": geometry["anchor_center"],
        "strand": geometry["strand"],
        "gene_id": row.get("gene_id", ""),
        "gene_name": row.get("gene_name", ""),
        "transcript_id": row.get("transcript_id", ""),
        "region_type": row.get("region_type", "region") or "region",
        "cell_line": cell_line,
        "assay": assay,
        "split": split,
        "count": count,
        "profile_loss_mask": int(count >= profile_min_count),
    }
    for column in extra_columns:
        record[column] = row.get(column, "")
    return record


def _count_valid_anchors(
    anchors_bed: str | Path,
    bed_has_header: bool | None,
    chrom_sizes: Mapping[str, int],
    input_length: int,
    output_length: int,
    split_column: str,
    train_chroms: set[str],
    validation_chroms: set[str],
    test_chroms: set[str],
    default_split: str,
) -> tuple[dict[str, int], int, list[str]]:
    columns, rows = iter_anchor_rows(anchors_bed, bed_has_header=bed_has_header)
    counts: dict[str, int] = {}
    skipped = 0
    for row in rows:
        if anchor_geometry(row, chrom_sizes, input_length, output_length) is None:
            skipped += 1
            continue
        split = assign_split(
            row,
            split_column=split_column,
            train_chroms=train_chroms,
            validation_chroms=validation_chroms,
            test_chroms=test_chroms,
            default_split=default_split,
        )
        counts[split] = counts.get(split, 0) + 1
    return counts, skipped, _extra_anchor_columns(columns)


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
    """Build a dataset with a two-pass, memory-bounded anchor scan."""
    require_genomics_dependencies()

    import pyBigWig
    from pyfaidx import Fasta

    if input_length < output_length or (input_length - output_length) % 2:
        raise ValueError("input_length must exceed output_length by an even number")

    train_chroms = set(train_chroms)
    validation_chroms = set(validation_chroms)
    test_chroms = set(test_chroms)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    genome = Fasta(str(fasta_path), sequence_always_upper=True)
    chrom_sizes = {chrom: len(genome[chrom]) for chrom in genome.keys()}
    split_counts, skipped, extra_columns = _count_valid_anchors(
        anchors_bed=anchors_bed,
        bed_has_header=bed_has_header,
        chrom_sizes=chrom_sizes,
        input_length=input_length,
        output_length=output_length,
        split_column=split_column,
        train_chroms=train_chroms,
        validation_chroms=validation_chroms,
        test_chroms=test_chroms,
        default_split=default_split,
    )
    if not split_counts:
        genome.close()
        raise ValueError("No valid anchors remain after coordinate validation")

    fieldnames = [*MANIFEST_COLUMNS, *extra_columns]
    writers: dict[str, dict] = {}
    for split, size in split_counts.items():
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        manifest_handle = (split_dir / "manifest.tsv").open("w", newline="")
        manifest_writer = csv.DictWriter(manifest_handle, fieldnames=fieldnames, delimiter="\t")
        manifest_writer.writeheader()
        writers[split] = {
            "sequence_codes": open_memmap(
                split_dir / "sequence_codes.npy", mode="w+", dtype=np.uint8, shape=(size, input_length)
            ),
            "profiles": open_memmap(
                split_dir / "profiles.npy", mode="w+", dtype=np.float32, shape=(size, output_length)
            ),
            "counts": open_memmap(split_dir / "counts.npy", mode="w+", dtype=np.float32, shape=(size,)),
            "profile_loss_mask": open_memmap(
                split_dir / "profile_loss_mask.npy", mode="w+", dtype=np.uint8, shape=(size,)
            ),
            "manifest_handle": manifest_handle,
            "manifest_writer": manifest_writer,
            "index": 0,
        }

    pos_bw = pyBigWig.open(str(pos_bw_path))
    neg_bw = pyBigWig.open(str(neg_bw_path))
    try:
        _, rows = iter_anchor_rows(anchors_bed, bed_has_header=bed_has_header)
        for row in rows:
            geometry = anchor_geometry(row, chrom_sizes, input_length, output_length)
            if geometry is None:
                continue
            split = assign_split(
                row,
                split_column=split_column,
                train_chroms=train_chroms,
                validation_chroms=validation_chroms,
                test_chroms=test_chroms,
                default_split=default_split,
            )
            writer = writers[split]
            index = writer["index"]
            seq = genome[str(geometry["chrom"])][
                int(geometry["input_start"]):int(geometry["input_end"])
            ].seq
            if geometry["strand"] == "-":
                seq = reverse_complement(seq)
            sequence_codes = encode_sequence(seq)
            if len(sequence_codes) != input_length:
                raise RuntimeError(f"Unexpected sequence length at {_sample_id(row, geometry)}")

            bw = pos_bw if geometry["strand"] == "+" else neg_bw
            profile = bigwig_values(
                bw,
                str(geometry["chrom"]),
                int(geometry["output_start"]),
                int(geometry["output_end"]),
                preserve_signed_signal=preserve_signed_signal,
            )
            if len(profile) != output_length:
                raise RuntimeError(f"Unexpected profile length at {_sample_id(row, geometry)}")
            if geometry["strand"] == "-":
                profile = profile[::-1].copy()
            count = float(profile.sum())

            writer["sequence_codes"][index] = sequence_codes
            writer["profiles"][index] = profile
            writer["counts"][index] = count
            writer["profile_loss_mask"][index] = int(count >= profile_min_count)
            writer["manifest_writer"].writerow(
                _manifest_record(
                    row=row,
                    geometry=geometry,
                    index=index,
                    split=split,
                    count=count,
                    profile_min_count=profile_min_count,
                    cell_line=cell_line,
                    assay=assay,
                    extra_columns=extra_columns,
                )
            )
            writer["index"] += 1
    finally:
        pos_bw.close()
        neg_bw.close()
        genome.close()
        for writer in writers.values():
            writer["manifest_handle"].close()
            for key in ("sequence_codes", "profiles", "counts", "profile_loss_mask"):
                writer[key].flush()

    for split, writer in writers.items():
        if writer["index"] != split_counts[split]:
            raise RuntimeError(
                f"Anchor count changed between passes for {split}: "
                f"expected {split_counts[split]}, wrote {writer['index']}"
            )

    summary = {
        "input_length": input_length,
        "output_length": output_length,
        "anchors_bed": str(anchors_bed),
        "skipped": skipped,
        "splits": split_counts,
    }
    with (output_dir / "dataset_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return {"skipped": skipped, **split_counts}


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
