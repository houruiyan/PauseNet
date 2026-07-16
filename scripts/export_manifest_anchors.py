#!/usr/bin/env python3
"""Export existing PauseNet windows as reusable --anchors-bed rows."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


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
REQUIRED_COLUMNS = {"chrom", "output_start", "output_end", "sample_id", "original_strand"}


def parse_manifest_argument(value: str) -> tuple[str, Path]:
    try:
        split, path = value.split("=", maxsplit=1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use --manifest split=/path/to/manifest.tsv") from exc
    if split not in {"train", "validation", "test"}:
        raise argparse.ArgumentTypeError("Split must be train, validation, or test")
    return split, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        action="append",
        required=True,
        type=parse_manifest_argument,
        help="Split and source manifest, e.g. train=/path/train/manifest.tsv. Repeat for each split.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--expected-count", type=int)
    args = parser.parse_args()

    manifests = dict(args.manifest)
    if set(manifests) != {"train", "validation", "test"}:
        parser.error("Provide exactly one manifest for train, validation, and test")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    with args.output.open("w", newline="") as output_handle:
        writer = csv.DictWriter(output_handle, fieldnames=OUTPUT_COLUMNS, delimiter="\t")
        writer.writeheader()
        for split in ("train", "validation", "test"):
            manifest_path = manifests[split]
            with manifest_path.open(newline="") as input_handle:
                reader = csv.DictReader(input_handle, delimiter="\t")
                missing = REQUIRED_COLUMNS.difference(reader.fieldnames or [])
                if missing:
                    raise ValueError(f"{manifest_path} is missing columns: {sorted(missing)}")
                for row in reader:
                    start = int(row["output_start"])
                    end = int(row["output_end"])
                    if end - start != 1000:
                        raise ValueError(f"Unexpected output length in {row['sample_id']}: {end - start}")
                    anchor_type = row.get("anchor_type") or row.get("sample_type") or "region"
                    writer.writerow(
                        {
                            "chrom": row["chrom"],
                            "start": start,
                            "end": end,
                            "sample_id": row["sample_id"],
                            "score": 0,
                            "strand": row["original_strand"],
                            "region_type": anchor_type,
                            "gene_id": row.get("gene_id", ""),
                            "gene_name": row.get("gene_name", ""),
                            "transcript_id": row.get("transcript_id", ""),
                            "split": split,
                            "anchor_type": anchor_type,
                            "anchor_index": row.get("anchor_index", ""),
                            "anchor_position": row.get("anchor_position", ""),
                            "window_index": row.get("window_index", ""),
                            "relative_start": row.get("relative_start", ""),
                            "relative_end": row.get("relative_end", ""),
                            "source_array_index": row.get("array_index", ""),
                        }
                    )
                    counts[split] += 1
                    counts[f"{split}:{anchor_type}"] += 1

    total = sum(counts[split] for split in ("train", "validation", "test"))
    if args.expected_count is not None and total != args.expected_count:
        raise SystemExit(f"Expected {args.expected_count:,} anchors, wrote {total:,}")
    summary = {"total": total, "splits": {split: counts[split] for split in ("train", "validation", "test")}}
    for split in ("train", "validation", "test"):
        summary["splits"][split] = {
            "total": counts[split],
            "by_region": {
                region: counts[f"{split}:{region}"]
                for region in ("TSS", "5SS", "3SS", "TES")
            },
        }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
