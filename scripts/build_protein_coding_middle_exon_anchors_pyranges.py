#!/usr/bin/env python3
"""Build TSS, TES, 5'SS and 3'SS BED anchors with PyRanges.

Workflow
--------
1. Keep protein-coding transcripts with at least three exons.
2. For each gene, select the transcript with the longest genomic span.
3. Order its exons in the 5'-to-3' transcription direction.
4. Select the middle exon.  If the exon count is even, select the first of
   the two middle exons in the transcription direction (the 5'-side one).
5. Emit TSS, TES and the middle exon's 5'SS/3'SS as one-base BED intervals.

All coordinates are zero-based and half-open.  A boundary anchor ``p`` is
represented in BED as ``[p, p + 1)``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyranges as pr


BED_COLUMNS = ["Chromosome", "Start", "End", "Name", "Score", "Strand"]


def protein_coding_mask(frame: pd.DataFrame) -> pd.Series:
    """Return a robust protein-coding transcript mask for GENCODE GTF files."""
    for column in ("transcript_type", "transcript_biotype"):
        if column in frame.columns:
            return frame[column].eq("protein_coding")
    for column in ("gene_type", "gene_biotype"):
        if column in frame.columns:
            return frame[column].eq("protein_coding")
    raise ValueError(
        "The GTF has none of transcript_type, transcript_biotype, "
        "gene_type or gene_biotype."
    )


def make_anchor_bed(
    frame: pd.DataFrame,
    anchor_type: str,
    positions: pd.Series,
) -> pd.DataFrame:
    """Create BED6 rows for one anchor type."""
    anchor = pd.DataFrame(
        {
            "Chromosome": frame["Chromosome"].astype(str),
            "Start": positions.astype(np.int64),
            "Name": (
                frame["gene_id"].astype(str)
                + "|"
                + frame["transcript_id"].astype(str)
                + "|"
                + anchor_type
            ),
            "Score": 0,
            "Strand": frame["Strand"].astype(str),
        }
    )
    anchor["End"] = anchor["Start"] + 1
    return anchor[BED_COLUMNS]


def build_anchor_bed(gtf_path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    annotation = pr.read_gtf(str(gtf_path))
    annotation_df = annotation.df

    required = {
        "Chromosome",
        "Start",
        "End",
        "Strand",
        "Feature",
        "gene_id",
        "transcript_id",
    }
    missing = required.difference(annotation_df.columns)
    if missing:
        raise ValueError(f"GTF is missing required columns: {sorted(missing)}")

    # Use transcript records to calculate transcript span.
    transcript_df = annotation_df[
        annotation_df["Feature"].eq("transcript")
    ].copy()
    transcript_df = transcript_df[protein_coding_mask(transcript_df)].copy()
    transcript_df = transcript_df[
        [
            "Chromosome",
            "Start",
            "End",
            "Strand",
            "gene_id",
            "transcript_id",
        ]
    ].drop_duplicates(["gene_id", "transcript_id"])

    # Keep protein-coding exon records and count exons per transcript.
    exon_df = annotation_df[annotation_df["Feature"].eq("exon")].copy()
    exon_df = exon_df[protein_coding_mask(exon_df)].copy()
    exon_df = exon_df[
        [
            "Chromosome",
            "Start",
            "End",
            "Strand",
            "gene_id",
            "transcript_id",
        ]
    ].drop_duplicates()

    transcript_keys = ["gene_id", "transcript_id"]
    exon_counts = (
        exon_df.groupby(transcript_keys, observed=True)
        .size()
        .rename("n_exons")
        .reset_index()
    )

    # At least three exons are required so the selected middle exon is internal
    # and has both a 3' splice site and a 5' splice site.
    eligible_transcript_df = transcript_df.merge(
        exon_counts[exon_counts["n_exons"] >= 3],
        on=transcript_keys,
        how="inner",
        validate="one_to_one",
    )
    eligible_transcript_df["transcript_length"] = (
        eligible_transcript_df["End"] - eligible_transcript_df["Start"]
    )

    # Longest transcript span per gene.  transcript_id breaks exact ties
    # deterministically.
    longest_transcript_df = (
        eligible_transcript_df.sort_values(
            ["gene_id", "transcript_length", "transcript_id"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        .drop_duplicates("gene_id", keep="first")
        .reset_index(drop=True)
    )

    selected_keys = longest_transcript_df[transcript_keys]
    selected_exon_df = exon_df.merge(
        selected_keys,
        on=transcript_keys,
        how="inner",
        validate="many_to_one",
    )

    # Sort exons in transcription direction: ascending coordinates on '+',
    # descending coordinates on '-'.
    selected_exon_df["transcription_sort"] = np.where(
        selected_exon_df["Strand"].eq("+"),
        selected_exon_df["Start"],
        -selected_exon_df["Start"],
    )
    selected_exon_df = selected_exon_df.sort_values(
        transcript_keys + ["transcription_sort", "End"],
        kind="mergesort",
    )
    selected_exon_df["exon_rank"] = (
        selected_exon_df.groupby(transcript_keys, observed=True).cumcount()
    )
    selected_exon_df["n_exons"] = (
        selected_exon_df.groupby(transcript_keys, observed=True)[
            "transcript_id"
        ].transform("size")
    )
    selected_exon_df["middle_rank"] = (
        (selected_exon_df["n_exons"] - 1) // 2
    )
    middle_exon_df = selected_exon_df[
        selected_exon_df["exon_rank"].eq(selected_exon_df["middle_rank"])
    ].copy()

    if len(middle_exon_df) != len(longest_transcript_df):
        raise RuntimeError(
            "Expected exactly one middle exon for every selected transcript."
        )

    # Transcript boundary anchors.  End is the zero-based half-open boundary.
    tss_position = np.where(
        longest_transcript_df["Strand"].eq("+"),
        longest_transcript_df["Start"],
        longest_transcript_df["End"],
    )
    tes_position = np.where(
        longest_transcript_df["Strand"].eq("+"),
        longest_transcript_df["End"],
        longest_transcript_df["Start"],
    )

    # For an internal exon:
    #   '+' strand: 3'SS = exon Start, 5'SS = exon End
    #   '-' strand: 5'SS = exon Start, 3'SS = exon End
    five_ss_position = np.where(
        middle_exon_df["Strand"].eq("+"),
        middle_exon_df["End"],
        middle_exon_df["Start"],
    )
    three_ss_position = np.where(
        middle_exon_df["Strand"].eq("+"),
        middle_exon_df["Start"],
        middle_exon_df["End"],
    )

    anchor_bed_df = pd.concat(
        [
            make_anchor_bed(
                longest_transcript_df,
                "TSS",
                pd.Series(tss_position, index=longest_transcript_df.index),
            ),
            make_anchor_bed(
                longest_transcript_df,
                "TES",
                pd.Series(tes_position, index=longest_transcript_df.index),
            ),
            make_anchor_bed(
                middle_exon_df,
                "5SS",
                pd.Series(five_ss_position, index=middle_exon_df.index),
            ),
            make_anchor_bed(
                middle_exon_df,
                "3SS",
                pd.Series(three_ss_position, index=middle_exon_df.index),
            ),
        ],
        ignore_index=True,
    )

    # PyRanges supplies chromosome-aware genomic sorting.
    anchor_bed_df = pr.PyRanges(anchor_bed_df).sort().df[BED_COLUMNS]
    summary = {
        "protein_coding_transcripts": len(transcript_df),
        "eligible_transcripts_with_at_least_3_exons": len(
            eligible_transcript_df
        ),
        "selected_genes": len(longest_transcript_df),
        "bed_rows": len(anchor_bed_df),
    }
    return anchor_bed_df, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gtf", required=True, type=Path)
    parser.add_argument("--output-bed", required=True, type=Path)
    args = parser.parse_args()

    anchor_bed_df, summary = build_anchor_bed(args.gtf)
    args.output_bed.parent.mkdir(parents=True, exist_ok=True)
    anchor_bed_df.to_csv(
        args.output_bed,
        sep="\t",
        header=False,
        index=False,
    )

    for key, value in summary.items():
        print(f"{key}: {value:,}")
    print(f"output: {args.output_bed}")


if __name__ == "__main__":
    main()
