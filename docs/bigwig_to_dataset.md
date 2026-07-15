# Build a Fixed Gene-Structure PauseNet Dataset from Strand-Specific bigWigs

`pausenet prepare-bigwig` builds the standard PauseNet dataset from a gene
annotation, a matched reference genome and positive/negative strand signal
tracks. It does not accept arbitrary peak-centered or anchor-centered windows.
Every example is one of five predefined 1-kb windows around a gene-structure
landmark.

## Required inputs

1. Positive-strand bigWig, for example `sample.pos.bw`.
2. Negative-strand bigWig, for example `sample.neg.bw`.
3. Reference genome FASTA matching the bigWig coordinates.
4. Chromosome-size file matching the FASTA.
5. A GTF or GTF.GZ annotation containing exon records.

Optional input:

- An expressed-gene BED-like file. Identifiers from columns 4 onward are used
  to restrict the dataset to expressed genes.

## Gene and transcript selection

The converter reads exon records from the GTF, optionally restricts them to
protein-coding transcripts, and keeps transcripts with at least two exons by
default. For each eligible expressed gene it selects one representative
transcript: the transcript with the largest summed exon length, then the
largest genomic span, then the most exons.

The selected transcript supplies one TSS and one TES. Each exon junction
supplies one 5'SS and one 3'SS. To prevent unusually intron-rich genes from
dominating the dataset, at most `--max-ss-per-gene` junctions are retained per
gene; if limiting is necessary, they are sampled evenly across the transcript.

## Fixed five-window scheme

All intervals below are in the transcription direction and are half-open,
relative to the landmark coordinate. Every interval is a distinct 1-kb model
example.

| Landmark | w0 | w1 | w2 | w3 | w4 |
| --- | --- | --- | --- | --- | --- |
| TSS | -1000..0 | 0..+1000 | +1000..+2000 | +2000..+3000 | +3000..+4000 |
| 5'SS | -2500..-1500 | -1500..-500 | -500..+500 | +500..+1500 | +1500..+2500 |
| 3'SS | -2500..-1500 | -1500..-500 | -500..+500 | +500..+1500 | +1500..+2500 |
| TES | -4000..-3000 | -3000..-2000 | -2000..-1000 | -1000..0 | 0..+1000 |

Thus the landmark is on a window boundary for TSS and TES, and at the centre
of `w2` for 5'SS and 3'SS. A 1-kb output window receives 557 bp of context on
each side, producing the fixed 2,114-bp model input.

## Command

Install PauseNet with the optional genomics dependencies:

```bash
pip install -e ".[genomics]"
```

Then run:

```bash
pausenet prepare-bigwig \
  --pos-bw /path/to/sample.pos.bw \
  --neg-bw /path/to/sample.neg.bw \
  --fasta /path/to/hg38.fa \
  --gtf /path/to/genes.gtf.gz \
  --expressed-genes-bed /path/to/expressed_genes.bed \
  --chrom-sizes /path/to/hg38.chrom.sizes \
  --output-dir /path/to/pausenet_dataset \
  --train-chroms chr1,chr2,chr3,chr4,chr5,chr6,chr7,chr9,chr10 \
  --validation-chroms chr8 \
  --test-chroms chr11,chr12 \
  --max-ss-per-gene 20 \
  --cell-line HEK293T \
  --assay NET-seq
```

Chromosome sets must be disjoint. Any chromosome outside the supplied sets is
assigned to `--default-split`, which is `train` by default. Specify all three
sets explicitly for a chromosome-held-out evaluation.

## Strand orientation and labels

Every output window is converted to the 5' to 3' transcription direction:

- `+` strand examples use the positive-strand bigWig and forward genomic DNA.
- `-` strand examples use the negative-strand bigWig; the DNA sequence is
  reverse-complemented and the 1-kb profile is reversed.

The converter treats negative bigWig values as display convention and uses the
absolute value as read signal. It writes `sequence_codes.npy`, `profiles.npy`,
`counts.npy`, `profile_loss_mask.npy`, `anchor_type_codes.npy`,
`original_strands.npy` and `manifest.tsv` to every split directory.

The manifest records `anchor_type`, `anchor_position`, `window_index`,
`relative_start`, `relative_end`, and the actual genomic input/output
coordinates. These fields let downstream analyses separate the five windows
from the same landmark and retain the original anchor context.

## Profile mask

`profile_loss_mask.npy` is 1 when the total signal in the 1-kb output window
is at least `--profile-min-count` (default 1). Count loss still uses all
examples; the mask excludes zero-signal profiles from the multinomial profile
loss and profile JSD metrics.
