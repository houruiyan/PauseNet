#!/bin/bash
# Render the 100-kb figure (genes / NET-seq K562 / CTCF motif FIMO / CTCF K562 / CTCF HeLaS3)
# Requires: pygenometracks env at /mnt/HDD8TB/houruiyan/env/conda_envs/pygenometracks
# Inputs: motif_tracks.ini + ctcf_motifs_fimo.display.bed (motif center +/-250 bp for visibility;
#         true 19-bp hits: .../DeepSHAP/k562_mnetseq_TF/run_fimo/fimo_100kb_annotated.tsv)
cd "$(dirname "$0")"
/mnt/HDD8TB/houruiyan/env/conda_envs/pygenometracks/bin/pyGenomeTracks \
    --tracks motif_tracks.ini --region chrX:30185000-30285000 \
    --outFileName motif_track.pdf
