#!/bin/bash

make_tracks_file --trackFiles ./all_genes_transcripts.gtf \
	./K562_NETseq.hg19.bw \
	./K562_CTCF.hg19.bw \
	./HeLaS3_CTCF.hg19.bw \
       -o  tf_tracks.ini
