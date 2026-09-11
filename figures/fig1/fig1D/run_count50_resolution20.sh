#!/bin/bash 

/mnt/HDD8TB/houruiyan/env/conda_envs/testPauseNet/bin/python \
  3.plot_normalized_jsd_by_region.py \
  --npz /mnt/HDD8TB/houruiyan/pausing_site/2_train_model/evaluate/hek293t_netseq/test_profiles.npz \
  --pos-bw /mnt/HDD8TB/houruiyan/pausing_site/data/HEK293T_NETseq/merge/merged/HEK293T.merged.pos.bw \
  --neg-bw /mnt/HDD8TB/houruiyan/pausing_site/data/HEK293T_NETseq/merge/merged/HEK293T.merged.neg.bw \
  --signal-unit 1 \
  --resolution 20 \
  --count-threshold 50 \
  --output-pdf hek293t_netseq_normalized_js_distance_res20_count50.pdf 
