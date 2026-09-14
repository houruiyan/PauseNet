#!/usr/bin/env bash
# Pairwise window-count correlations between NET-seq / PRO-seq / GRO-seq (HEK293T)
# Input npz: /mnt/HDD8TB/houruiyan/pausing_site/2_train_model/evaluate/<tech>/test_profiles.npz
set -euo pipefail

cd /mnt/HDD8TB/houruiyan/pausing_site/4_plot_figure/fig5

EVAL_DIR=/mnt/HDD8TB/houruiyan/pausing_site/2_train_model/evaluate

# 1) Measured (observed) counts: technology-vs-technology correlation
python3 plot_tech_correlation.py \
    --netseq ${EVAL_DIR}/hek293t_netseq/test_profiles.npz \
    --proseq ${EVAL_DIR}/hek293t_proseq/test_profiles.npz \
    --groseq ${EVAL_DIR}/hek293t_groseq/test_profiles.npz \
    --cell HEK293T --outdir ./tech_correlation_out \
    --combined --shared-axis-limit

# 2) Model predictions: technology-vs-technology correlation
#    (ProCapNet Fig. 6C-style: do the models agree more than the measurements?)
python3 plot_tech_correlation.py \
    --netseq ${EVAL_DIR}/hek293t_netseq/test_profiles.npz \
    --proseq ${EVAL_DIR}/hek293t_proseq/test_profiles.npz \
    --groseq ${EVAL_DIR}/hek293t_groseq/test_profiles.npz \
    --source predicted \
    --cell HEK293T --outdir ./tech_correlation_out_predicted \
    --combined --shared-axis-limit
