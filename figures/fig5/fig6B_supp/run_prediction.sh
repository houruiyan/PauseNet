/mnt/HDD8TB/houruiyan/env/conda_envs/testPauseNet/bin/python \
  2.plot_cross_cell_model_test_heatmap.py \
  --model HEK293T=/mnt/HDD8TB/houruiyan/pausing_site/2_train_model/train/hek293t_netseq/best_model.pt \
  --model K562=/mnt/HDD8TB/houruiyan/pausing_site/2_train_model/train/k562_mnetseq/best_model.pt \
  --model HeLaS3=/mnt/HDD8TB/houruiyan/pausing_site/2_train_model/train/helas3_netseq/best_model.pt \
  --model MOLT4=/mnt/HDD8TB/houruiyan/pausing_site/2_train_model/train/molt4_netseq/best_model.pt \
  --test HEK293T=/mnt/HDD8TB/houruiyan/pausing_site/data/HEK293T_NETseq/dataset \
  --test K562=/mnt/HDD8TB/houruiyan/pausing_site/data/K562_mNETseq/dataset \
  --test HeLaS3=/mnt/HDD8TB/houruiyan/pausing_site/data/HeLaS3_NETseq/dataset \
  --test MOLT4=/mnt/HDD8TB/houruiyan/pausing_site/data/MOLT4_NETseq/dataset \
  --device auto \
  --vmin 0.40 \
  --vmax 0.65
