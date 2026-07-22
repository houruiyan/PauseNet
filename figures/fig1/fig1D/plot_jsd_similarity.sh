python plot_jsd_similarity.py \
  --manifest <test/manifest.tsv> \
  --per-window-jsd <per_window_jsd.npz> \
  --output-dir <output_dir> \
  --resolution-bp 20 \
  --min-count-exclusive 100 \
  --global-minmax \
  --x-label "Normalized (1-JSD)" \
  --hide-count-annotation
