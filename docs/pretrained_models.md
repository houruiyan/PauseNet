# Pretrained Models

Pretrained PauseNet weights are distributed outside the Git repository because
checkpoint files can be large.

Recommended release layout:

```text
pretrained/
  hek293t_netseq/
    model_card.md
    metrics.json
    config.yaml
```

Place checkpoint files in GitHub Releases, Zenodo or Hugging Face, then link
them from the model card.

## Using a Pretrained Model

```bash
pausenet evaluate \
  --data-dir /path/to/user_dataset \
  --checkpoint /path/to/pausenet_hek293t_netseq.pt \
  --output-dir outputs/user_dataset_eval \
  --split test
```
