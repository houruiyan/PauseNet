# PauseNet user smoke test

From the repository root, after installing Python >=3.9 and PauseNet:

```bash
pip install -e .
python examples/smoke_test/run_smoke_test.py
```

No GPU, network download, pretrained checkpoint or genomics extras are needed
after installation. A unique temporary directory is printed and retained for
inspection. Alternatively, select a **new** directory:

```bash
python examples/smoke_test/run_smoke_test.py --output-dir outputs/user-smoke-001
```

Existing output directories are refused to protect previous results.
The script generates 8 training, 4 validation and 4 test examples in the standard
PauseNet split format (2,114-bp sequence codes and 1,000-position profiles),
including ambiguous bases and a zero-count example in each split. All examples
are synthetic; the manifest does not claim real genomic coordinates.

The test runs the public training and evaluation CLI on CPU, using a small
8-channel, 2-block model for two epochs. This is **not** the manuscript's
128-channel, 11-block model and must not be used to assess biological accuracy.
The saved checkpoint is a toy model, not released pretrained weights.

Expected final message:

```text
PASS: dataset loading, CPU training, checkpoint reload and prediction checks.
```

Outputs include `data/{train,validation,test}/`, `smoke.yaml`,
`training/best_model.pt`, `training/training_history.tsv`,
`evaluation/test_profiles.npz` and `evaluation/test_metrics.json`.
Checks cover array shape, finite/nonnegative predictions, profile probabilities
summing to one, unchanged observed counts, correct zero-count masking and finite
losses. Numerical metric values can vary between PyTorch versions/platforms;
there is no accuracy threshold for this synthetic example.

To rerun individual CLI steps, use the absolute paths printed by the script:

```bash
pausenet train --config /path/to/smoke.yaml
pausenet evaluate --data-dir /path/to/data --checkpoint /path/to/training/best_model.pt --output-dir /path/to/new-evaluation --device cpu --num-workers 0 --batch-size 2
```

For the existing developer test suite, install `pytest` and run `python -m pytest
tests`. That suite is separate from this user-facing end-to-end smoke test.
