# Ours V1 learnable-aggregation alpha extraction

This utility extracts the softmax-normalized learnable-aggregation
coefficients from the repository's selected Ours V1 reproduction best
checkpoints for CIFAR-100, Flowers-102, and Chaoyang. These values describe
the named checkpoints and must not be labeled as values extracted from a
different original-paper checkpoint.

It performs no training, dataset loading, model forward pass, or Ours V2
analysis. For every dataset, it verifies:

- checkpoint identity: `method=Ours`, `student=deit_ti`, and the expected
  dataset;
- raw aggregation-logit shape: `3 x 12`;
- finite, non-negative softmax coefficients with each stage summing to one;
- numerical agreement with `run_summary.json["aggregation_weights"]` within
  `1e-6`.

Run from the repository root:

```bash
python methods/Ours/aggregation_alpha/extract_alpha.py \
  --output-dir outputs/ours_v1_aggregation_alpha
```

The command writes:

```text
outputs/ours_v1_aggregation_alpha/
├── aggregation_alpha.json
├── aggregation_alpha.csv
└── aggregation_alpha.md
```

Teacher stages and student blocks use one-based indexing in all output files.
The JSON additionally records checkpoint identity, epoch, accuracy, SHA-256,
the complete `3 x 12` matrix, and the three largest coefficients per stage.
