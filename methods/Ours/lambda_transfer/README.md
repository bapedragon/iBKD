# Ours v1 lambda=0.25 cross-dataset transfer

This control tests whether the CIFAR-100 lambda=0.25 improvement transfers to
Flowers-102 and Chaoyang. It reuses the completed, artifact-verified
lambda=0.5 runs and changes only the convex feature-loss balance:

```text
L_feature = 0.25 * L_fuse + 0.75 * L_align
```

## Locked comparisons

| Dataset | lambda=0.5 reference | Reference protocol |
|---|---:|---|
| Flowers-102 | 74.81% | DeiT-Ti, batch 64, 300 epochs, seed 1 |
| Chaoyang | 81.11% | DeiT-Ti, batch 64, 300 epochs, seed 1 |

The log-only Chaoyang 81.95% value is not used as the paired reference because
its individual checkpoint and summary were never imported. The 81.11% run has
the complete checkpoint, summary, and producer log required for a controlled
comparison.

The runner audits both reference summaries before training. The student,
teacher identity, data split, batches, optimizer, schedule, seed, precision,
Ours source, grid policy, and adaptive controller remain fixed. Output paths,
protocol labels, and lambda are the only intentional differences.

## H200 commands

Optional two-epoch preflight:

```bash
python methods/Ours/lambda_transfer/run_flowers_chaoyang.py \
  --timing-run --num-workers 4
```

Full paired run:

```bash
python methods/Ours/lambda_transfer/run_flowers_chaoyang.py \
  --full-run --num-workers 4 \
  --output-dir /app/output/ours_v1_lambda_0p25_flowers_chaoyang_300ep_seed1
```

Flowers-102 is downloaded and checksum-validated under `./data` when absent.
Chaoyang must be mounted at `/app/data/chaoyang`. The full run reuses the
committed teacher checkpoints under `teachers/checkpoints/`.

The final line prints both new best Top-1 values:

```text
[FINAL_TOP1_SUMMARY_LAMBDA_0P25] Flowers102=...% Chaoyang=...%
```
