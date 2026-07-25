# Ours v1 CIFAR-100: batch 128 ablation

This experiment changes exactly one training hyperparameter relative to the
selected Ours v1 CIFAR-100 run:

| Setting | Selected run | This ablation |
|---|---:|---:|
| Train batch size | 64 | **128** |

Everything else remains locked to
`cifar100_deit_ti_ours_researcher_sync_v1`: DeiT-Ti from scratch, fixed
32-pixel ResNet56 teacher, 300 epochs, AdamW, learning rate `5e-4`, minimum
learning rate `5e-6`, weight decay `0.05`, warm-up factor `0.001`, 20 warm-up
epochs, cosine decay, label smoothing `0`, drop path `0.1`, FP32, seed `1`,
test batch 200, direct-resize evaluation, researcher adaptive guidance, and
the larger-grid `32/16/14` policy.

The selected batch-64 result is `82.90%` Top-1. This batch-128 run must be
reported as a batch-size ablation and must not overwrite that result.

## H200 timing run

```bash
python methods/Ours/cifar100/batch128_ablation/train.py \
  --timing-run \
  --num-workers 4 \
  --run-name ours_v1_cifar100_batch128_timing_2ep_seed1
```

## H200 full run

```bash
python methods/Ours/cifar100/batch128_ablation/train.py \
  --student-epochs 300 \
  --batch-size 128 \
  --num-workers 4 \
  --run-name ours_v1_cifar100_batch128_300ep_seed1 \
  --output-dir /app/output
```

The entrypoint itself locks the dataset, batch size, protocol name, and all
other Ours v1 defaults. Explicit locked values are accepted when they match;
conflicting overrides stop before data loading or training.

The batch-128 lambda controls are isolated under
[`lambda_sweep/`](lambda_sweep/README.md).
