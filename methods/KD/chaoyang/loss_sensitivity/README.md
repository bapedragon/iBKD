# Chaoyang KD loss sensitivity

This folder keeps the limited Chaoyang logit-KD loss search separate from the
completed baseline and from other distillation methods.

## Purpose

The completed baseline A used the RepDistiller CIFAR-100 example coefficients:

```text
A: T=4, 0.10 * CE + 0.90 * KD
```

This runner evaluates only the two requested alternatives:

```text
B: T=2, 0.50 * CE + 0.50 * KD
C: T=2, 0.75 * CE + 0.25 * KD
```

## Locked controls

Except for `temperature` and `KD weight`, B and C are identical to the
completed 300-epoch Chaoyang KD rerun:

| Setting | Locked value |
|---|---:|
| Dataset | official Chaoyang train 4,021 / test 2,139 |
| Teacher | fixed 32 x 32 ResNet56 checkpoint |
| Student | scratch DeiT-Ti, 224 x 224 |
| Epochs | 300 |
| Batch size | 64 |
| Optimizer | AdamW |
| LR / weight decay | `5e-4` / `0.05` |
| Warm-up / schedule | 5 epochs / cosine |
| Label smoothing | `0.1` |
| Seed | `42` |
| Evaluation | test Top-1; best and latest saved separately |

No validation split is introduced in this limited comparison. Consequently,
selecting the largest test result is a test-guided sensitivity check, not an
unbiased validation-selected estimate. All A/B/C results should remain in the
experiment record.

## H200 timing run

The timing mode runs two full-data epochs for B and then C. It checks the
pipeline and estimates the 300-epoch runtime; its accuracy is not a final
result.

```bash
python methods/KD/chaoyang/loss_sensitivity/run.py \
  --timing-run \
  --data-dir /app/data/chaoyang \
  --num-workers 4
```

Timing outputs default to:

```text
./outputs/kd_chaoyang_loss_sensitivity/
├── variant_b_T2_ce0.50_kd0.50_timing_2ep_seed42/
├── variant_c_T2_ce0.75_kd0.25_timing_2ep_seed42/
├── status.json
└── summary.json
```

## H200 full run

After the timing run succeeds:

```bash
python methods/KD/chaoyang/loss_sensitivity/run.py \
  --full-run \
  --data-dir /app/data/chaoyang \
  --output-dir /app/output/kd_chaoyang_loss_sensitivity_300ep_seed42 \
  --num-workers 4
```

The full-run output root contains independent B and C run directories. Each
directory contains `student_best.pt`, `student_latest.pt`, and `summary.json`;
the root also contains a combined `summary.json`.
