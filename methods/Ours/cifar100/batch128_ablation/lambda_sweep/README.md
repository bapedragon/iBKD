# Ours v1 CIFAR-100 batch-128 lambda controls

The completed batch-128 baseline used `lambda=0.5` and reached `82.60%`
Best Top-1 (`82.46%` latest). This folder runs the two remaining requested
controls sequentially:

1. `lambda=0`: `L_feature = L_align`
2. `lambda=0.25`: `L_feature = 0.25 L_fuse + 0.75 L_align`

Every other setting is locked to the same Ours v1 batch-128 protocol:
300 epochs, seed 1, FP32, train/test batch 128/200, AdamW, learning rate
`5e-4`, minimum learning rate `5e-6`, weight decay `0.05`, 20-epoch warm-up,
label smoothing 0, drop path 0.1, fixed 32-pixel teacher, 224-pixel student,
adaptive guidance, and the larger-grid `32/16/14` policy.

Full H200 sequence:

```bash
python methods/Ours/cifar100/batch128_ablation/lambda_sweep/run_lambda_0_0p25.py \
  --full-run \
  --num-workers 4
```

Results are isolated below:

```text
/app/output/ours_v1_cifar100_batch128_lambda_0_0p25_300ep_seed1/
├── lambda_0/
├── lambda_0p25/
├── sequence_status.json
└── sequence_summary.json
```

The sequence prints both Best and Latest Top-1 values at the end. Based on the
completed batch-128 baseline time, the two runs are expected to take about
6 hours 25 minutes, below the 10-hour pod limit.
