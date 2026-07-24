# KD: Chaoyang / DeiT-Ti

`train.py` independently trains a DeiT-Ti student with standard logit KD from
the fixed Chaoyang ResNet56 teacher. The batch runners are convenience tools;
this entry point does not depend on them.

## Locked protocol

| Setting | Value |
|---|---:|
| Data | official mounted train/test splits |
| Epochs | 100 |
| Batch size | 64 |
| Warm-up | 5 epochs |
| Student / teacher input | 224 x 224 / 32 x 32 |
| Optimizer | AdamW |
| Initial LR / weight decay | `5e-4` / `0.05` |
| Label smoothing / seed | `0.1` / `42` |

The teacher receives the same augmented student view after bilinear resize to
32 x 32. KD uses temperature `4.0` and weight `0.9`; see
[`../README.md`](../README.md) for the exact objective.

## Run

```bash
python methods/KD/chaoyang/train.py --timing-run --data-dir /app/data/chaoyang --num-workers 4
python methods/KD/chaoyang/train.py --student-epochs 100 --data-dir /app/data/chaoyang --num-workers 4 --run-name kd_chaoyang_deit_ti_100ep --output-dir /app/output
```

Each run writes `student_best.pt`, `student_latest.pt`, and `summary.json` in
its run directory.

## Recorded epoch variants

The original dataset-specific run used 100 epochs. A later controlled rerun
changed only the epoch count to 300 and retained every other setting above;
both produced the same best Top-1 (`62.79%`). The B/C loss-sensitivity
comparison below is locked to that 300-epoch rerun so its full runs differ
only in KD temperature and CE/KD balance.

## Limited loss-sensitivity check

The completed `T=4`, KD-weight `0.9` result is not overwritten. Requested
`T=2` variants B (`CE:KD=0.50:0.50`) and C (`0.75:0.25`) are isolated under
[`loss_sensitivity`](loss_sensitivity/README.md), including a sequential
timing runner and separate output directories.
