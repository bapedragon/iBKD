# CUB-200 Table-1 seven-backbone extension

This experiment family extends the paper's Table 1 to CUB-200-2011 without
mixing it with the separate ResNet50-224 transfer or scratch families.

## Locked protocol

| Item | Value |
|---|---|
| Dataset | CUB-200-2011 official train/test split, `5,994 / 5,794` |
| Annotations | class labels only; no box, part, or attribute supervision |
| Teacher | one shared scratch CIFAR-style ResNet56, `32x32`, 300 epochs |
| Students | DeiT-Ti, ConViT-Ti, CvT-13, PiT-Ti, PVTv2-B0, T2T-ViT-7, T2T-ViT-14 |
| Student initialization/input | scratch / `224x224` |
| Student schedule | 300 epochs, AdamW `5e-4 -> 5e-6`, weight decay `0.05`, 20-epoch warm-up |
| Augmentation | audited public-LG strong augmentation |
| Seed/precision | `1` / FP32 |
| Per-backbone runs | Vanilla-b128, LG-b128, ALG-b128, Ours-b64, Ours-b128 |

The complete matrix is **36 tasks**, not 29:

```text
1 teacher + 7 backbones * 5 student settings = 36
```

Every guided student loads and hash-verifies the same completed
`teacher_resnet56_cub200_32_best.pt`. The two-epoch teacher timing checkpoint is
measured only to estimate teacher training time and is never supplied to a
student.

## Backbone provenance

The seven architecture definitions and the exact public LG feature positions
come from:

- repository: <https://github.com/lkhl/tiny-transformers>
- commit: `d2165f74049c906b0afc9f957491960fb3c0cc8b`

| Student | Official LG selected blocks |
|---|---|
| DeiT-Ti | `0, 6, 11` |
| ConViT-Ti | `0, 6, 11` |
| CvT-13 | `0, 6, 11` |
| PiT-Ti | `0, 6, 11` |
| PVTv2-B0 | `0, 3, 7` |
| T2T-ViT-7 | `0, 3, 6` |
| T2T-ViT-14 | `0, 7, 13` |

LG keeps the official stage-wise 1x1 projections, larger-grid bilinear
alignment, summed mean MSE, and coefficient `2.5`. ALG changes only the
guidance controller to the published equations, window `50`, threshold
`-0.02`, `>=` stop boundary, and zero controller warm-up.

## Ours connection for hierarchical backbones

The existing Ours module aggregates all Transformer blocks, which requires the
blocks to have a common channel count and grid. DeiT-Ti and ConViT-Ti already
provide `192x14x14`. For CvT, PiT, PVTv2, and T2T, an explicit per-block
adapter performs:

1. a parameter-free identity when the block already has 192 channels,
   otherwise a learned 1x1 projection to 192 channels; and
2. bilinear resize to `14x14`.

The converted blocks then enter the unchanged Ours all-block aggregation,
alignment, and fusion modules. This adapter is saved with each result and must
be disclosed in the paper's implementation details. Ours is measured at both
batch 64 and batch 128; the method, optimizer, and loss are otherwise identical.

## Complete timing run

```bash
python methods/table1_cub200/run_timing.py --timing-run --data-dir /app/output/table1_cub200_7backbone_36task_timing_seed1_v2/data/cub200 --output-dir /app/output/table1_cub200_7backbone_36task_timing_seed1_v2 --num-workers 4
```

The runner installs pinned requirements when needed, downloads and validates
CUB once, and executes two full-dataset epochs for every task. It writes:

- `sequence_status.json` after every task;
- `timing_summary.csv` with all 36 individual estimates; and
- `timing_summary.json` with Teacher, students-only, total, Teacher-reuse, and
  600-minute pod-limit totals.

Successful completion ends with:

```text
[FINAL_TOTAL_ESTIMATE] tasks=36 teacher=... students35=... with_teacher=... reuse_teacher=...
[POD_LIMIT_CHECK] limit=600m estimated=...m status=PASS|FAIL
[FINAL_RESULT] summary=...
[SEQUENCE_DONE] completed_tasks=36/36
```

If one combination runs out of memory, the sequence stops on that exact task
and preserves all earlier summaries. The batch or experiment grouping must
then be revised based on the recorded failure rather than silently omitting the
model.

## First full-training group: Teacher + DeiT-Ti LG/ALG

The first bounded full run deliberately retrains the scratch ResNet56-32
teacher and then supplies that exact run's best checkpoint and manifest to
DeiT-Ti LG and ALG:

```bash
python methods/table1_cub200/run_deit_lg_alg.py --full-run --data-dir /app/output/table1_cub200_deit_lg_alg_full_seed1/data/cub200 --output-dir /app/output/table1_cub200_deit_lg_alg_full_seed1 --num-workers 4
```

All three tasks run for 300 epochs. The measured timing estimate is 2h 53m
58s, below the 600-minute pod limit. The final log repeats the Teacher, LG,
and ALG best Top-1 values. Copyable Issue fields are in
`H200_DEIT_LG_ALG_FULL_ISSUE.md`.
