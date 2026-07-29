# CUB-200 Table-1 seven-backbone extension

This experiment family extends the paper's Table 1 to CUB-200-2011 without
mixing it with the separate ResNet50-224 transfer or scratch families.

## Locked protocol

| Item | Value |
|---|---|
| Dataset | CUB-200-2011 official train/test split, `5,994 / 5,794` |
| Annotations | class labels only; no box, part, or attribute supervision |
| Teacher | fixed build-543 scratch CIFAR-style ResNet56, `32x32`, 300 epochs, **36.40%** |
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

Every guided student loads and hash-verifies the exact build-543 checkpoint
`teachers/checkpoints/cub200_table1_resnet56_32/teacher_resnet56_cub200_32_best.pt`
(epoch 275, **36.40%**, SHA-256
`06f75192b1c108c89e480843cb4f72dfb28aa762d7b11e7ac327333dd54b51f5`).
`train.py` rejects any other teacher manifest. The two-epoch teacher timing
checkpoint is measured only to estimate teacher training time and is never
supplied to a student.

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

## Completed first full-training group: build 543

The first bounded full run trained the scratch ResNet56-32 teacher and then
supplied that exact run's best checkpoint and manifest to DeiT-Ti LG and ALG:

```bash
python methods/table1_cub200/run_deit_lg_alg.py --full-run --data-dir /app/output/table1_cub200_deit_lg_alg_full_seed1/data/cub200 --output-dir /app/output/table1_cub200_deit_lg_alg_full_seed1 --num-workers 4
```

All three tasks completed for 300 epochs:

| Task | Best epoch | Best Top-1 | Last Top-1 |
|---|---:|---:|---:|
| ResNet56-32 teacher | 275 | **36.40%** | 36.12% |
| DeiT-Ti LG, batch 128 | 241 | **44.51%** | 43.61% |
| DeiT-Ti ALG, batch 128 | 286 | **47.70%** | 47.53% |

The verified artifacts are under `teachers/checkpoints/cub200_table1_resnet56_32/`,
`results/{LG,ALG}/cub200/table1_cub200_deit_ti_*`, and
`results/run_logs/h200_build-543_*`. The producer command above and its Issue
fields remain for provenance.

## Direct Table-1 student training

The original 36-task baseline matrix is complete. For any controlled rerun or
follow-up, do not retrain or substitute the teacher. Run the student directly
through `train.py`; its default `--teacher-root` is the fixed build-543
directory, and the exact hash is checked before training:

```bash
python methods/table1_cub200/train.py \
  --full-run \
  --student convit_ti \
  --method lg \
  --batch-size 128 \
  --data-dir /app/output/data/cub200 \
  --output-dir /app/output/table1_cub200_students \
  --num-workers 4
```

Valid student keys are `deit_ti`, `convit_ti`, `cvt_13`, `pit_ti`,
`pvtv2_b0`, `t2t_vit_7`, and `t2t_vit_14`. Guided LG, ALG, and Ours jobs all
fail closed if the manifest does not identify the build-543 teacher above.

## Completed DeiT-Ti Ours group: build 547

This bounded full run reused the fixed build-543 teacher and trained the two
remaining DeiT-Ti Ours settings:

```bash
python methods/table1_cub200/run_deit_ours.py --full-run --data-dir /app/output/table1_cub200_deit_ours_full_seed1/data/cub200 --output-dir /app/output/table1_cub200_deit_ours_full_seed1 --num-workers 4
```

The locked order was Ours-b64 followed by Ours-b128. Both scratch DeiT-Ti
students completed 300 epochs at 224x224. Ours-b64 reached **48.31%** at
epoch 263 (last **47.15%**); Ours-b128 reached **48.36%** at epoch 277
(last **47.84%**). Both summaries and checkpoints record the fixed build-543
teacher hash. Copyable Issue fields remain in `H200_DEIT_OURS_FULL_ISSUE.md`.

## Completed DeiT-Ti Vanilla baseline: build 548

Vanilla is the teacher-free scratch DeiT-Ti baseline at 224x224, batch 128,
and 300 epochs. The direct training entry point now checks and installs the
complete pinned Table-1 runtime (`timm`, `einops`, `fvcore`, `iopath`, and
`yacs`) before model construction:

```bash
python methods/table1_cub200/train.py --full-run --student deit_ti --method vanilla --batch-size 128 --data-dir /app/output/table1_cub200_deit_vanilla_full_seed1_v2/data/cub200 --output-dir /app/output/table1_cub200_deit_vanilla_full_seed1_v2 --run-name table1_cub200_deit_ti_vanilla_b128_full_300ep_seed1 --num-workers 4
```

The valid retry reached **17.69%** at epoch 156 and finished at **16.78%**.
The `v2` output root distinguishes it from the earlier job that stopped before
epoch 1 because `iopath` had not been installed. Copyable Issue fields remain
in `H200_DEIT_VANILLA_FULL_ISSUE.md`.

## Five-setting backbone runner and completed builds 551–556

`run_backbone_all.py` trains one student backbone in the locked order
Vanilla-b128, LG-b128, ALG-b128, Ours-b64, and Ours-b128. Vanilla remains
teacher-free; all four guided runs load and hash-check the fixed build-543
ResNet56-32 teacher. The runner validates every summary and best checkpoint
and prints the complete six-value Teacher/Vanilla/LG/ALG/Ours64/Ours128 line.

The first use was ConViT-Ti:

```bash
python methods/table1_cub200/run_backbone_all.py --full-run --student convit_ti --data-dir /app/output/table1_cub200_convit_all_full_seed1/data/cub200 --output-dir /app/output/table1_cub200_convit_all_full_seed1 --num-workers 4
```

Builds 551–556 used the same runner to complete all five settings for the six
non-DeiT backbones:

| Student | Method | Batch | Best epoch | Best Top-1 | Last Top-1 |
|---|---|---:|---:|---:|---:|
| ConViT-Ti | Vanilla | 128 | 261 | **22.94%** | 22.73% |
| ConViT-Ti | LG | 128 | 263 | **45.55%** | 45.15% |
| ConViT-Ti | ALG | 128 | 300 | **51.05%** | 51.05% |
| ConViT-Ti | Ours | 64 | 278 | **53.18%** | 52.90% |
| ConViT-Ti | Ours | 128 | 295 | **51.73%** | 51.35% |
| CvT-13 | Vanilla | 128 | 194 | **29.19%** | 27.98% |
| CvT-13 | LG | 128 | 99 | **43.01%** | 39.30% |
| CvT-13 | ALG | 128 | 247 | **46.63%** | 45.84% |
| CvT-13 | Ours | 64 | 248 | **47.98%** | 46.95% |
| CvT-13 | Ours | 128 | 224 | **46.82%** | 45.36% |
| PiT-Ti | Vanilla | 128 | 149 | **20.16%** | 20.06% |
| PiT-Ti | LG | 128 | 111 | **42.37%** | 40.30% |
| PiT-Ti | ALG | 128 | 196 | **43.63%** | 42.75% |
| PiT-Ti | Ours | 64 | 267 | **44.44%** | 43.70% |
| PiT-Ti | Ours | 128 | 221 | **42.72%** | 41.11% |
| PVTv2-B0 | Vanilla | 128 | 277 | **47.15%** | 46.70% |
| PVTv2-B0 | LG | 128 | 246 | **45.43%** | 45.32% |
| PVTv2-B0 | ALG | 128 | 291 | **50.05%** | 49.86% |
| PVTv2-B0 | Ours | 64 | 274 | **52.73%** | 52.24% |
| PVTv2-B0 | Ours | 128 | 254 | **52.69%** | 52.04% |
| T2T-ViT-7 | Vanilla | 128 | 224 | **26.73%** | 25.98% |
| T2T-ViT-7 | LG | 128 | 222 | **46.63%** | 45.72% |
| T2T-ViT-7 | ALG | 128 | 295 | **50.57%** | 50.47% |
| T2T-ViT-7 | Ours | 64 | 276 | **54.59%** | 54.14% |
| T2T-ViT-7 | Ours | 128 | 291 | **52.97%** | 52.76% |
| T2T-ViT-14 | Vanilla | 128 | 267 | **19.90%** | 19.57% |
| T2T-ViT-14 | LG | 128 | 103 | **46.05%** | 43.30% |
| T2T-ViT-14 | ALG | 128 | 254 | **48.45%** | 47.51% |
| T2T-ViT-14 | Ours | 64 | 226 | **49.00%** | 47.88% |
| T2T-ViT-14 | Ours | 128 | 259 | **47.98%** | 46.41% |

Every guided checkpoint records the fixed build-543 teacher hash. All
hierarchical Ours checkpoints preserve their learned per-block
projection/resize adapter state. With the earlier DeiT-Ti group, these results
complete all 35 baseline students plus the one fixed teacher. The same runner
remains available for controlled reruns. Copyable ConViT Issue fields are in
`H200_CONVIT_ALL_FULL_ISSUE.md`.
