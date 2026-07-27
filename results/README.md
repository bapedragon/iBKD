# Curated student results

This directory is the compact, reporting-oriented view of completed H200
student runs. Raw Pod output folders contained repeated batch wrappers and both
`best` and `latest` checkpoints. Here they are normalized to:

```text
results/
├── <Method>/<dataset>/<protocol-id>/
│   ├── run_summary.json
│   └── student_best.pt
├── PENDING_IMPORTS.md
├── run_logs/
└── CHECKSUMS.sha256
```

The protocol-ID directory is mandatory. No checkpoint or summary may be placed
directly under a dataset directory. This prevents a new researcher-sync run
from overwriting or being mistaken for an older run of the same method and
dataset. The canonical IDs currently used are:

| Protocol ID | Meaning |
|---|---|
| `generic_kd_v2_300ep_seed42` | completed CIFAR-100 generic KD-family run |
| `generic_kd_v2_200ep_seed42_historical` | historical Flowers generic run |
| `generic_kd_v2_100ep_seed42_historical` | historical Chaoyang generic run |
| `generic_kd_300ep_epoch_only_v1_seed42` | Flowers/Chaoyang generic rerun with the earlier recipe and a 300-epoch horizon |
| `pre_researcher_sourcegrid_300ep_seed42_historical` | pre-sync Ours CIFAR run |
| `pre_researcher_papergrid_100ep_seed42_historical` | pre-sync Ours Chaoyang run |
| `pre_researcher_batch128_300ep_seed1_historical` | pre-sync ALG Chaoyang run |
| `researcher_sync_v1_300ep_seed1` | researcher-synchronized Ours/ALG family; Flowers uses train+val 2,040 / test 6,149 |
| `researcher_sync_v2_official_three_way_300ep_seed1_historical` | historical Flowers run: train 1,020, val-best 1,020, final test 6,149 once |
| `table4_grid_permuted_researcher_sync_v1_300ep_seed1_permseed1` | Table 4 global, fixed, stage-wise spatial permutation; the same permutation feeds K/V and both feature targets |
| `table4_kv_independent_researcher_sync_v1_300ep_seed1_k1_v1001` | Table 4 global permutation with independent K/V seeds (`1` and `1001`) |
| `table4_local_patch2_researcher_sync_v1_300ep_seed1_permseed1` | Table 4 fixed permutation restricted to non-overlapping `2x2` windows |
| `table4_token_space_researcher_sync_v1_300ep_seed1` | Table 4 token-space Linear-Q/K/V control |
| `table7_lambda_0_researcher_sync_v1_300ep_seed1` | Table 7 alignment-only endpoint (`lambda=0`) |
| `table7_lambda_0p25_researcher_sync_v1_300ep_seed1` | Table 7 `0.25 L_fuse + 0.75 L_align` |
| `table7_lambda_0p75_researcher_sync_v1_300ep_seed1` | Table 7 `0.75 L_fuse + 0.25 L_align` |
| `table7_lambda_1_researcher_sync_v1_300ep_seed1` | Table 7 fusion-only endpoint (`lambda=1`) |
| `table7_lambda_0_relative_position_v1_300ep_seed1` | Ours V2 relative-position Table 7 alignment-only endpoint |
| `table7_lambda_0p5_relative_position_v1_300ep_seed1` | Ours V2 relative-position Table 7 balanced reference |
| `paper_lg_v2_trainval_test_b128_300ep_seed1` | selected pure-ALG Flowers train batch 128 |
| `paper_lg_v2_trainval_test_b64_300ep_seed1` | pure-ALG Flowers train batch 64 control |
| `paper_source_v2_trainval_test_b128_300ep_seed1` | Ours Flowers batch-128 protocol-separated control |
| `paper_lg_v2_b128_300ep_seed1` | pure-ALG Chaoyang batch-128 control |
| `paper_lg_v2_b64_300ep_seed1` | selected pure-ALG Chaoyang batch-64 result |
| `cifar100_locked_b64_v1_300ep_seed1` | Ours Chaoyang CIFAR-100-locked batch-64 control |
| `cub200_deit_ti_official_lg_v1_300ep_seed1` | CUB-200 transfer of official static LG mechanics |
| `cub200_deit_ti_alg_paper_official_lg_v1_300ep_seed1` | CUB-200 ALG paper controller on the official LG base |
| `cub200_deit_ti_ours_scratch_teacher_v1_300ep_seed1` | CUB-200 Ours with the shared scratch ResNet56 teacher |
| `cub200_deit_ti_lg_resnet50_224_transfer_adaptation_v1_300ep_seed1` | CUB-200 LG adaptation with the ImageNet1K-V2 ResNet50-224 teacher |
| `cub200_deit_ti_alg_resnet50_224_transfer_adaptation_v1_300ep_seed1` | CUB-200 ALG adaptation with the ImageNet1K-V2 ResNet50-224 teacher |
| `cub200_deit_ti_ours_resnet50_224_transfer_v1_300ep_seed1` | CUB-200 Ours with the ImageNet1K-V2 ResNet50-224 teacher |
| `cub200_deit_ti_ce_lg_official_b128_300ep_seed1` | CUB-200 teacher-free Vanilla control for the LG/ALG batch-128 profile |
| `cub200_deit_ti_lg_resnet50_224_scratch_teacher_ablation_v1_300ep_seed1` | CUB-200 LG with the random-init ResNet50-224 teacher |
| `cub200_deit_ti_alg_resnet50_224_scratch_teacher_ablation_v1_300ep_seed1` | CUB-200 ALG with the random-init ResNet50-224 teacher |
| `cub200_deit_ti_ce_ours_current_b64_300ep_seed1` | CUB-200 teacher-free Vanilla control for the Ours batch-64 profile |
| `cub200_deit_ti_ours_resnet50_224_scratch_teacher_ablation_v1_300ep_seed1` | CUB-200 Ours with the random-init ResNet50-224 teacher |
| `table1_cub200_deit_ti_lg_b128_full_300ep_seed1` | CUB Table-1 DeiT-Ti LG using the fixed build-543 ResNet56-32 teacher |
| `table1_cub200_deit_ti_alg_b128_full_300ep_seed1` | CUB Table-1 DeiT-Ti ALG using the fixed build-543 ResNet56-32 teacher |
| `table1_cub200_deit_ti_vanilla_b128_full_300ep_seed1` | CUB Table-1 teacher-free DeiT-Ti Vanilla baseline |
| `table1_cub200_deit_ti_ours_b64_full_300ep_seed1` | CUB Table-1 DeiT-Ti Ours batch 64 using the fixed build-543 teacher |
| `table1_cub200_deit_ti_ours_b128_full_300ep_seed1` | CUB Table-1 DeiT-Ti Ours batch 128 using the fixed build-543 teacher |
| `table1_cub200_convit_ti_vanilla_b128_full_300ep_seed1` | CUB Table-1 teacher-free ConViT-Ti Vanilla baseline |
| `table1_cub200_convit_ti_lg_b128_full_300ep_seed1` | CUB Table-1 ConViT-Ti LG using the fixed build-543 teacher |
| `table1_cub200_convit_ti_alg_b128_full_300ep_seed1` | CUB Table-1 ConViT-Ti ALG using the fixed build-543 teacher |
| `table1_cub200_convit_ti_ours_b64_full_300ep_seed1` | CUB Table-1 ConViT-Ti Ours batch 64 using the fixed build-543 teacher |
| `table1_cub200_convit_ti_ours_b128_full_300ep_seed1` | CUB Table-1 ConViT-Ti Ours batch 128 using the fixed build-543 teacher |
| `table1_cub200_cvt_13_vanilla_b128_full_300ep_seed1` | CUB Table-1 teacher-free CvT-13 Vanilla baseline |
| `table1_cub200_cvt_13_lg_b128_full_300ep_seed1` | CUB Table-1 CvT-13 LG using the fixed build-543 teacher |
| `table1_cub200_cvt_13_alg_b128_full_300ep_seed1` | CUB Table-1 CvT-13 ALG using the fixed build-543 teacher |
| `table1_cub200_cvt_13_ours_b64_full_300ep_seed1` | CUB Table-1 CvT-13 Ours batch 64 using the fixed build-543 teacher |
| `table1_cub200_cvt_13_ours_b128_full_300ep_seed1` | CUB Table-1 CvT-13 Ours batch 128 using the fixed build-543 teacher |
| `researcher_sync_v1_batch128_ablation_300ep_seed1` | CIFAR-100 Ours v1 with only train batch changed to 128 |
| `cub200_deit_ti_ce_both_imagenet_pretrained_b128_100ep_seed1` | CUB-200 pretrained DeiT-Ti Vanilla control |
| `cub200_deit_ti_lg_resnet50_224_both_imagenet_pretrained_b128_100ep_seed1` | CUB-200 pretrained ResNet50/DeiT-Ti LG |
| `cub200_deit_ti_alg_resnet50_224_both_imagenet_pretrained_b128_100ep_seed1` | CUB-200 pretrained ResNet50/DeiT-Ti ALG |
| `cub200_deit_ti_ours_resnet50_224_both_imagenet_pretrained_b64_100ep_seed1` | CUB-200 pretrained ResNet50/DeiT-Ti Ours, batch 64 |
| `cub200_deit_ti_ours_resnet50_224_both_imagenet_pretrained_b128_100ep_seed1` | CUB-200 pretrained ResNet50/DeiT-Ti Ours batch-128 ablation |

Account names and H200 build numbers are kept only under `run_logs`; they do
not determine checkpoint placement. `PENDING_IMPORTS.md` records jobs that
have started but whose output archives have not yet been verified and added.

Only the selected best checkpoint is committed. The original downloaded
outputs retain `student_latest.pt`; its exact final-epoch accuracy is also
preserved in `run_summary.json`. This avoids doubling repository size and H200
clone time without discarding the reported result. For the four ResNet50-224
Ours runs whose combined source checkpoints exceed GitHub's limit, the exact
student and auxiliary-module states are committed as
`student_best.pt` and `ours_module_best.pt`, respectively. The adjacent
`artifact_manifest.json` records source and committed hashes and the
lossless reassembly rule.

All 81 currently committed student checkpoints were loaded with PyTorch and verified against
their summaries for dataset, method, best accuracy, and checkpoint epoch.
The Top-1 value is read from the adjacent summary; file names are deliberately
stable (`student_best.pt`) inside the provenance-rich protocol directory.

## Consolidated DeiT-Ti reproduction table

| Method | Transfer operator | CIFAR-100 | Flowers-102 | Chaoyang | CUB-200 |
|---|---|---:|---:|---:|---:|
| Vanilla DeiT-Ti | - | 65.08 | 50.06 | 82.00 |  |
| KD | Logits | 69.10 | 48.95 | 62.79 |  |
| CRD | Pooled contrastive | 68.59 | 49.06 | 79.85 |  |
| ReviewKD | Projected fusion | 75.65 | 61.88 | 82.75 |  |
| MGD | Masked reconstruction | 75.68 | 54.66 | 81.81 |  |
| OFA | Logit-space projection | 67.73 | 46.41 | 78.03 |  |
| LG | Direct match (static) |  |  |  | 46.93 |
| ALG | Scheduled match (static) |  | 73.15 | 83.54 | 49.02 |
| **Ours** | **Grid-space, learnable** | **82.90** | **74.81** | **81.95\*** | **48.72** |

Blank cells mean not yet run under the intended method-specific protocol; they
do not mean zero accuracy. Flowers ALG uses train batch 128 (`73.15%`) and
Chaoyang ALG uses train batch 64 (`83.54%`); both checkpoints and summaries
are verified. `*` marks the pending-artifact status for the Chaoyang Ours
result. The Vanilla values are draft references; every other populated cell
is a reproduction result from this project.

Protocol families used in this table:

- generic CIFAR-100: `generic_kd_v2_300ep_seed42`;
- generic Flowers/Chaoyang: `generic_kd_300ep_epoch_only_v1_seed42`;
- Flowers ALG: `paper_lg_v2_trainval_test_b128_300ep_seed1`;
- Chaoyang ALG: `paper_lg_v2_b64_300ep_seed1`;
- Ours: `researcher_sync_v1_300ep_seed1` (Flowers train batch 64);
- CUB-200: the three `cub200_deit_ti_*_300ep_seed1` protocol directories.

## CIFAR-100

Shared setup: ResNet56 teacher at 32 x 32 and scratch DeiT-Ti student at
224 x 224 for 300 epochs. Generic methods use seed 42; the researcher-sync
Ours run uses seed 1.

| Method | Best epoch | Best Top-1 | Last Top-1 | Vanilla gap | Status |
|---|---:|---:|---:|---:|---|
| Vanilla DeiT-Ti | - | 65.08% | - | - | Draft reference |
| KD | 191 | **69.10%** | 68.59% | +4.02 pp | Verified |
| CRD | 79 | **68.59%** | 66.74% | +3.51 pp | Verified |
| Ours (researcher sync) | 288 | **82.90%** | 82.62% | +17.82 pp | Verified; current synchronized run |
| Ours (batch-128 ablation) | 292 | **82.60%** | 82.46% | +17.52 pp | Verified; only train batch changed from 64 to 128 |
| Ours (pre-sync) | 296 | **79.52%** | 79.49% | +14.44 pp | Historical source-grid run |
| ReviewKD | 233 | **75.65%** | 75.50% | +10.57 pp | Verified |
| MGD | 215 | **75.68%** | 75.31% | +10.60 pp | Verified |
| OFA | 263 | **67.73%** | 67.50% | +2.65 pp | Verified |

The researcher-sync Ours result is `82.90%`. The working-paper value recorded
by the repository is `82.42%`, a difference of `+0.48 pp` (the separately
communicated `82.43%` value gives `+0.47 pp`). The older `79.52%` checkpoint
remains in a distinct historical protocol directory and was not overwritten.
The batch-128 ablation is `0.30 pp` below the batch-64 researcher-sync
reference. It keeps the same Ours v1 method, fixed teacher, 300-epoch horizon,
seed, optimizer, controller, and loss, and changes only train batch size.

### Table 4 attribution control

The controls below keep the full-Ours training protocol fixed and change only
the stated spatial or attention intervention.

| Configuration | Best epoch | Best Top-1 | Last Top-1 | Gap to full Ours | Artifact status |
|---|---:|---:|---:|---:|---|
| Full Ours | 288 | **82.90%** | 82.62% | - | Verified |
| Global joint-K/V grid permutation | 298 | **81.79%** | 81.61% | -1.11 pp | Verified |
| Independent global K/V permutations | 298 | **81.00%** | 80.87% | -1.90 pp | Verified |
| Local `2x2` patch-grid permutation | 297 | **82.46%** | 82.41% | -0.44 pp | Verified |
| Token-space Linear Q/K/V | 290 | **83.12%** | 82.88% | +0.22 pp | Verified |

The global joint-K/V result is `+1.99 pp` above the draft token-space row
(`79.80%`), while the direct token-space remeasurement is `83.12%`. The latter
is consistent with the documented mathematical equivalence between a shared
`1x1` convolution and a token-wise linear layer after reshaping. Results are
reported as measured rather than replaced with draft expectations.

Artifacts are stored under the matching protocol directories in
`Ours/cifar100/`: `table4_grid_permuted_*`, `table4_kv_independent_*`,
`table4_local_patch2_*`, and `table4_token_space_*`.

### Table 7 loss-balance controls

All rows below reuse the exact full-Ours CIFAR-100 protocol and change only the
convex feature-loss balance:
`lambda L_fuse + (1-lambda) L_align`.

| Lambda | Feature loss | Best epoch | Best Top-1 | Last Top-1 | Gap to full Ours | Status |
|---:|---|---:|---:|---:|---:|---|
| 0 | `L_align` | 269 | **83.29%** | 83.17% | +0.39 pp | Verified |
| 0.25 | `0.25 L_fuse + 0.75 L_align` | 289 | **83.40%** | 83.24% | +0.50 pp | Verified |
| 0.5 | `0.5 L_fuse + 0.5 L_align` | 288 | **82.90%** | 82.62% | reference | Reused verified full Ours |
| 0.75 | `0.75 L_fuse + 0.25 L_align` | 296 | **82.63%** | 82.55% | -0.27 pp | Verified rerun |
| 1.0 | `L_fuse` | 288 | **82.29%** | 81.87% | -0.61 pp | Verified |
| `(lambda_1, lambda_2)` | independent weights | - | - | - | - | Pair not yet fixed |

The imported sweep checkpoints are stored in the four matching
`table7_lambda_*_researcher_sync_v1_300ep_seed1` protocol directories. The
`lambda=0.5` reference reuses the full-Ours checkpoint.

### Ours V2 relative-position Table 7 control

This separate pair uses the position-aware `relative_position_v1`
architecture for both rows and changes only `lambda`. It must not be mixed
with the pre-V2 sweep above.

| Lambda | Feature loss | Best epoch | Best Top-1 | Last Top-1 | Gap to V2 `lambda=0.5` |
|---:|---|---:|---:|---:|---:|
| 0 | `L_align` | 277 | **83.43%** | 83.42% | +0.59 pp |
| 0.5 | `0.5 L_fuse + 0.5 L_align` | 273 | **82.84%** | 82.55% | reference |

Both runs completed in one H200 sequence and are stored under
`OursV2/cifar100/table7_lambda_{0,0p5}_relative_position_v1_300ep_seed1/`.

## CUB-200-2011 — shared scratch-teacher comparison

All rows use the official train/test split (`5,994`/`5,794`), a scratch
ResNet56 teacher at 32 x 32, a scratch DeiT-Ti student at 224 x 224, FP32,
300 epochs, and seed 1. The selected teacher reached **37.25%** at epoch 283
and all three students reference its exact SHA-256
`e3db747360950e20133ef2698b464ef543edfa521b703dee45e89155d4f92815`.

| Method | Train batch | Best epoch | Best Top-1 | Last Top-1 | Gap to LG | Status |
|---|---:|---:|---:|---:|---:|---|
| LG | 128 | 222 | **46.93%** | 46.34% | reference | Verified |
| ALG | 128 | 251 | **49.02%** | 48.26% | +2.09 pp | Verified |
| Ours | 64 | 263 | **48.72%** | 48.17% | +1.79 pp | Verified |

LG is the direct transfer of official static locality guidance; ALG changes
only the paper controller on that base. Ours retains its researcher-sync
batch-64 protocol and feature module. Because the source LG/ALG work does not
publish a CUB configuration, these rows are reported as controlled protocol
transfers rather than source-paper reproductions.

## CUB-200-2011 — separate Table-1 extension

Build 543 trained the dedicated scratch ResNet56 teacher at 32 x 32 and the
first two DeiT-Ti students. Builds 547, 548, 551, and 552 completed the
remaining DeiT-Ti rows plus all ConViT-Ti and CvT-13 rows. Every student uses
224 x 224 inputs and a 300-epoch schedule; Vanilla is teacher-free and all
guided rows use the fixed build-543 teacher. This family does not reuse or
replace the primary build-509 CUB teacher above.

| Student | Method | Train batch | Best epoch | Best Top-1 | Last Top-1 | Build |
|---|---|---:|---:|---:|---:|---:|
| ResNet56-32 | Teacher | 128 | 275 | **36.40%** | 36.12% | 543 |
| DeiT-Ti | Vanilla | 128 | 156 | **17.69%** | 16.78% | 548 |
| DeiT-Ti | LG | 128 | 241 | **44.51%** | 43.61% | 543 |
| DeiT-Ti | ALG | 128 | 286 | **47.70%** | 47.53% | 543 |
| DeiT-Ti | Ours | 64 | 263 | **48.31%** | 47.15% | 547 |
| DeiT-Ti | Ours | 128 | 277 | **48.36%** | 47.84% | 547 |
| ConViT-Ti | Vanilla | 128 | 261 | **22.94%** | 22.73% | 551 |
| ConViT-Ti | LG | 128 | 263 | **45.55%** | 45.15% | 551 |
| ConViT-Ti | ALG | 128 | 300 | **51.05%** | 51.05% | 551 |
| ConViT-Ti | Ours | 64 | 278 | **53.18%** | 52.90% | 551 |
| ConViT-Ti | Ours | 128 | 295 | **51.73%** | 51.35% | 551 |
| CvT-13 | Vanilla | 128 | 194 | **29.19%** | 27.98% | 552 |
| CvT-13 | LG | 128 | 99 | **43.01%** | 39.30% | 552 |
| CvT-13 | ALG | 128 | 247 | **46.63%** | 45.84% | 552 |
| CvT-13 | Ours | 64 | 248 | **47.98%** | 46.95% | 552 |
| CvT-13 | Ours | 128 | 224 | **46.82%** | 45.36% | 552 |

The fixed teacher is
`teachers/checkpoints/cub200_table1_resnet56_32/teacher_resnet56_cub200_32_best.pt`
with SHA-256
`06f75192b1c108c89e480843cb4f72dfb28aa762d7b11e7ac327333dd54b51f5`.
Every remaining guided Table-1 student must use that exact checkpoint; the
Table-1 training entry point rejects any different manifest or hash.

## CUB-200-2011 — ImageNet-pretrained ResNet50-224 teacher

Build 511 is the historical teacher-only transfer family: the ResNet50 teacher
starts from ImageNet1K-V2 and is fine-tuned for 200 epochs, while every
DeiT-Ti student starts from random initialization and trains for 300 epochs.
It is separate from the newer full-transfer family, which also pretrains its
students and uses a 100-epoch student horizon.

| Method | Train batch | Best epoch | Best Top-1 | Last Top-1 | Status |
|---|---:|---:|---:|---:|---|
| Teacher, ResNet50-224 | 64 | 161 | **84.10%** | 83.79% | Verified checkpoint and manifest |
| LG adaptation | 128 | 263 | **35.19%** | 34.54% | Verified |
| ALG adaptation | 128 | 118 | **29.72%** | 27.58% | Verified |
| Ours | 64 | 93 | **30.65%** | 27.98% | Verified |

The teacher model tensors are exact. Its source checkpoint also contains an
optimizer state and is 191.7 MB, so the committed checkpoint omits only that
retraining state and remains below GitHub's 100 MB single-file limit. The
original full teacher `best` and `latest` files remain in the local
`IBAM_weight` archive. The committed manifest records both hashes.

## CUB-200-2011 — full ImageNet transfer

Build 523 initializes both the ResNet50-224 teacher and all DeiT-Ti students
from ImageNet weights. The teacher trains for 200 epochs and all students for
100 epochs. This is separate from build 511, where only the teacher is
pretrained and the students start from scratch and train for 300 epochs.

| Method | Train batch | Best epoch | Best Top-1 | Last Top-1 | Status |
|---|---:|---:|---:|---:|---|
| Teacher, pretrained ResNet50-224 | 64 | 161 | **84.10%** | 83.79% | Verified; exact model tensors match build 511 |
| Vanilla, pretrained DeiT-Ti | 128 | 100 | **73.06%** | 73.06% | Verified |
| LG, pretrained DeiT-Ti | 128 | 80 | **75.61%** | 75.60% | Verified |
| ALG, pretrained DeiT-Ti | 128 | 100 | **74.16%** | 74.16% | Verified |
| Ours, pretrained DeiT-Ti | 64 | 94 | **73.71%** | 73.37% | Verified |
| Ours, pretrained DeiT-Ti | 128 | 100 | **74.85%** | 74.85% | Verified batch ablation |

The build-523 teacher source SHA-256 is
`80f46b08ea2b2c5398c951268b937f3be0abe47f08bf7617e6dcd4e49a4db82b`.
It differs at the full archive level from build 511, but all 320 model-state
tensors are exactly equal under `torch.equal`. The repository therefore
reuses the verified compact teacher checkpoint and records both source
hashes, manifests, metrics, and summaries instead of committing a duplicate
96 MB model.

## CUB-200-2011 — random-init ResNet50-224 teacher ablation

Build 519 completed all six tasks. The sequence log, five student
checkpoint/summary pairs, and the re-supplied teacher directory were received
and verified.

| Method | Train batch | Best epoch | Best Top-1 | Last Top-1 | Status |
|---|---:|---:|---:|---:|---|
| Teacher, random-init ResNet50-224 | 64 | 139 | **48.31%** | 47.70% | Verified checkpoint, manifest, metrics, and summary |
| Vanilla, LG profile | 128 | 236 | **17.52%** | 16.57% | Verified |
| LG | 128 | 244 | **29.67%** | 29.29% | Verified |
| ALG | 128 | 275 | **26.67%** | 26.20% | Verified |
| Vanilla, Ours profile | 64 | 165 | **16.86%** | 16.10% | Verified |
| Ours | 64 | 103 | **30.17%** | 29.01% | Verified |

The LG/ALG/Ours summaries consistently reference the teacher source SHA-256
`6307a8289f8ddec5c79e8284af8f07d883d037aeaf936062a6833720e4f74ba7`,
which was independently rehashed and matched. The source checkpoint includes
optimizer state and exceeds GitHub's single-file limit; the committed form
retains the exact 320 model tensors and all metadata while omitting only
`optimizer_state`. Its compaction manifest records both hashes.

## Flowers-102 — completed 300-epoch results

The five generic methods below retain their earlier Flowers hyperparameters
(batch 64, warm-up 5, seed 42) and change only the epoch/cosine horizon from
200 to 300. Training uses the official train+val images (`2,040`) and reports
on the official test set (`6,149`).

| Method | Best epoch | Best Top-1 | Last Top-1 | Vanilla gap |
|---|---:|---:|---:|---:|
| Vanilla DeiT-Ti | - | 50.06% | - | - |
| KD | 105 | **48.95%** | 46.69% | -1.11 pp |
| CRD | 172 | **49.06%** | 48.06% | -1.00 pp |
| ReviewKD | 256 | **61.88%** | 61.52% | +11.82 pp |
| MGD | 248 | **54.66%** | 54.09% | +4.60 pp |
| OFA | 201 | **46.41%** | 45.54% | -3.65 pp |

### Selected method-specific Flowers results

Both selected rows use train+val (`2,040`) / test (`6,149`), 300 epochs, seed
1, and a scratch DeiT-Ti at 224 x 224. They intentionally preserve their own
method protocols instead of forcing one method's batch setting onto the other.

| Method | Train / eval batch | Best epoch | Best Top-1 | Last Top-1 | Method-paper reference | Gap | Artifact status |
|---|---:|---:|---:|---:|---:|---:|---|
| ALG (paper/public-LG protocol) | 128 / 200 | 274 | **73.15%** | 72.92% | 69.04% | +4.11 pp | Checkpoint and summary verified |
| Ours (CIFAR-100-matched researcher sync) | 64 / 200 | 251 | **74.81%** | 74.21% | 70.31% | +4.50 pp | Checkpoint and summary verified |

The Ours batch-64 artifacts are committed at
`Ours/flowers102/researcher_sync_v1_300ep_seed1/`. The ALG batch-128 artifacts
are committed at `ALG/flowers102/paper_lg_v2_trainval_test_b128_300ep_seed1/`.
The same sequence produced Ours batch-128 best `72.78%` at epoch 264, last
`72.29%`, and guidance-stop epoch 211; its artifacts are kept at
`Ours/flowers102/paper_source_v2_trainval_test_b128_300ep_seed1/` as an
auxiliary protocol result rather than the selected batch-64 Ours row.

### Auxiliary researcher-sync comparison

The earlier shared researcher-sync batch used train/eval batch `64/200` for
both methods. It remains useful as an ablation, but its ALG row is not the
selected pure ALG-paper result in the consolidated table.

| Method | Best epoch | Best Top-1 | Last Top-1 | Method-paper reference | Gap |
|---|---:|---:|---:|---:|---:|
| ALG (researcher-sync batch 64) | 288 | **75.02%** | 74.87% | 69.04% | +5.98 pp |
| Ours (researcher-sync batch 64) | 251 | **74.81%** | 74.21% | 70.31% | +4.50 pp |

The pure-ALG batch comparison additionally produced Flowers batch-64
`75.05%` (epoch 298, last `74.60%`, guidance stop 185). The later
method-separated batch-128 sequence produced Ours `72.78%` and ALG `73.15%`;
only ALG batch 128 is selected from that sequence because the requested Ours
reporting row is the CIFAR-100-matched batch-64 run. These results must not be
confused with the three-way split audit below.

### Historical official three-way split audit

The `researcher_sync_v2_official_three_way_300ep_seed1_historical` runs used
train `1,020`, validation `1,020`, and test `6,149`. The checkpoint was selected
on validation and the test set was evaluated once.

| Method | Best val epoch | Best val Top-1 | Final test Top-1 |
|---|---:|---:|---:|
| ALG | 269 | 71.57% | **63.57%** |
| Ours | 288 | 70.10% | **61.10%** |

### Historical 200-epoch generic results

Fixed protocol: ResNet56 teacher at 32 x 32, scratch DeiT-Ti student at
224 x 224, 200 epochs, seed 42.

| Method | Best epoch | Best Top-1 | Last Top-1 | Vanilla gap |
|---|---:|---:|---:|---:|
| Vanilla DeiT-Ti | - | 50.06% | - | - |
| KD | 105 | **47.91%** | 46.77% | -2.15 pp |
| CRD | 91 | **49.49%** | 48.20% | -0.57 pp |
| ReviewKD | 149 | **58.89%** | 58.72% | +8.83 pp |
| MGD | 172 | **53.42%** | 53.21% | +3.36 pp |
| OFA | 159 | **46.09%** | 45.55% | -3.97 pp |

## Chaoyang — completed 300-epoch generic results

The five methods retain the earlier Chaoyang recipe (batch 64, warm-up 5,
seed 42) and change only the epoch/cosine horizon from 100 to 300.

| Method | Best epoch | Best Top-1 | Last Top-1 | Vanilla gap |
|---|---:|---:|---:|---:|
| Vanilla DeiT-Ti | - | 82.00% | - | - |
| KD | 49 | **62.79%** | 57.60% | -19.21 pp |
| CRD | 189 | **79.85%** | 78.45% | -2.15 pp |
| ReviewKD | 166 | **82.75%** | 81.25% | +0.75 pp |
| MGD | 155 | **81.81%** | 80.93% | -0.19 pp |
| OFA | 212 | **78.03%** | 75.88% | -3.97 pp |

### Historical 100-epoch generic results

All rows use the fixed ResNet56 teacher at 32 x 32 and a scratch DeiT-Ti
student at 224 x 224. The five generic methods and the historical Ours run use
100 epochs and seed 42. The stored ALG row below is the earlier pre-sync
300-epoch, batch-128 result. Any researcher-synchronized Chaoyang ALG result
must be imported under a new protocol ID rather than overwriting this row.

| Method | Best epoch | Best Top-1 | Last Top-1 | Vanilla gap | Status |
|---|---:|---:|---:|---:|---|
| Vanilla DeiT-Ti | - | 82.00% | - | - | Draft reference |
| ALG | 235 | **80.32%** | 79.71% | -1.68 pp | Verified pre-sync batch-128 run |
| Ours | 82 | **81.21%** | 80.46% | -0.79 pp | Historical 100-epoch paper-grid run |
| KD | 15 | **62.79%** | 56.80% | -19.21 pp | Verified |
| CRD | 61 | **79.66%** | 77.93% | -2.34 pp | Verified |
| ReviewKD | 86 | **81.72%** | 81.07% | -0.28 pp | Verified |
| MGD | 80 | **80.69%** | 79.94% | -1.31 pp | Verified |
| OFA | 90 | **75.55%** | 74.99% | -6.45 pp | Verified |

The stored Ours result is the earlier 100-epoch, seed-42 run with teacher-grid
targets `32 x 32`, `16 x 16`, and `8 x 8`. Its checkpoint and summary are
explicitly named `historical` so they cannot be confused with the pending
300-epoch matched ALG-base reruns.

### Pure-ALG batch comparison and auxiliary Ours run

Build 480 completed four 300-epoch, seed-1 runs and preserved the full log,
sequence summary, and individual checkpoints/per-run summaries.

| Method | Train batch | Best epoch | Best Top-1 | Last Top-1 | Guidance stop |
|---|---:|---:|---:|---:|---:|
| ALG | 128 | 281 | **80.97%** | 80.46% | 213 |
| ALG | 64 | 292 | **83.54%** | 82.84% | 183 |
| Ours (CIFAR-100-locked protocol) | 64 | 271 | **81.11%** | 80.22% | 192 |

The ALG batch-64 result is only `+0.04 pp` from the working-paper `83.50%`
value and is the intended ALG Chaoyang reproduction shown in the consolidated
table. The Ours `81.11%` row is an auxiliary CIFAR-100-locked run and does not
replace the separate pending `81.95%` researcher-sync Ours result. The three
ALG artifacts are under `ALG/flowers102/paper_lg_v2_trainval_test_b64_300ep_seed1/`,
`ALG/chaoyang/paper_lg_v2_b128_300ep_seed1/`, and
`ALG/chaoyang/paper_lg_v2_b64_300ep_seed1/`; the Ours artifact is under
`Ours/chaoyang/cifar100_locked_b64_v1_300ep_seed1/`.

## Import status

The 300-epoch generic Flowers/Chaoyang batch and the researcher-sync Ours/ALG
batch have been imported and verified. Historical 200/100-epoch results remain
in separate protocol directories; no old checkpoint was overwritten.

## Source runs

- `run_logs/h200_build-450_combined-generic-kd.log`: Chaoyang five methods,
  Flowers-102 five methods, then CIFAR-100 KD.
- `run_logs/h200_build-452_cifar100-ours-crd.log`: CIFAR-100 Ours and CRD.
- `run_logs/h200_build-453_cifar100-reviewkd-mgd.log`: CIFAR-100 ReviewKD
  and MGD.
- `run_logs/h200_build-454_cifar100-ofa.log`: CIFAR-100 OFA.
- `run_logs/h200_build-457_chaoyang-ours-papergrid-100ep.log`: historical
  Chaoyang Ours paper-grid run (100 epochs, seed 42).
- `run_logs/h200_build-461_chaoyang-alg-public-base-300ep.log`: Chaoyang ALG
  run on the audited public LG/ALG base (300 epochs, seed 1).
- `run_logs/h200_build-471_generic-kd-flowers-chaoyang-300ep.log`: the ten
  completed 300-epoch generic Flowers/Chaoyang runs.
- `run_logs/h200_build-475_researcher-sync-ours-alg-300ep.log`: Ours
  CIFAR-100, Ours Flowers-102, and ALG Flowers-102 researcher-sync runs.
- `run_logs/h200_build-477_flowers-official-three-way-ours-alg-300ep.log`:
  historical Flowers train/val/test audit.
- `run_logs/h200_build-479_flowers-alg-ours-protocol-separated-300ep.log`:
  pure ALG batch-128 and Ours batch-128 Flowers sequence; adjacent sequence
  JSON records task commands and best values.
- `run_logs/h200_build-480_alg-batch-comparison-ours-chaoyang-300ep.log`:
  ALG Flowers batch 64, ALG Chaoyang batches 128/64, and Ours Chaoyang batch
  64; adjacent sequence JSON records the four completed tasks.
- `run_logs/h200_build-509_cub200-shared-teacher-lg-alg-ours-300ep/`:
  shared scratch ResNet56 teacher followed by LG, ALG, and Ours CUB-200
  students; the directory preserves the combined log and sequence status.
- `run_logs/h200_build-511_cub200-resnet50-224-imagenet1k-v2-teacher-lg-alg-ours-300ep/`:
  ImageNet1K-V2 ResNet50-224 teacher followed by scratch LG, ALG, and Ours
  students.
- `run_logs/h200_build-519_cub200-resnet50-224-scratch-teacher-vanilla-lg-alg-ours-300ep/`:
  random-init ResNet50-224 teacher, two profile-matched Vanilla controls, LG,
  ALG, and Ours; all six tasks are log-complete.
- `run_logs/h200_build-522_cifar100-ours-v1-batch128-ablation-300ep.log`:
  Ours v1 CIFAR-100 researcher-sync batch-128 ablation (`82.60%`).
- `run_logs/h200_build-523_cub200-full-transfer-pretrained-teacher-students-100ep/`:
  pretrained ResNet50 teacher followed by pretrained Vanilla, LG, ALG, and
  Ours batches 64/128; all six tasks completed.
- `run_logs/h200_build-543_table1-cub200-resnet56-32-deit-ti-lg-alg-300ep/`:
  separate Table-1 ResNet56-32 teacher followed by DeiT-Ti LG and ALG; all
  three tasks completed and the teacher is fixed for future Table-1 students.
- `run_logs/h200_build-547_table1-cub200-deit-ti-ours-b64-b128-300ep/`:
  DeiT-Ti Ours batches 64/128; both tasks completed.
- `run_logs/h200_build-548_table1-cub200-deit-ti-vanilla-b128-300ep/`:
  teacher-free DeiT-Ti Vanilla batch-128 run.
- `run_logs/h200_build-551_table1-cub200-convit-ti-all-five-300ep/` and
  `run_logs/h200_build-552_table1-cub200-cvt13-all-five-300ep/`: complete
  Vanilla/LG/ALG/Ours64/Ours128 sequences for ConViT-Ti and CvT-13.
- `run_logs/h200_build-482_table4-grid-permutation-cifar100-300ep.log`:
  Table 4 grid-permutation control (`81.79%`).
- `run_logs/h200_build-484_table7-lambda0-cifar100-300ep.log`:
  Table 7 `lambda=0` control (`83.29%`).
- `run_logs/h200_build-485_table7-lambda0p25-cifar100-300ep.log`:
  Table 7 `lambda=0.25` control (`83.40%`).
- `run_logs/flowers102_alg128_ours128_300ep_final_excerpt.txt`: supplied final
  lines from the method-separated Flowers sequence (ALG batch 128 `73.15%`,
  Ours batch 128 `72.78%`); superseded for auditing by the complete build-479
  log but retained as the originally supplied excerpt.

Generic methods use the CNN-to-ViT adapters documented in each method
directory. They should not be described as unmodified original CNN-to-CNN
experiments. Ours is maintained separately from the five generic baselines.
