# H200 outputs awaiting artifact import

This file is an inbox, not a result table. A row means that an H200 job was
submitted or completed but its output archive has not yet been received and
verified in this repository. Accuracy cells intentionally remain blank.

## Current status

The overnight batch received on 2026-07-22 and builds 479, 480, 482, 484,
485, 488, 490, 493, 494, 496, 503, 509, 511, 519, 522, 523, 543, 547, 548,
and 551–556 received through 2026-07-29 have been imported where artifacts
were present. One
completed result still awaits checkpoint artifacts:

| Method | Dataset | Protocol ID | Log-verified result | Expected destination | Missing artifacts |
|---|---|---|---:|---|---|
| Ours | Chaoyang | `researcher_sync_v1_300ep_seed1` | 81.95% | `results/Ours/chaoyang/researcher_sync_v1_300ep_seed1/` | `student_best.pt`, `run_summary.json` |

The completed Chaoyang H200 log reports best epoch 292, last Top-1 81.11%,
guidance stop epoch 193, and selected best Top-1 81.95%. This value may be
shown with a pending-artifact marker, but it is not counted among the
committed and PyTorch-verified checkpoints.

The following received families were loaded with PyTorch, checked against
their JSON summaries, and imported without replacing historical results:

- `generic_kd_300ep_epoch_only_v1_seed42`: five Flowers-102 and five
  Chaoyang generic-KD runs;
- `researcher_sync_v1_300ep_seed1`: Ours CIFAR-100, Ours Flowers-102, and
  ALG Flowers-102;
- `researcher_sync_v2_official_three_way_300ep_seed1_historical`: the
  Flowers train/validation/test audit retained for provenance.
- `table4_grid_permuted_researcher_sync_v1_300ep_seed1_permseed1`: Table 4
  grid-permutation best checkpoint and summary;
- `table7_lambda_0_researcher_sync_v1_300ep_seed1`: Table 7 lambda-zero best
  checkpoint and summary;
- `table7_lambda_0p25_researcher_sync_v1_300ep_seed1`: Table 7 lambda-0.25
  best checkpoint and summary.
- `table7_lambda_0p75_researcher_sync_v1_300ep_seed1` and
  `table7_lambda_1_researcher_sync_v1_300ep_seed1`: the remaining received
  Table 7 convex-sweep checkpoints and summaries;
- `table4_kv_independent_researcher_sync_v1_300ep_seed1_k1_v1001`,
  `table4_local_patch2_researcher_sync_v1_300ep_seed1_permseed1`, and
  `table4_token_space_researcher_sync_v1_300ep_seed1`: the three received
  Table 4 follow-up controls;
- `table7_lambda_0_relative_position_v1_300ep_seed1` and
  `table7_lambda_0p5_relative_position_v1_300ep_seed1`: the paired Ours V2
  relative-position Table 7 runs from build 503;
- `paper_lg_v2_trainval_test_b128_300ep_seed1`: selected ALG Flowers batch-128
  result from build 479;
- `paper_source_v2_trainval_test_b128_300ep_seed1`: auxiliary Ours Flowers
  batch-128 result from build 479;
- `paper_lg_v2_trainval_test_b64_300ep_seed1`,
  `paper_lg_v2_b128_300ep_seed1`, and `paper_lg_v2_b64_300ep_seed1`: pure-ALG
  batch controls from build 480;
- `cifar100_locked_b64_v1_300ep_seed1`: auxiliary Ours Chaoyang batch-64
  result from build 480.
- `cub200_deit_ti_official_lg_v1_300ep_seed1`,
  `cub200_deit_ti_alg_paper_official_lg_v1_300ep_seed1`, and
  `cub200_deit_ti_ours_scratch_teacher_v1_300ep_seed1`: the shared-teacher
  CUB-200 LG/ALG/Ours sequence from build 509.
- `cub200_deit_ti_lg_resnet50_224_transfer_adaptation_v1_300ep_seed1`,
  `cub200_deit_ti_alg_resnet50_224_transfer_adaptation_v1_300ep_seed1`, and
  `cub200_deit_ti_ours_resnet50_224_transfer_v1_300ep_seed1`: the build-511
  students using the verified ImageNet1K-V2 ResNet50-224 teacher; that teacher
  is stored under `teachers/checkpoints/cub200_resnet50_224_imagenet1k_v2/`;
- `cub200_deit_ti_ce_lg_official_b128_300ep_seed1`,
  `cub200_deit_ti_lg_resnet50_224_scratch_teacher_ablation_v1_300ep_seed1`,
  `cub200_deit_ti_alg_resnet50_224_scratch_teacher_ablation_v1_300ep_seed1`,
  `cub200_deit_ti_ce_ours_current_b64_300ep_seed1`, and
  `cub200_deit_ti_ours_resnet50_224_scratch_teacher_ablation_v1_300ep_seed1`:
  the five verified build-519 students. Their summaries preserve the scratch
  teacher's source SHA-256. The re-supplied teacher checkpoint,
  manifest, metrics, and summary are now verified under
  `teachers/checkpoints/cub200_resnet50_224_scratch/`.
- `researcher_sync_v1_batch128_ablation_300ep_seed1`: the verified Ours v1
  CIFAR-100 train-batch-128 control from build 522;
- `cub200_deit_ti_ce_both_imagenet_pretrained_b128_100ep_seed1`,
  `cub200_deit_ti_lg_resnet50_224_both_imagenet_pretrained_b128_100ep_seed1`,
  `cub200_deit_ti_alg_resnet50_224_both_imagenet_pretrained_b128_100ep_seed1`,
  and the two
  `cub200_deit_ti_ours_resnet50_224_both_imagenet_pretrained_b{64,128}_100ep_seed1`
  protocols: the five verified pretrained students from build 523. The
  pretrained teacher is tensor-identical to the existing compact build-511
  checkpoint, and its second source hash is recorded in the teacher manifest.
- `table1_cub200_deit_ti_lg_b128_full_300ep_seed1` and
  `table1_cub200_deit_ti_alg_b128_full_300ep_seed1`: the verified build-543
  CUB Table-1 DeiT-Ti LG/ALG runs. Their exact ResNet56-32 teacher is fixed
  under `teachers/checkpoints/cub200_table1_resnet56_32/` for all future
  Table-1 students.
- `table1_cub200_deit_ti_ours_b{64,128}_full_300ep_seed1` and
  `table1_cub200_deit_ti_vanilla_b128_full_300ep_seed1`: the verified
  builds 547/548 rows that complete the five-setting DeiT-Ti Table-1 group;
- all five `table1_cub200_convit_ti_*_full_300ep_seed1` protocols from build
  551 and all five `table1_cub200_cvt_13_*_full_300ep_seed1` protocols from
  build 552. Guided summaries all reference the fixed build-543 teacher;
  teacher-free Vanilla rows carry no teacher metadata. The reattached build
  503 pair and four CUB archives were byte-identical to the already organized
  local copies and were not duplicated.
- all five protocols for each of PiT-Ti, PVTv2-B0, T2T-ViT-7, and
  T2T-ViT-14 from builds 553–556. The 20 best/latest checkpoint pairs,
  summaries, final summaries, sequence statuses, and logs were verified.
  Guided summaries all reference the fixed build-543 teacher. These runs
  complete the original 36-task CUB Table-1 baseline matrix. The four repeated
  CUB archives were byte-identical to the retained local dataset and were not
  duplicated.

Add new jobs below this section only after submission, and remove their rows
after the same import gate has passed.

## Import gate

Before moving a pending output into `results/`:

1. load `student_best.pt` with PyTorch;
2. compare method, dataset, epoch, Top-1, seed, and protocol arguments with
   `summary.json`;
3. verify the teacher SHA-256 against `teachers/checkpoints/manifest.json`;
4. place it only in its exact protocol-ID directory;
5. update the result table and `CHECKSUMS.sha256`;
6. change or remove the corresponding pending row only after verification.
