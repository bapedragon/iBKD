# Fixed 32 x 32 teacher checkpoints

These are the primary ResNet56 teacher weights selected after the full H200
runs. The same checkpoint must be reused for every compared KD method on the
corresponding dataset.

| Dataset | Selected checkpoint | H200 build | Epoch | Top-1 | Draft reference | Gap |
|---|---|---:|---:|---:|---:|---:|
| CIFAR-100 | `cifar100/teacher_resnet56_cifar100_32_best.pt` | 438 | 300 | 71.91% | 70.43% | +1.48 pp |
| Flowers-102 | `flowers102/teacher_resnet56_flowers102_32_best.pt` | 447 | 389 | 66.03% | 66.33% | -0.30 pp |
| Chaoyang | `chaoyang/teacher_resnet56_chaoyang_32_best.pt` | 443 | 94 | 76.72% | 77.20% | -0.48 pp |
| CUB-200-2011 | `cub200/teacher_resnet56_cub200_32_best.pt` | 509 | 283 | 37.25% | - | - |

Every dataset directory also contains:

- `training_config.json`: locked settings written before training;
- `metrics.csv`: epoch-by-epoch measurements;
- `training_summary.json`: final result, paths, timing, protocol, and hashes;
- `training_log.txt`: full H200 issue output.

Only the primary `best` checkpoint is committed. The locally archived H200
output still retains `best`, `latest`, and `closest_to_reference` files.
For CUB, the generated `run_manifest.json`, metrics, and training summary are
stored with the checkpoint; its combined teacher/LG/ALG/Ours log and sequence
status are under `results/run_logs/h200_build-509_*`.

## Separate CUB ResNet50-224 teacher artifacts

These alternate teachers belong to isolated 224 x 224 protocol families and
never replace the primary scratch ResNet56 checkpoint above.

| Family | Selected checkpoint | H200 build | Best epoch | Best Top-1 | Last Top-1 | Status |
|---|---|---:|---:|---:|---:|---|
| ImageNet1K-V2 ResNet50 teacher | `cub200_resnet50_224_imagenet1k_v2/teacher_resnet50_cub200_224_best.pt` | 511, 523 | 161 | **84.10%** | 83.79% | Verified; source model tensors equal |
| Random-init ResNet50 teacher ablation | `cub200_resnet50_224_scratch/` | 519 | 139 | **48.31%** | 47.70% | Verified; compact exact-model form |

The build-511 and build-523 source checkpoints are each about 191.7 MB because
they include optimizer state. Their full archive hashes differ, but all 320
model-state tensors are exactly equal under `torch.equal`. The committed form
retains the exact model tensors and metadata, omits only `optimizer_state`,
and records both source hashes plus the committed SHA-256 in
`artifact_manifest.json`. Both source `best`/`latest` pairs remain in the
local archive. The build-523 source manifest, metrics, and summary are kept
with a `build523` suffix.

The re-supplied build-519 archive includes its teacher `best`, `latest`,
manifest, 200-row metrics, and summary. The source best checkpoint is about
191.7 MB because it contains optimizer state. The committed form retains all
320 exact model tensors and metadata, omits only `optimizer_state`, and
records source/committed hashes in `artifact_manifest.json`.

## Separate CUB Table-1 teacher

The CUB Table-1 extension has its own fixed teacher and never reuses the
primary build-509 CUB teacher:

| Family | Selected checkpoint | H200 build | Best epoch | Best Top-1 | SHA-256 |
|---|---|---:|---:|---:|---|
| CUB Table 1 | `cub200_table1_resnet56_32/teacher_resnet56_cub200_32_best.pt` | 543 | 275 | **36.40%** | `06f75192b1c108c89e480843cb4f72dfb28aa762d7b11e7ac327333dd54b51f5` |

All current and future guided runs under `methods/table1_cub200/` validate
this exact identity before training.

Verify hashes, metadata, strict state-dict loading, and a 32 x 32 forward pass:

```bash
python teachers/verify_checkpoints.py --dataset all
```

Verify the shared build-511/build-523 ResNet50-224 checkpoint with:

```bash
python teachers/verify_checkpoints.py \
  --checkpoint-root teachers/checkpoints/cub200_resnet50_224_imagenet1k_v2 \
  --dataset cub200
```

The datasets are not included. Chaoyang remains mounted separately at
`/app/data/chaoyang`.
