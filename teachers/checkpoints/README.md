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
| ImageNet1K-V2 teacher / scratch students | `cub200_resnet50_224_imagenet1k_v2/teacher_resnet50_cub200_224_best.pt` | 511 | 161 | **84.10%** | 83.79% | Verified |
| Random-init ResNet50 teacher ablation | `cub200_resnet50_224_scratch/` | 519 | 139 | **48.31%** | 47.70% | Log verified; teacher files not supplied |

The build-511 source checkpoint is 191.7 MB because it includes an optimizer
state. The committed form retains all exact model tensors and metadata, omits
only `optimizer_state`, and records both the source and committed SHA-256 in
`artifact_manifest.json`. The complete source `best` and `latest` files remain
in the local archive.

The build-519 full log and five student archives were supplied, but its
teacher `best`, `latest`, `manifest`, `metrics`, and `summary` files were not.
That teacher remains listed in `results/PENDING_IMPORTS.md`.

Verify hashes, metadata, strict state-dict loading, and a 32 x 32 forward pass:

```bash
python teachers/verify_checkpoints.py --dataset all
```

Verify the separate build-511 ResNet50-224 checkpoint with:

```bash
python teachers/verify_checkpoints.py \
  --checkpoint-root teachers/checkpoints/cub200_resnet50_224_imagenet1k_v2 \
  --dataset cub200
```

The datasets are not included. Chaoyang remains mounted separately at
`/app/data/chaoyang`.
