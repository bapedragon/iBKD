# CUB-200 full ImageNet-transfer family

This family is the repository's conventional pretrained CUB-200 comparison.
It is isolated from every scratch or teacher-only-transfer CUB result.

## Locked protocol

| Component | Initialization | Input | Training |
|---|---|---:|---|
| Teacher | TorchVision ResNet50 `IMAGENET1K_V2` | 224 x 224 | CUB fine-tuning, 200 epochs |
| Vanilla-b128 | timm DeiT-Ti `fb_in1k` | 224 x 224 | CE only, 100 epochs, batch 128 |
| LG | timm DeiT-Ti `fb_in1k` | 224 x 224 | official LG mechanics, 100 epochs, batch 128 |
| ALG | timm DeiT-Ti `fb_in1k` | 224 x 224 | ALG paper controller on official LG, 100 epochs, batch 128 |
| Ours-b64 | timm DeiT-Ti `fb_in1k` | 224 x 224 | unchanged Ours mechanics, 100 epochs, batch 64 |
| Ours-b128 | timm DeiT-Ti `fb_in1k` | 224 x 224 | Ours batch-only ablation, 100 epochs, batch 128 |

The exact student weight identifier is
`timm/deit_tiny_patch16_224.fb_in1k`. Pretraining applies to the DeiT
backbone. The CUB 200-class classifier is necessarily newly initialized:
Vanilla/LG/ALG retain the public LG zero-initialized head, while Ours retains
its existing timm head initialization. No feature loss, optimizer grouping,
controller, augmentation, or scheduler is changed merely to match Ours.
Only the planned student horizon is changed from the historical LG scratch
schedule to 100 epochs, reflecting common pretrained CUB fine-tuning practice.

The sequential order is:

```text
Teacher -> VanillaB128 -> LG -> ALG -> OursB64 -> OursB128
```

Vanilla is intentionally run only at batch 128. Ours is the only method run
at both batch 64 and 128 because batch 64 is the unchanged repository default
and batch 128 is the requested batch-only comparison.

## Separation from previous CUB experiments

| Family | Teacher | Students |
|---|---|---|
| `cub200_resnet56_32_scratch` | ResNet56 scratch, 32 px | DeiT-Ti scratch, 224 px |
| `cub200_common_transfer_resnet50_224` | ResNet50 ImageNet-pretrained, 224 px | DeiT-Ti scratch, 224 px |
| `cub200_resnet50_224_scratch` | ResNet50 scratch, 224 px | DeiT-Ti scratch, 224 px |
| `cub200_resnet50_deit_ti_224_both_imagenet_pretrained` | ResNet50 ImageNet-pretrained, 224 px | DeiT-Ti ImageNet-pretrained, 224 px |

The runner validates the teacher manifest and every student summary before it
continues. It fails if either side is scratch, if an input is not 224, or if
a batch size differs from the locked sequence.

## Timing run

```text
python methods/run_cub200_full_transfer_all.py --timing-run --num-workers 4 --output-dir /app/output/cub200_full_transfer_all_100ep_timing_seed1
```

The timing run performs two real epochs for all six tasks. It verifies weight
downloads, feature shapes, forward/backward execution, memory use, and the
estimated combined runtime. It does not measure convergence, and its teacher
checkpoint must not be reused for the full run.

The earlier 300-epoch timing run remains a conservative validity check for
this epoch-only reduction: the first two epochs, model shapes, batches, and
memory path are unchanged, while the planned full workload only decreases.

## Full run

```text
python methods/run_cub200_full_transfer_all.py --full-run --num-workers 4 --output-dir /app/output/cub200_full_transfer_all_100ep_full_seed1
```

The final log line reports all six Best Top-1 values:

```text
[FINAL_TOP1_SUMMARY_224_FULL_TRANSFER] Teacher=...% VanillaB128=...% LG=...% ALG=...% OursB64=...% OursB128=...%
```

The official CUB page warns that CUB images may overlap ImageNet. These runs
must therefore be reported explicitly as ImageNet-pretrained transfer
experiments, separately from all scratch families.
