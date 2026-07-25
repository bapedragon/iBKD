# CUB-200 paired both-scratch 100-epoch family

This family is the initialization-only control for
`cub200_resnet50_deit_ti_224_both_imagenet_pretrained`.

## Locked protocol

| Component | Initialization | Input | Training |
|---|---|---:|---|
| Teacher | random-init ResNet50 | 224 x 224 | CUB training, 200 epochs |
| Vanilla-b128 | random-init DeiT-Ti | 224 x 224 | CE only, 100 epochs, batch 128 |
| LG | random-init DeiT-Ti | 224 x 224 | official LG mechanics, 100 epochs, batch 128 |
| ALG | random-init DeiT-Ti | 224 x 224 | ALG paper controller on official LG, 100 epochs, batch 128 |
| Ours-b64 | random-init DeiT-Ti | 224 x 224 | unchanged Ours mechanics, 100 epochs, batch 64 |
| Ours-b128 | random-init DeiT-Ti | 224 x 224 | Ours batch-only ablation, 100 epochs, batch 128 |

Architecture, input resolution, CUB split, data recipe, optimizer, scheduler,
seed, method mechanics, epoch counts, sequence, and batch sizes match the
fully pretrained 100-epoch experiment. The only controlled change is that
ImageNet initialization is removed from both the teacher and every student.

This is not the historical `cub200_resnet50_224_scratch` experiment, whose
student horizon was 300 epochs and whose sequence included Vanilla-b64
instead of Ours-b128.

The exact order is:

```text
Teacher -> VanillaB128 -> LG -> ALG -> OursB64 -> OursB128
```

## Timing run

```text
python methods/run_cub200_both_scratch_100ep_all.py --timing-run --num-workers 4 --output-dir /app/output/cub200_both_scratch_100ep_timing_seed1
```

## Full run

```text
python methods/run_cub200_both_scratch_100ep_all.py --full-run --num-workers 4 --output-dir /app/output/cub200_both_scratch_100ep_full_seed1
```

The final log line reports all six Best Top-1 values:

```text
[FINAL_TOP1_SUMMARY_224_BOTH_SCRATCH_100EP] Teacher=...% VanillaB128=...% LG=...% ALG=...% OursB64=...% OursB128=...%
```
