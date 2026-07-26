# CUB-200 paired both-scratch 300-epoch student family

This is the student-horizon-only follow-up to
`cub200_resnet50_deit_ti_224_both_scratch_100ep`.

## Locked protocol

| Component | Initialization | Input | Training |
|---|---|---:|---|
| Teacher | random-init ResNet50 | 224 x 224 | CUB training, 200 epochs |
| Vanilla-b128 | random-init DeiT-Ti | 224 x 224 | CE only, 300 epochs, batch 128 |
| LG | random-init DeiT-Ti | 224 x 224 | official LG mechanics, 300 epochs, batch 128 |
| ALG | random-init DeiT-Ti | 224 x 224 | ALG paper controller on official LG, 300 epochs, batch 128 |
| Ours-b64 | random-init DeiT-Ti | 224 x 224 | unchanged Ours mechanics, 300 epochs, batch 64 |
| Ours-b128 | random-init DeiT-Ti | 224 x 224 | Ours batch-only ablation, 300 epochs, batch 128 |

The teacher remains at 200 epochs deliberately. This preserves the exact
teacher used by the 100-epoch family and isolates the effect of extending
student training. Teacher and student epoch counts do not need to match
because the completed teacher is frozen during every student run.

Architecture, CUB split, 224 x 224 input, data recipe, optimizer, scheduler
recipe, seed, method mechanics, sequence, and batch sizes otherwise match the
100-epoch paired scratch experiment.

This is not the historical `cub200_resnet50_224_scratch` 300-epoch runner.
That sequence included Vanilla-b64 instead of Ours-b128. This runner retains:

```text
Teacher200 -> VanillaB128-300 -> LG-300 -> ALG-300 -> OursB64-300 -> OursB128-300
```

## Timing run

```text
python methods/run_cub200_both_scratch_300ep_all.py --timing-run --num-workers 4 --output-dir /app/output/cub200_both_scratch_300ep_timing_seed1
```

## Full run

```text
python methods/run_cub200_both_scratch_300ep_all.py --full-run --num-workers 4 --output-dir /app/output/cub200_both_scratch_300ep_full_seed1
```

The final log line reports all six Best Top-1 values:

```text
[FINAL_TOP1_SUMMARY_224_BOTH_SCRATCH_300EP] Teacher=...% VanillaB128=...% LG=...% ALG=...% OursB64=...% OursB128=...%
```
