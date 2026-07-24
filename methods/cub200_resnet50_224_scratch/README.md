# CUB-200 ResNet50-224 scratch-teacher ablation

This is a third, isolated CUB-200 family. It must not overwrite either:

- `cub200_resnet56_32_scratch`
- `cub200_common_transfer_resnet50_224`

The controlled comparison against the existing ResNet50-224 transfer family
changes only the teacher initialization:

| field | existing transfer family | this scratch ablation |
|---|---|---|
| teacher | ResNet50, ImageNet1K-V2 | ResNet50, random initialization |
| teacher input | 224 x 224 | 224 x 224 |
| teacher recipe | SGD, LR 0.01, 200 epochs | same |
| student | DeiT-Ti, random initialization | same |
| student input | 224 x 224 | same |
| split | official CUB train/test | same |

## Why the scratch teacher still uses 200 epochs

The 200-epoch teacher schedule is not claimed to be an official CUB standard
or an optimal from-scratch ResNet50 recipe. It is retained from the existing
ResNet50-224 transfer family so that the controlled ablation changes only one
factor: `ImageNet1K-V2` initialization versus random initialization. The
architecture, resolution, data split, augmentation, optimizer, learning rate,
and epoch count remain fixed.

If the scratch teacher underfits or has not converged at epoch 200, a
scratch-optimized teacher schedule can be studied later. Such a run must be
labeled as a separate protocol because changing both initialization and the
training schedule would no longer isolate the pretraining effect.

The sequence contains six tasks:

1. scratch ResNet50-224 teacher
2. Vanilla DeiT-Ti, official-LG profile, batch 128
3. LG, batch 128
4. ALG, batch 128
5. Vanilla DeiT-Ti, current-Ours profile, batch 64
6. Ours, batch 64

Vanilla is teacher-free and uses CE only. Two Vanilla profiles are required
because official LG/ALG and current Ours use different optimizer grouping,
classifier initialization, batch size, and evaluation interpolation.

The timing run performs two real epochs for every task. It validates the
scratch manifest, feature shapes, forward/backward passes, memory use, and the
estimated sum against the 600-minute pod limit. Timing accuracy is not a
convergence result, and a two-epoch teacher must never be reused for a full
student run.

The final stdout line is:

```text
[FINAL_TOP1_SUMMARY_224_SCRATCH] TeacherScratch224=...% VanillaB128=...% LG=...% ALG=...% VanillaB64=...% Ours=...%
```

Copyable H200 forms:

- timing: `H200_TIMING_ISSUE.md`
- full training: `H200_FULL_ISSUE.md`
