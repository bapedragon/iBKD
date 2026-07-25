# CUB-200 ResNet50-224 teacher-transfer family

This directory documents a separate CUB-200 experiment family. It never
reuses or overwrites the existing `cub200_resnet56_32_scratch` teacher.

There is no single official CUB training recipe. This repository calls the
following a **common transfer-learning protocol**, reflecting the widely used
ResNet50 + ImageNet pretraining + 224-pixel setup:

- dataset: CUB-200-2011 official train/test split (`5,994 / 5,794`)
- teacher: TorchVision ResNet50 initialized with `IMAGENET1K_V2`
- teacher input: **224 x 224**
- teacher train transform: RandomResizedCrop(224), horizontal flip, ImageNet
  normalization
- teacher evaluation: Resize(256), CenterCrop(224), ImageNet normalization
- optimization: full fine-tuning, SGD, LR `0.01`, momentum `0.9`, Nesterov,
  weight decay `5e-4`, cosine decay, batch `64`, 200 epochs, seed `1`, FP32
- students: **random-initialized** DeiT-Ti at **224 x 224**, 300 epochs,
  seed `1`

This historical family pretrains only the teacher. It is not the conventional
full-transfer comparison in which both teacher and student use ImageNet
initialization. That separate experiment is documented in
[`../cub200_full_transfer/`](../cub200_full_transfer/).

The CUB project page warns that the dataset contains images that overlap
ImageNet. Therefore this pretrained family is not a clean scratch comparison
and must be reported separately from the ResNet56-32 family.

## Teacher feature contract

The ResNet50 teacher exposes its last three stages:

| Stage | Channels | Grid at 224 input |
|---|---:|---:|
| `layer2` | 512 | 28 x 28 |
| `layer3` | 1024 | 14 x 14 |
| `layer4` | 2048 | 7 x 7 |

LG uses the public LG feature-loss mechanics adapted to these ResNet50 stages.
ALG uses the paper controller on that same adapted LG base; ALG has no public
official implementation. Ours retains its current method/controller and only
consumes the new teacher feature contract. Consequently, LG/ALG results in
this family are protocol adaptations, not claims of an official ResNet50-CUB
configuration.

## Separation guarantees

| Property | Existing scratch family | This transfer family |
|---|---|---|
| protocol ID | `cub200_resnet56_32_scratch` | `cub200_common_transfer_resnet50_224` |
| teacher | ResNet56, random initialization | ResNet50, ImageNet1K-V2 |
| teacher input | 32 x 32 | **224 x 224** |
| runner | `run_cub200_lg_alg_ours.py` | `run_cub200_resnet50_224_lg_alg_ours.py` |
| output names | `cub200_lg_alg_ours_*` | `cub200_resnet50_224_lg_alg_ours_*` |

Every 224 student loads the exact manifest produced by the 224 teacher run.
The manifest records the model name, input resolution, pretrained source,
feature stages, channels, grids, checksum, epoch, and Top-1.

## Required timing run

Run this before a full request:

```text
python methods/run_cub200_resnet50_224_lg_alg_ours.py --timing-run --num-workers 4 --output-dir /app/output/cub200_resnet50_224_lg_alg_ours_timing_seed1
```

The timing run performs two real epochs for the teacher and each student. It
downloads/loads the actual pretrained ResNet50, checks the manifest, verifies
feature and logits shapes, executes forward/backward optimization, exposes
CUDA OOM or tensor-shape failures, and estimates the combined H200 runtime.
It cannot establish final accuracy or convergence.

Successful completion ends with:

```text
[SEQUENCE_DONE_224] completed_tasks=4/4
[POD_LIMIT_CHECK_224] status=PASS ...
[FINAL_TOP1_SUMMARY_224] Teacher224=...% LG224=...% ALG224=...% Ours224=...%
```

## Full run

Submit only after the timing run passes:

```text
python methods/run_cub200_resnet50_224_lg_alg_ours.py --full-run --num-workers 4 --output-dir /app/output/cub200_resnet50_224_lg_alg_ours_full_seed1
```

## References

- [Official CUB-200-2011 dataset page and ImageNet-overlap warning](https://www.vision.caltech.edu/datasets/cub_200_2011/)
- [TorchVision pretrained model and ResNet50 weight documentation](https://docs.pytorch.org/vision/2.0/models.html)
- [Example CUB fine-grained classifier built from pretrained ResNet50](https://github.com/cyizhuo/Fine-Grained-Image-Classification)
