# Teacher stage

This directory contains the complete 32 x 32 ResNet56 teacher stage used
before downstream CNN-to-ViT experiments.

```text
teachers/
├── checkpoints/
│   ├── cifar100/
│   ├── flowers102/
│   ├── chaoyang/
│   ├── cub200/
│   ├── cub200_resnet50_224_imagenet1k_v2/
│   ├── README.md
│   └── manifest.json
├── train_teacher_cifar100.py
├── train_teacher_flowers.py
├── train_teacher_chaoyang.py
└── verify_checkpoints.py
```

The training recipes are documented in the repository-level `PROTOCOL.md`.
The fixed weights, metrics, configs, summaries, logs, and integrity hashes are
documented in `checkpoints/README.md`.

Verify all four selected teacher checkpoints from the repository root:

```bash
python teachers/verify_checkpoints.py --dataset all
```

The separate verified CUB ImageNet1K-V2 ResNet50-224 teacher from builds 511
and 523 is stored under `checkpoints/cub200_resnet50_224_imagenet1k_v2/` with
its own manifest. The two source archives have exactly equal model tensors,
and both source hashes are retained. Build 519's random-init ResNet50-224
teacher is not stored here because its teacher artifact directory was absent
from the supplied archive.
