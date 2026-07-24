# Adaptive Locality Guidance (ALG)

The active ALG implementation is method-isolated from `Ours`. It runs the
published adaptive controller on the official LG training base implemented in
[`methods/LG`](../LG).

## Locked method

- LG source: `lkhl/tiny-transformers` commit
  `d2165f74049c906b0afc9f957491960fb3c0cc8b`.
- Student features: DeiT-Ti blocks `[0, 6, 11]`.
- Teacher features: ResNet56 stages `[0, 1, 2]`.
- Alignment: learned stage-wise `1 x 1` projections, bilinear resize to the
  larger grid, and the sum of stage mean MSE.
- Training loss while active: `CE + 2.5 * LG`.
- Adaptive controller: ALG paper equations, window `50`, threshold `-0.02`,
  `smoothed derivative >= threshold` stop boundary, and no extra controller
  warm-up. The optimizer retains the official 20-epoch LR warm-up.
- DeiT classifier head: zero initialized as in the official LG source.
- AdamW: official four parameter groups; biases, one-dimensional parameters,
  `cls_token`, and `pos_embed` receive zero weight decay.
- Batch `128`, eval batch `200`, 300 epochs, seed `1`, FP32, direct bilinear
  224-pixel student view and bilinear 32-pixel teacher view.

The standard wrappers reject the historical researcher normalization,
strict-`>` stop comparison, controller warm-up 20, `draft_common`, and
Ours-matched optimizer/data settings.

Primary sources:

- [ALG paper DOI](https://doi.org/10.1109/TNNLS.2024.3515076)
- [Official LG paper](https://arxiv.org/abs/2207.10026)
- [Official LG repository](https://github.com/lkhl/tiny-transformers)

The active wrappers use the ALG paper's own DeiT-Ti comparison values:
`82.06%`/epoch `124` on CIFAR-100, `69.04%`/epoch `188` on
Flowers-102, and `83.50%`/epoch `108` on Chaoyang. Values from the evolving
project draft are not used as ALG-paper references.

The implementation is a behavior-preserving PyTorch/timm port of the public
LG source rather than the authors' historical environment. It also uses this
repository's fixed ResNet56 checkpoints so that all compared methods share
the same teachers. These are the only controlled integration differences;
no Ours loss, Ours module, Ours optimizer setting, or Ours data policy enters
the active ALG path.

## Entry points

```bash
python methods/ALG/cifar100/train.py
python methods/ALG/flowers102/train.py
python methods/ALG/chaoyang/train.py
python methods/ALG/cub200/train.py
```

Audit all three paper datasets and measure their 300-epoch runtime estimates:

```bash
python methods/ALG/run_three_dataset_timing.py --timing-run --num-workers 4
```

After the timing audit passes, run all three canonical 300-epoch jobs
sequentially and persist every completed result under `/app/output`:

```bash
python methods/ALG/run_three_dataset_full.py \
  --full-run \
  --num-workers 4 \
  --chaoyang-data-dir /app/data/chaoyang
```

The full runner writes a separate directory for each dataset and updates
`three_dataset_full_summary.json` after every completed run. A later failure
therefore does not overwrite or hide an earlier completed checkpoint.

CUB-200 is a protocol transfer, not a result claimed by either source paper:
it uses the authors' official CUB train/test split, the repository's shared
scratch ResNet56 teacher, and the otherwise unchanged LG/ALG mechanics.

Historical noncanonical diagnostics are labeled under [`legacy`](legacy) and
are excluded from active runners.
