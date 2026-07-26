# Fixed CUB-200 Table-1 teacher

This directory contains the exact selected teacher from H200 build 543.
Every current and future guided student under `methods/table1_cub200/` must
load this directory and pass its manifest/hash check.

| Field | Locked value |
|---|---|
| Architecture | scratch CIFAR-style ResNet56 |
| Input | `32x32` |
| Training | 300 epochs, seed 1 |
| Selected epoch | 275 |
| Selected Top-1 | **36.40%** |
| Checkpoint SHA-256 | `06f75192b1c108c89e480843cb4f72dfb28aa762d7b11e7ac327333dd54b51f5` |
| H200 source | build 543 |

This is deliberately separate from the earlier primary CUB ResNet56
checkpoint (`37.25%`, build 509). The two protocol families must not exchange
teachers.
