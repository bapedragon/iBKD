#!/usr/bin/env python3
"""Run official LG mechanics with fully pretrained CUB-200 backbones."""

from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from methods.LG.entrypoint import run_dataset  # noqa: E402


def has_option(option: str) -> bool:
    return any(
        argument == option or argument.startswith(f"{option}=")
        for argument in sys.argv[1:]
    )


if __name__ == "__main__":
    if "--no-student-pretrained" in sys.argv[1:]:
        raise SystemExit(
            "This wrapper locks ImageNet-pretrained DeiT-Ti; remove "
            "--no-student-pretrained."
        )
    if not has_option("--student-pretrained"):
        sys.argv[1:1] = ["--student-pretrained"]
    if not has_option("--teacher-image-size"):
        sys.argv[1:1] = ["--teacher-image-size", "224"]
    if not has_option("--student-epochs"):
        sys.argv[1:1] = ["--student-epochs", "100"]
    run_dataset(
        "cub200",
        "cub200_deit_ti_lg_resnet50_224_both_imagenet_pretrained_v1",
    )
