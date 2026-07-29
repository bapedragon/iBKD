"""Fixed teacher identity for the CUB-200 Table-1 experiment family."""

from __future__ import annotations

from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TABLE1_TEACHER_ROOT = (
    REPOSITORY_ROOT / "teachers/checkpoints/cub200_table1_resnet56_32"
)
TABLE1_TEACHER_SHA256 = (
    "06f75192b1c108c89e480843cb4f72dfb28aa762d7b11e7ac327333dd54b51f5"
)
TABLE1_TEACHER_BUILD = 543
TABLE1_TEACHER_EPOCH = 275
TABLE1_TEACHER_TOP1 = 36.39972385226096
TABLE1_TEACHER_RECIPE = (
    "cub200_official_split_resnet56_32_scratch_300ep_seed1"
)


def validate_table1_teacher_spec(spec: dict[str, Any]) -> None:
    """Reject any teacher other than the fixed H200 build-543 checkpoint."""

    expected = {
        "checkpoint": "teacher_resnet56_cub200_32_best.pt",
        "sha256": TABLE1_TEACHER_SHA256,
        "epoch": TABLE1_TEACHER_EPOCH,
        "top1": TABLE1_TEACHER_TOP1,
        "num_classes": 200,
        "input_resolution": 32,
        "recipe_name": TABLE1_TEACHER_RECIPE,
        "pretrained": False,
        "selected_kind": "best",
    }
    mismatches = {
        key: {"expected": value, "actual": spec.get(key)}
        for key, value in expected.items()
        if spec.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "CUB-200 Table-1 students are locked to the H200 build-543 "
            f"teacher; mismatches={mismatches}"
        )


def validate_val_selected_teacher_spec(
    spec: dict[str, Any],
    *,
    validation_split_sha256: str,
    validation_split_seed: int,
    val_per_class: int,
) -> None:
    """Validate a scratch teacher selected without consulting official test."""

    expected = {
        "num_classes": 200,
        "input_resolution": 32,
        "pretrained": False,
        "selected_kind": "best_validation",
        "validation_split_sha256": validation_split_sha256,
        "validation_split_seed": validation_split_seed,
        "val_per_class": val_per_class,
    }
    mismatches = {
        key: {"expected": value, "actual": spec.get(key)}
        for key, value in expected.items()
        if spec.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "CUB-200 validation-selected students require a teacher chosen "
            f"on the identical fixed validation split; mismatches={mismatches}"
        )
