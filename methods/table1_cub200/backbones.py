"""Audited Table-1 student backbone construction and feature contracts."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
VENDOR_ROOT = REPOSITORY_ROOT / "third_party" / "tiny_transformers"
OFFICIAL_LG_REPOSITORY = "https://github.com/lkhl/tiny-transformers"
OFFICIAL_LG_COMMIT = "d2165f74049c906b0afc9f957491960fb3c0cc8b"


@dataclass(frozen=True)
class BackboneSpec:
    key: str
    display_name: str
    config_path: str
    model_type: str
    selected_feature_indices: tuple[int, int, int]
    depth: int
    common_ours_channels: int = 192
    common_ours_grid: int = 14


BACKBONES: dict[str, BackboneSpec] = {
    "deit_ti": BackboneSpec(
        "deit_ti",
        "DeiT-Ti",
        "configs/deit/deit-ti_c100_ours.yaml",
        "DeiT",
        (0, 6, 11),
        12,
    ),
    "convit_ti": BackboneSpec(
        "convit_ti",
        "ConViT-Ti",
        "configs/convit/convit-ti_c100_ours.yaml",
        "ConViT",
        (0, 6, 11),
        12,
    ),
    "cvt_13": BackboneSpec(
        "cvt_13",
        "CvT-13",
        "configs/cvt/cvt-13_c100_ours.yaml",
        "CvT",
        (0, 6, 11),
        13,
    ),
    "pit_ti": BackboneSpec(
        "pit_ti",
        "PiT-Ti",
        "configs/pit/pit-ti_c100_ours.yaml",
        "PiT",
        (0, 6, 11),
        12,
    ),
    "pvtv2_b0": BackboneSpec(
        "pvtv2_b0",
        "PVTv2-B0",
        "configs/pvtv2/pvtv2-b0_c100_ours.yaml",
        "PVTv2",
        (0, 3, 7),
        8,
    ),
    "t2t_vit_7": BackboneSpec(
        "t2t_vit_7",
        "T2T-ViT-7",
        "configs/t2t/t2t-7_c100_ours.yaml",
        "T2TViT",
        (0, 3, 6),
        7,
    ),
    "t2t_vit_14": BackboneSpec(
        "t2t_vit_14",
        "T2T-ViT-14",
        "configs/t2t/t2t-14_c100_ours.yaml",
        "T2TViT",
        (0, 7, 13),
        14,
    ),
}


def _load_vendor_modules() -> tuple[Any, Any]:
    vendor_text = str(VENDOR_ROOT)
    if vendor_text not in sys.path:
        sys.path.insert(0, vendor_text)
    from pycls.core.config import cfg, reset_cfg

    cfg.defrost()
    reset_cfg()
    # Importing pycls.models registers all seven official model classes.
    import pycls.models  # noqa: F401
    from pycls.models.build import build_model

    return cfg, build_model


def create_student(
    student_key: str,
    *,
    num_classes: int = 200,
) -> tuple[Any, BackboneSpec]:
    """Create one scratch student using the official LG architecture config."""

    if student_key not in BACKBONES:
        raise KeyError(f"Unknown Table-1 student {student_key!r}")
    spec = BACKBONES[student_key]
    config_path = VENDOR_ROOT / spec.config_path
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    cfg, build_model = _load_vendor_modules()
    cfg.merge_from_file(str(config_path))
    cfg.defrost()
    cfg.MODEL.NUM_CLASSES = int(num_classes)
    cfg.MODEL.IMG_SIZE = 224
    cfg.DISTILLATION.ENABLE_INTER = False
    cfg.DISTILLATION.ENABLE_LOGIT = False
    cfg.DISTILLATION.TEACHER_WEIGHTS = None
    cfg.freeze()

    model = build_model()
    if type(model).__name__ != spec.model_type:
        raise RuntimeError(
            f"Expected official {spec.model_type}, created {type(model).__name__}"
        )
    feature_dims = tuple(int(value) for value in model.feature_dims)
    if len(feature_dims) != spec.depth:
        raise RuntimeError(
            f"{spec.display_name} feature depth mismatch: "
            f"expected={spec.depth} runtime={len(feature_dims)}"
        )
    if max(spec.selected_feature_indices) >= len(feature_dims):
        raise RuntimeError(
            f"{spec.display_name} selected features "
            f"{spec.selected_feature_indices} exceed depth {len(feature_dims)}"
        )
    return model, spec


def forward_student(
    student: Any,
    images: Any,
) -> tuple[list[Any], Any]:
    """Return every official block feature and classifier logits."""

    logits = student(images)
    features = list(student.features)
    if not features:
        raise RuntimeError("Official Table-1 student produced no block features")
    return features, logits
