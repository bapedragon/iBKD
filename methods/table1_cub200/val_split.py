"""Deterministic class-stratified validation split for official CUB train data."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any


DEFAULT_VAL_PER_CLASS = 6
DEFAULT_SPLIT_SEED = 2027


def _rank_key(split_seed: int, image_id: int) -> str:
    return hashlib.sha256(f"{split_seed}:{image_id}".encode("utf-8")).hexdigest()


def build_stratified_train_val_indices(
    dataset: Any,
    *,
    val_per_class: int = DEFAULT_VAL_PER_CLASS,
    split_seed: int = DEFAULT_SPLIT_SEED,
) -> tuple[list[int], list[int], dict[str, Any]]:
    """Split a CUB official-train dataset without consulting official test data."""

    if val_per_class <= 0:
        raise ValueError("val_per_class must be positive")
    records = getattr(dataset, "records", None)
    if not isinstance(records, list) or not records:
        raise ValueError("dataset must expose a non-empty .records list")

    by_class: dict[int, list[tuple[int, Any]]] = defaultdict(list)
    for index, record in enumerate(records):
        if not bool(getattr(record, "is_train", False)):
            raise ValueError("validation splitting accepts official-train records only")
        by_class[int(record.target)].append((index, record))

    train_indices: list[int] = []
    val_indices: list[int] = []
    val_image_ids: list[int] = []
    class_counts: dict[str, dict[str, int]] = {}
    for target in sorted(by_class):
        ranked = sorted(
            by_class[target],
            key=lambda item: (
                _rank_key(split_seed, int(item[1].image_id)),
                int(item[1].image_id),
            ),
        )
        if len(ranked) <= val_per_class:
            raise ValueError(
                f"class {target} has {len(ranked)} samples, "
                f"not enough for val_per_class={val_per_class}"
            )
        val_items = ranked[:val_per_class]
        train_items = ranked[val_per_class:]
        val_indices.extend(index for index, _ in val_items)
        train_indices.extend(index for index, _ in train_items)
        val_image_ids.extend(int(record.image_id) for _, record in val_items)
        class_counts[str(target)] = {
            "official_train": len(ranked),
            "train": len(train_items),
            "validation": len(val_items),
        }

    train_indices.sort()
    val_indices.sort()
    val_image_ids.sort()
    if set(train_indices).intersection(val_indices):
        raise RuntimeError("train and validation indices overlap")
    if sorted(train_indices + val_indices) != list(range(len(records))):
        raise RuntimeError("train/validation indices do not partition official train")

    encoded_ids = "\n".join(str(value) for value in val_image_ids).encode("utf-8")
    manifest = {
        "protocol": "official_train_stratified_fixed_validation",
        "split_seed": int(split_seed),
        "val_per_class": int(val_per_class),
        "class_count": len(by_class),
        "official_train_samples": len(records),
        "train_samples": len(train_indices),
        "validation_samples": len(val_indices),
        "validation_image_ids_sha256": hashlib.sha256(encoded_ids).hexdigest(),
        "validation_image_ids": val_image_ids,
        "class_counts": class_counts,
    }
    return train_indices, val_indices, manifest
