#!/usr/bin/env python3
"""Train an ImageNet-pretrained or scratch ResNet50 CUB teacher at 224px."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.transforms import InterpolationMode


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from teachers import train_teacher_cifar100 as common  # noqa: E402
from teachers.cub200_dataset import (  # noqa: E402
    CUB200Dataset,
    EXPECTED_TEST_IMAGES,
    EXPECTED_TRAIN_IMAGES,
    ensure_cub200,
)


NUM_CLASSES = 200
IMAGE_SIZE = 224
RESIZE_SIZE = 256
TRAIN_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 128
PLANNED_EPOCHS = 200
BASE_LR = 0.01
WEIGHT_DECAY = 5e-4
SEED = 1
MODEL_NAME = "resnet50_cub200_imagenet1k_v2_224"
PRETRAINED_SOURCE = "torchvision.ResNet50_Weights.IMAGENET1K_V2"
SCRATCH_MODEL_NAME = "resnet50_cub200_scratch_224"
FEATURE_STAGE_NAMES = ("layer2", "layer3", "layer4")
FEATURE_CHANNELS = (512, 1024, 2048)
FEATURE_SPATIAL_SIZES = (28, 14, 7)
RECIPE_NAME = "cub200_common_transfer_resnet50_224_imagenet1k_v2_200ep_seed1"
SCRATCH_RECIPE_NAME = "cub200_resnet50_224_scratch_200ep_seed1"
TRANSFER_PROTOCOL_FAMILY = "cub200_common_transfer_resnet50_224"
SCRATCH_PROTOCOL_FAMILY = "cub200_resnet50_224_scratch"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ResNet50CUB200(nn.Module):
    """TorchVision ResNet50 with a CUB head and three late-stage features."""

    def __init__(self, *, pretrained: bool = False) -> None:
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = resnet50(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, NUM_CLASSES)
        nn.init.normal_(self.backbone.fc.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.backbone.fc.bias)

    def forward_features(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        backbone = self.backbone
        x = backbone.conv1(x)
        x = backbone.bn1(x)
        x = backbone.relu(x)
        x = backbone.maxpool(x)
        x = backbone.layer1(x)
        f2 = backbone.layer2(x)
        f3 = backbone.layer3(f2)
        f4 = backbone.layer4(f3)
        return f2, f3, f4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        final_feature = self.forward_features(x)[-1]
        pooled = self.backbone.avgpool(final_feature)
        return self.backbone.fc(torch.flatten(pooled, 1))


def common_transfer_train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                IMAGE_SIZE,
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def common_transfer_test_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(
                RESIZE_SIZE,
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("./data/cub200"))
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs"))
    parser.add_argument("--run-name", default=None)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--timing-run", action="store_true")
    modes.add_argument("--smoke", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--smoke-train-samples", type=int, default=128)
    parser.add_argument("--smoke-test-samples", type=int, default=128)
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Require an already extracted CUB_200_2011 directory.",
    )
    parser.add_argument(
        "--initialization",
        choices=("imagenet1k_v2", "scratch"),
        default="imagenet1k_v2",
        help=(
            "Teacher initialization. The scratch option is a controlled "
            "ResNet50-224 ablation and keeps every other teacher setting fixed."
        ),
    )
    parser.add_argument(
        "--no-pretrained-download",
        action="store_true",
        help=(
            "Testing escape hatch only: initialize ResNet50 randomly. The locked "
            "common-transfer runner never supplies this option."
        ),
    )
    return parser.parse_args()


def deterministic_subset(
    dataset: Dataset[Any], size: int, seed: int
) -> Dataset[Any]:
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[
        : min(size, len(dataset))
    ]
    return Subset(dataset, indices.tolist())


def build_loaders(
    args: argparse.Namespace, device: torch.device
) -> tuple[DataLoader[Any], DataLoader[Any], Path]:
    dataset_root = ensure_cub200(args.data_dir, download=not args.no_download)
    train_dataset: Dataset[Any] = CUB200Dataset(
        dataset_root,
        split="train",
        transform=common_transfer_train_transform(),
    )
    test_dataset: Dataset[Any] = CUB200Dataset(
        dataset_root,
        split="test",
        transform=common_transfer_test_transform(),
    )
    if len(train_dataset) != EXPECTED_TRAIN_IMAGES:
        raise RuntimeError(f"Unexpected CUB train size: {len(train_dataset)}")
    if len(test_dataset) != EXPECTED_TEST_IMAGES:
        raise RuntimeError(f"Unexpected CUB test size: {len(test_dataset)}")
    if args.smoke:
        train_dataset = deterministic_subset(
            train_dataset, args.smoke_train_samples, SEED
        )
        test_dataset = deterministic_subset(
            test_dataset, args.smoke_test_samples, SEED + 1
        )
    shared = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        **shared,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        **shared,
    )
    common.log(
        f"[DATA] root={dataset_root} train_samples={len(train_dataset)} "
        f"test_samples={len(test_dataset)} split=official_train/test"
    )
    return train_loader, test_loader, dataset_root


def protocol_check(model: nn.Module) -> None:
    model.eval()
    with torch.inference_mode():
        features = model.forward_features(torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE))
        logits = model(torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE))
    expected_features = tuple(
        (1, channels, size, size)
        for channels, size in zip(
            FEATURE_CHANNELS, FEATURE_SPATIAL_SIZES, strict=True
        )
    )
    feature_shapes = tuple(tuple(feature.shape) for feature in features)
    if feature_shapes != expected_features or tuple(logits.shape) != (1, NUM_CLASSES):
        raise RuntimeError(
            "ResNet50-224 teacher protocol check failed: "
            f"features={feature_shapes} logits={tuple(logits.shape)}"
        )
    common.log(
        "[PROTOCOL_CHECK] status=PASS "
        f"features={feature_shapes} logits={tuple(logits.shape)}"
    )


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    accuracy: float,
    best_accuracy: float,
    epoch_times: list[float],
    mode: str,
    pretrained: bool,
    model_name: str,
    recipe_name: str,
    protocol_family: str,
) -> dict[str, Any]:
    state = model.state_dict()
    return {
        "epoch": epoch,
        "accuracy": accuracy,
        "best_accuracy": best_accuracy,
        "model": state,
        "model_state": state,
        "optimizer_state": optimizer.state_dict(),
        "model_name": model_name,
        "architecture": "TorchVision ResNet50",
        "dataset": "cub200",
        "num_classes": NUM_CLASSES,
        "input_resolution": IMAGE_SIZE,
        "feature_stage_names": FEATURE_STAGE_NAMES,
        "feature_channels": FEATURE_CHANNELS,
        "feature_spatial_sizes": FEATURE_SPATIAL_SIZES,
        "train_split": "official_train",
        "evaluation_split": "official_test",
        "recipe_name": recipe_name,
        "protocol_family": protocol_family,
        "pretrained": pretrained,
        "pretrained_source": PRETRAINED_SOURCE if pretrained else None,
        "epoch_times": epoch_times,
        "mode": mode,
    }


def write_manifest(
    run_dir: Path,
    best_path: Path,
    payload: dict[str, Any],
    *,
    protocol_family: str,
) -> Path:
    manifest_path = run_dir / "manifest.json"
    spec = {
        "selected_kind": "best",
        "checkpoint": best_path.name,
        "sha256": common.sha256_file(best_path),
        "epoch": int(payload["epoch"]),
        "top1": float(payload["accuracy"]),
        "model_name": payload["model_name"],
        "architecture": payload["architecture"],
        "num_classes": NUM_CLASSES,
        "input_resolution": IMAGE_SIZE,
        "feature_stage_names": list(FEATURE_STAGE_NAMES),
        "feature_channels": list(FEATURE_CHANNELS),
        "feature_spatial_sizes": list(FEATURE_SPATIAL_SIZES),
        "recipe_name": payload["recipe_name"],
        "pretrained": bool(payload["pretrained"]),
        "pretrained_source": payload["pretrained_source"],
        "protocol_family": protocol_family,
    }
    common.atomic_json_save(
        {"version": 1, "teachers": {"cub200": spec}}, manifest_path
    )
    return manifest_path


def train(args: argparse.Namespace) -> None:
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.smoke_train_samples <= 0 or args.smoke_test_samples <= 0:
        raise ValueError("Smoke sample counts must be positive")
    common.install_signal_handlers()
    common.seed_everything(SEED)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = "smoke" if args.smoke else "timing" if args.timing_run else "full"
    epochs = 1 if args.smoke else 2 if args.timing_run else PLANNED_EPOCHS
    if args.no_pretrained_download and args.initialization == "scratch":
        raise ValueError(
            "--no-pretrained-download and --initialization scratch are aliases; "
            "supply only --initialization scratch for a production ablation."
        )
    use_pretrained = (
        args.initialization == "imagenet1k_v2"
        and not args.no_pretrained_download
    )
    model_name = MODEL_NAME if use_pretrained else SCRATCH_MODEL_NAME
    recipe_name = RECIPE_NAME if use_pretrained else SCRATCH_RECIPE_NAME
    protocol_family = (
        TRANSFER_PROTOCOL_FAMILY
        if use_pretrained
        else SCRATCH_PROTOCOL_FAMILY
    )
    run_name = args.run_name or f"teacher_{recipe_name}_{mode}"
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_stem = (
        "teacher_resnet50_cub200_224"
        if use_pretrained
        else "teacher_resnet50_cub200_224_scratch"
    )
    best_path = run_dir / f"{checkpoint_stem}_best.pt"
    latest_path = run_dir / f"{checkpoint_stem}_latest.pt"
    metrics_path = run_dir / "metrics.csv"
    summary_path = run_dir / "summary.json"

    common.log("=" * 80)
    initialization_label = "IMAGENET TRANSFER" if use_pretrained else "SCRATCH"
    common.log(
        f"TRAIN CUB-200 RESNET50 TEACHER ({initialization_label}, 224 x 224)"
    )
    common.log("=" * 80)
    common.log(
        f"[MODE] mode={mode} actual_epochs={epochs} "
        f"planned_epochs={PLANNED_EPOCHS}"
    )
    common.log(f"[PATH] run_dir={run_dir.resolve()}")
    common.log(
        f"[PROTOCOL_FAMILY] {protocol_family} "
        "separate_from=cub200_resnet56_32_scratch"
    )
    common.log(
        "[PROTOCOL] official_split train=5994 test=5794 architecture=ResNet50 "
        f"input=224 pretrained={'ImageNet1K_V2' if use_pretrained else 'False'} "
        "full_finetune=True optimizer=SGD "
        "lr=0.01 momentum=0.9 nesterov=True weight_decay=0.0005 "
        "cosine=200ep batch=64 seed=1 fp32=True"
    )
    if use_pretrained:
        common.log(
            "[PRETRAINING_NOTICE] The official CUB page warns that CUB images may "
            "overlap ImageNet. Report this transfer-learning family separately "
            "from scratch-teacher results."
        )
    else:
        common.log(
            "[CONTROLLED_ABLATION] teacher_initialization=random; "
            "architecture=ResNet50 input=224 data/split/optimizer/schedule "
            "match the pretrained-teacher family"
        )
    model = ResNet50CUB200(pretrained=use_pretrained)
    if args.no_pretrained_download:
        common.log(
            "[TEST_ONLY_DEVIATION] ResNet50 pretrained download disabled; "
            "use --initialization scratch for production runs."
        )
    protocol_check(model)
    model.to(device)
    train_loader, test_loader, dataset_root = build_loaders(args, device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=BASE_LR,
        momentum=0.9,
        nesterov=True,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=PLANNED_EPOCHS, eta_min=0.0
    )
    criterion = nn.CrossEntropyLoss()
    scaler = common.create_grad_scaler(False)

    with metrics_path.open("w", newline="", encoding="utf-8") as metrics_file:
        csv.writer(metrics_file).writerow(
            (
                "epoch",
                "train_loss",
                "train_top1",
                "test_top1",
                "best_top1",
                "learning_rate",
                "epoch_seconds",
            )
        )

    best_accuracy = -1.0
    latest_accuracy = -1.0
    epoch_times: list[float] = []
    start = time.time()
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        learning_rate = optimizer.param_groups[0]["lr"]
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_size = targets.size(0)
            total_loss += float(loss.detach()) * batch_size
            correct += common.top1_correct(logits.detach(), targets)
            total += batch_size

        latest_accuracy = common.evaluate(model, test_loader, device, False)
        scheduler.step()
        epoch_seconds = time.time() - epoch_start
        epoch_times.append(epoch_seconds)
        is_best = latest_accuracy > best_accuracy
        if is_best:
            best_accuracy = latest_accuracy
        payload = checkpoint_payload(
            model,
            optimizer,
            epoch=epoch,
            accuracy=latest_accuracy,
            best_accuracy=best_accuracy,
            epoch_times=epoch_times,
            mode=mode,
            pretrained=use_pretrained,
            model_name=model_name,
            recipe_name=recipe_name,
            protocol_family=protocol_family,
        )
        common.atomic_torch_save(payload, latest_path)
        if is_best:
            common.atomic_torch_save(payload, best_path)
        with metrics_path.open("a", newline="", encoding="utf-8") as metrics_file:
            csv.writer(metrics_file).writerow(
                (
                    epoch,
                    f"{total_loss / max(1, total):.8f}",
                    f"{100.0 * correct / max(1, total):.6f}",
                    f"{latest_accuracy:.6f}",
                    f"{best_accuracy:.6f}",
                    f"{learning_rate:.10f}",
                    f"{epoch_seconds:.6f}",
                )
            )
        average_epoch = sum(epoch_times) / len(epoch_times)
        common.log(
            f"[TEACHER224][{epoch:03d}/{epochs:03d}] "
            f"loss={total_loss / max(1, total):.4f} "
            f"train_acc={100.0 * correct / max(1, total):.2f}% "
            f"test_acc={latest_accuracy:.2f}% best={best_accuracy:.2f}% "
            f"lr={learning_rate:.8f} time={epoch_seconds:.1f}s "
            f"est_{PLANNED_EPOCHS}={common.format_duration(average_epoch * PLANNED_EPOCHS)}"
            + (" saved_best" if is_best else "")
        )

    best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
    manifest_path = write_manifest(
        run_dir,
        best_path,
        best_payload,
        protocol_family=protocol_family,
    )
    elapsed = time.time() - start
    average_epoch = sum(epoch_times) / len(epoch_times)
    summary = {
        "status": "complete",
        "mode": mode,
        "dataset": "cub200",
        "dataset_root": str(dataset_root),
        "protocol_family": protocol_family,
        "model_name": model_name,
        "recipe_name": recipe_name,
        "input_resolution": IMAGE_SIZE,
        "pretrained": use_pretrained,
        "pretrained_source": PRETRAINED_SOURCE if use_pretrained else None,
        "completed_epoch": epochs,
        "planned_epochs": PLANNED_EPOCHS,
        "latest_top1": latest_accuracy,
        "best_top1": best_accuracy,
        "avg_epoch_seconds": average_epoch,
        "estimated_planned_seconds": average_epoch * PLANNED_EPOCHS,
        "estimated_planned_human": common.format_duration(
            average_epoch * PLANNED_EPOCHS
        ),
        "elapsed_seconds": elapsed,
        "paths": {
            "best": str(best_path.resolve()),
            "latest": str(latest_path.resolve()),
            "manifest": str(manifest_path.resolve()),
            "metrics": str(metrics_path.resolve()),
        },
        "sha256": {"best": common.sha256_file(best_path)},
    }
    common.atomic_json_save(summary, summary_path)
    common.log(
        f"[FINAL_RESULT] teacher224_best_top1={best_accuracy:.2f}% "
        f"checkpoint={best_path.resolve()}"
    )
    common.log(f"[FINAL_RESULT] manifest={manifest_path.resolve()}")
    common.log(
        f"[DONE] CUB-200 ResNet50-224 "
        f"{'transfer' if use_pretrained else 'scratch'} teacher training "
        "completed successfully."
    )


def main() -> None:
    try:
        train(parse_args())
    except Exception as error:
        common.log(f"[FATAL] {type(error).__name__}: {error}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
