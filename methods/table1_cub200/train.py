#!/usr/bin/env python3
"""Train or time one CUB-200 Table-1 student/method combination."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from methods.LG.runtime import (  # noqa: E402
    AdaptiveGuidanceController,
    StaticGuidanceController,
    create_scheduler,
)
from methods.Ours.core import (  # noqa: E402
    AdaptiveGuidanceController as OursAdaptiveGuidanceController,
)
from methods.Ours.ours import Ours  # noqa: E402
from methods.table1_cub200.adapters import (  # noqa: E402
    OfficialLGFeatureLoss,
    OursAllBlockAdapter,
)
from methods.table1_cub200.backbones import (  # noqa: E402
    BACKBONES,
    OFFICIAL_LG_COMMIT,
    OFFICIAL_LG_REPOSITORY,
    create_student,
    forward_student,
)
from teachers.cub200_dataset import CUB200Dataset, ensure_cub200  # noqa: E402
from teachers.verify_checkpoints import (  # noqa: E402
    DEFAULT_CHECKPOINT_ROOT,
    load_teacher,
)


PLANNED_EPOCHS = 300
NUM_CLASSES = 200
IMAGE_SIZE = 224
TEACHER_IMAGE_SIZE = 32
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
METHODS = ("vanilla", "lg", "alg", "ours")


def log(message: str = "") -> None:
    print(message, flush=True)


def format_duration(seconds: float) -> str:
    rounded = max(0, int(round(seconds)))
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def atomic_json_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", choices=tuple(BACKBONES), required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--timing-run", action="store_true")
    modes.add_argument("--full-run", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=200)
    parser.add_argument("--data-dir", type=Path, default=Path("./data/cub200"))
    parser.add_argument(
        "--teacher-root",
        type=Path,
        default=DEFAULT_CHECKPOINT_ROOT,
    )
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise ValueError("Batch sizes must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.method != "ours" and args.batch_size != 128:
        raise ValueError("Table-1 Vanilla/LG/ALG are locked to batch 128")
    if args.method == "ours" and args.batch_size not in {64, 128}:
        raise ValueError("CUB Table-1 Ours is measured only at batch 64 or 128")
    if args.seed != 1:
        raise ValueError("The primary Table-1 extension is locked to seed 1")


def build_loaders(
    args: argparse.Namespace,
    device: torch.device,
    timm: Any,
) -> tuple[DataLoader[Any], DataLoader[Any], Path]:
    train_transform = timm.data.create_transform(
        input_size=(3, IMAGE_SIZE, IMAGE_SIZE),
        is_training=True,
        color_jitter=0.4,
        auto_augment="rand-m9-mstd0.5-inc1",
        re_prob=0.25,
        re_mode="pixel",
        re_count=1,
        interpolation="bicubic",
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    dataset_root = ensure_cub200(args.data_dir)
    train_dataset = CUB200Dataset(
        dataset_root,
        split="train",
        transform=train_transform,
    )
    test_dataset = CUB200Dataset(
        dataset_root,
        split="test",
        transform=test_transform,
    )

    def seed_worker(worker_id: int) -> None:
        del worker_id
        worker_seed = torch.initial_seed() % (2**32)
        random.seed(worker_seed)

    generator = torch.Generator().manual_seed(args.seed)
    common = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator,
        **common,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
        **common,
    )
    if len(train_dataset) != 5994 or len(test_dataset) != 5794:
        raise RuntimeError(
            "Unexpected CUB official split: "
            f"train={len(train_dataset)} test={len(test_dataset)}"
        )
    return train_loader, test_loader, dataset_root


def teacher_features(
    teacher: nn.Module,
    images: torch.Tensor,
) -> list[torch.Tensor]:
    teacher_images = F.interpolate(
        images,
        size=(TEACHER_IMAGE_SIZE, TEACHER_IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    )
    return list(teacher.forward_features(teacher_images))


def no_decay_parameter_groups(
    named_modules: list[tuple[str, nn.Module]],
    *,
    lr: float,
    weight_decay: float,
) -> list[dict[str, Any]]:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    seen: set[int] = set()
    skip_tokens = ("cls_token", "pos_embed", "distill_token")
    for prefix, module in named_modules:
        for name, parameter in module.named_parameters():
            if not parameter.requires_grad or id(parameter) in seen:
                continue
            seen.add(id(parameter))
            qualified = f"{prefix}.{name}"
            if (
                parameter.ndim == 1
                or name.endswith(".bias")
                or any(token in qualified for token in skip_tokens)
            ):
                no_decay.append(parameter)
            else:
                decay.append(parameter)
    return [
        {
            "name": "no_decay",
            "params": no_decay,
            "lr": lr,
            "weight_decay": 0.0,
        },
        {
            "name": "decay",
            "params": decay,
            "lr": lr,
            "weight_decay": weight_decay,
        },
    ]


@torch.inference_mode()
def evaluate(
    student: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
) -> float:
    student.eval()
    correct = 0
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        _, logits = forward_student(student, images)
        correct += int((logits.argmax(dim=1) == targets).sum())
        total += targets.size(0)
    return 100.0 * correct / max(1, total)


def make_controller(args: argparse.Namespace) -> Any | None:
    if args.method == "vanilla":
        return None
    if args.method == "lg":
        return StaticGuidanceController(beta=2.5)
    if args.method == "alg":
        return AdaptiveGuidanceController(
            beta=2.5,
            threshold=-0.02,
            smoothing_window=50,
            warm_up=0,
            stop_comparison="paper_ge",
            derivative_mode="paper_equations",
        )
    controller_args = argparse.Namespace(
        beta_schedule="alg",
        beta_on=2.5,
        alg_threshold=-0.02,
        alg_smoothing_window=50,
        alg_warmup_epochs=20,
        guidance_stop_epoch=None,
    )
    return OursAdaptiveGuidanceController(controller_args)


def controller_observe(
    controller: Any | None,
    epoch: int,
    guidance_loss: float,
    beta: float,
) -> None:
    if controller is not None and beta > 0.0:
        controller.observe(epoch, guidance_loss)


def train(args: argparse.Namespace) -> None:
    validate_args(args)
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = False
    import timm

    if timm.__version__ != "1.0.27":
        raise RuntimeError(f"Expected timm==1.0.27, found {timm.__version__}")

    actual_epochs = 2 if args.timing_run else PLANNED_EPOCHS
    mode = "timing_2ep" if args.timing_run else "full_300ep"
    run_name = args.run_name or (
        f"table1_cub200_{args.student}_{args.method}_"
        f"b{args.batch_size}_{mode}_seed1"
    )
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    best_path = run_dir / "student_best.pt"
    latest_path = run_dir / "student_latest.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student, backbone_spec = create_student(args.student, num_classes=NUM_CLASSES)
    student = student.to(device)
    train_loader, test_loader, dataset_root = build_loaders(args, device, timm)

    teacher: nn.Module | None = None
    teacher_spec: dict[str, Any] | None = None
    method_module: nn.Module | None = None
    ours_adapter: OursAllBlockAdapter | None = None
    if args.method != "vanilla":
        teacher, _, teacher_spec = load_teacher(
            "cub200",
            device=device,
            checkpoint_root=args.teacher_root,
        )
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad = False
        teacher_channels = tuple(
            int(value)
            for value in teacher_spec.get("feature_channels", (16, 32, 64))
        )
        if teacher_channels != (16, 32, 64):
            raise RuntimeError(
                f"Table-1 CUB requires ResNet56 channels 16/32/64, got {teacher_channels}"
            )
        if args.method in {"lg", "alg"}:
            method_module = OfficialLGFeatureLoss(
                student.feature_dims,
                backbone_spec.selected_feature_indices,
                teacher_channels,
            ).to(device)
        else:
            ours_adapter = OursAllBlockAdapter(
                student.feature_dims,
                output_channels=backbone_spec.common_ours_channels,
                output_grid=backbone_spec.common_ours_grid,
            ).to(device)
            method_module = Ours(
                student_channels=backbone_spec.common_ours_channels,
                teacher_channels=teacher_channels,
                num_student_blocks=backbone_spec.depth,
                num_heads=4,
                spatial_kernel_size=5,
                grid_resize_mode="larger",
            ).to(device)

    with torch.no_grad():
        probe_batch = 2
        probe = torch.zeros(
            probe_batch,
            3,
            IMAGE_SIZE,
            IMAGE_SIZE,
            device=device,
        )
        probe_features, probe_logits = forward_student(student, probe)
        if len(probe_features) != backbone_spec.depth:
            raise RuntimeError(
                f"{backbone_spec.display_name} runtime feature count "
                f"{len(probe_features)} != {backbone_spec.depth}"
            )
        if tuple(probe_logits.shape) != (probe_batch, NUM_CLASSES):
            raise RuntimeError(f"Unexpected logits shape {tuple(probe_logits.shape)}")
        teacher_probe: list[torch.Tensor] | None = None
        probe_guidance = probe_logits.new_zeros(())
        if teacher is not None:
            teacher_probe = teacher_features(teacher, probe)
            if len(teacher_probe) != 3:
                raise RuntimeError("ResNet56 teacher must expose three stages")
            if args.method in {"lg", "alg"}:
                assert isinstance(method_module, OfficialLGFeatureLoss)
                probe_guidance = method_module(probe_features, teacher_probe)
            else:
                assert ours_adapter is not None
                assert isinstance(method_module, Ours)
                adapted = ours_adapter(probe_features)
                align, fuse, _, _, _ = method_module(adapted, teacher_probe)
                probe_guidance = 0.5 * align + 0.5 * fuse
        if not bool(torch.isfinite(probe_guidance)):
            raise RuntimeError("Non-finite guidance probe")

    lr = 5e-4
    min_lr = 5e-6
    weight_decay = 0.05
    warmup_epochs = 20
    warmup_factor = 0.001
    if args.method == "ours":
        trainable = list(student.parameters())
        assert method_module is not None and ours_adapter is not None
        trainable.extend(ours_adapter.parameters())
        trainable.extend(method_module.parameters())
        optimizer = torch.optim.AdamW(
            trainable,
            lr=lr,
            weight_decay=weight_decay,
        )
        optimizer_contract = "ours_single_group_all_parameters_decay_0.05"
    else:
        named_modules = [("student", student)]
        if method_module is not None:
            named_modules.append(("guidance", method_module))
        optimizer = torch.optim.AdamW(
            no_decay_parameter_groups(
                named_modules,
                lr=lr,
                weight_decay=weight_decay,
            ),
            lr=lr,
            weight_decay=weight_decay,
        )
        optimizer_contract = "official_lg_no_decay_exclusions"
    scheduler = create_scheduler(
        optimizer,
        planned_epochs=PLANNED_EPOCHS,
        lr=lr,
        min_lr=min_lr,
        warmup_epochs=warmup_epochs,
        warmup_factor=warmup_factor,
    )
    controller = make_controller(args)

    log("=" * 88)
    log("CUB-200 TABLE-1 EXTENSION")
    log("=" * 88)
    log(
        f"[TASK] student={backbone_spec.display_name} method={args.method.upper()} "
        f"batch={args.batch_size} mode={mode}"
    )
    log(
        f"[MODE] actual_epochs={actual_epochs} planned_epochs={PLANNED_EPOCHS}"
    )
    log(
        "[PROTOCOL_LOCK] dataset=CUB-200-2011 split=5994/5794 "
        "teacher=ResNet56-scratch-32 student=scratch-224 "
        "epochs=300 AdamW=5e-4->5e-6 wd=0.05 warmup=20 seed=1 fp32=True"
    )
    log(
        f"[SOURCE] official_lg_repo={OFFICIAL_LG_REPOSITORY} "
        f"commit={OFFICIAL_LG_COMMIT}"
    )
    log(
        f"[MODEL] params={count_parameters(student):,} "
        f"feature_dims={tuple(int(v) for v in student.feature_dims)} "
        f"selected_lg={backbone_spec.selected_feature_indices}"
    )
    log(
        f"[FEATURE_CHECK] student={tuple(tuple(x.shape) for x in probe_features)} "
        f"teacher={None if teacher_probe is None else tuple(tuple(x.shape) for x in teacher_probe)} "
        f"guidance_probe={float(probe_guidance):.6f}"
    )
    if args.method == "ours":
        assert ours_adapter is not None and method_module is not None
        log(
            "[OURS_ADAPTER] all_blocks=True common_channels=192 common_grid=14 "
            f"adapter_params={count_parameters(ours_adapter):,} "
            f"ours_params={count_parameters(method_module):,}"
        )
    log(f"[OPTIMIZER] contract={optimizer_contract}")

    best_accuracy = float("-inf")
    latest_accuracy = float("-inf")
    epoch_times: list[float] = []
    training_start = time.time()
    for epoch in range(1, actual_epochs + 1):
        epoch_start = time.time()
        epoch_lr = float(optimizer.param_groups[0]["lr"])
        beta = 0.0 if controller is None else float(controller.beta_for_epoch(epoch))
        student.train()
        if method_module is not None:
            method_module.train()
        if ours_adapter is not None:
            ours_adapter.train()
        total_loss = 0.0
        total_ce = 0.0
        total_guidance = 0.0
        correct = 0
        total = 0

        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            frozen_teacher_features: list[torch.Tensor] | None = None
            if teacher is not None and beta > 0.0:
                with torch.no_grad():
                    frozen_teacher_features = teacher_features(teacher, images)
            block_features, logits = forward_student(student, images)
            ce = F.cross_entropy(logits, targets)
            guidance = ce.new_zeros(())
            if frozen_teacher_features is not None:
                if args.method in {"lg", "alg"}:
                    assert isinstance(method_module, OfficialLGFeatureLoss)
                    guidance = method_module(
                        block_features,
                        frozen_teacher_features,
                    )
                else:
                    assert ours_adapter is not None
                    assert isinstance(method_module, Ours)
                    adapted_features = ours_adapter(block_features)
                    alignment, fusion, _, _, _ = method_module(
                        adapted_features,
                        frozen_teacher_features,
                    )
                    guidance = 0.5 * alignment + 0.5 * fusion
            loss = ce + beta * guidance
            loss.backward()
            optimizer.step()

            batch_size = targets.size(0)
            total += batch_size
            total_loss += float(loss.detach()) * batch_size
            total_ce += float(ce.detach()) * batch_size
            total_guidance += float(guidance.detach()) * batch_size
            correct += int((logits.detach().argmax(dim=1) == targets).sum())

        average_guidance = total_guidance / max(1, total)
        controller_observe(controller, epoch, average_guidance, beta)
        latest_accuracy = evaluate(student, test_loader, device)
        epoch_seconds = time.time() - epoch_start
        epoch_times.append(epoch_seconds)
        best_accuracy = max(best_accuracy, latest_accuracy)
        scheduler.step(epoch)
        average_epoch = sum(epoch_times) / len(epoch_times)
        log(
            f"[{args.method.upper()}:{backbone_spec.display_name}]"
            f"[{epoch:03d}/{actual_epochs:03d}] "
            f"loss={total_loss / max(1, total):.4f} "
            f"ce={total_ce / max(1, total):.4f} "
            f"guidance={average_guidance:.4f} beta={beta:.2f} "
            f"train_acc={100.0 * correct / max(1, total):.2f}% "
            f"test_acc={latest_accuracy:.2f}% best={best_accuracy:.2f}% "
            f"lr={epoch_lr:.8g} time={epoch_seconds:.1f}s "
            f"est_300={format_duration(average_epoch * PLANNED_EPOCHS)}"
        )

        if args.full_run:
            payload = {
                "student": student.state_dict(),
                "method_module": (
                    None if method_module is None else method_module.state_dict()
                ),
                "ours_adapter": (
                    None if ours_adapter is None else ours_adapter.state_dict()
                ),
                "epoch": epoch,
                "accuracy": latest_accuracy,
                "best_accuracy": best_accuracy,
                "student_key": args.student,
                "method": args.method,
                "batch_size": args.batch_size,
                "teacher": teacher_spec,
                "official_lg_commit": OFFICIAL_LG_COMMIT,
            }
            atomic_torch_save(payload, latest_path)
            if latest_accuracy >= best_accuracy:
                atomic_torch_save(payload, best_path)

    elapsed = time.time() - training_start
    average_epoch = sum(epoch_times) / len(epoch_times)
    summary = {
        "status": "complete",
        "mode": mode,
        "dataset": "cub200",
        "dataset_root": str(dataset_root.resolve()),
        "student_key": args.student,
        "student": backbone_spec.display_name,
        "method": args.method,
        "batch_size": args.batch_size,
        "actual_epochs": actual_epochs,
        "planned_epochs": PLANNED_EPOCHS,
        "best_top1": best_accuracy,
        "latest_top1": latest_accuracy,
        "epoch_times_seconds": epoch_times,
        "avg_epoch_seconds": average_epoch,
        "estimated_planned_seconds": average_epoch * PLANNED_EPOCHS,
        "estimated_planned_human": format_duration(
            average_epoch * PLANNED_EPOCHS
        ),
        "timing_elapsed_seconds": elapsed,
        "teacher": teacher_spec,
        "backbone_source": {
            "repository": OFFICIAL_LG_REPOSITORY,
            "commit": OFFICIAL_LG_COMMIT,
        },
        "optimizer_contract": optimizer_contract,
        "ours_adapter": (
            None
            if args.method != "ours"
            else {
                "all_blocks": True,
                "output_channels": 192,
                "output_grid": 14,
                "batch": args.batch_size,
            }
        ),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "timm": timm.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
    }
    atomic_json_save(summary, summary_path)
    log(
        f"[TIMING_RESULT] student={backbone_spec.display_name} "
        f"method={args.method.upper()} batch={args.batch_size} "
        f"avg_epoch={average_epoch:.2f}s "
        f"estimated_300={summary['estimated_planned_human']}"
    )
    log(f"[FINAL_RESULT] summary={summary_path.resolve()}")
    log("[DONE] Table-1 CUB student task completed successfully.")


def main() -> None:
    try:
        train(parse_args())
    except Exception as error:
        log(f"[FATAL] {type(error).__name__}: {error}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
