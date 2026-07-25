#!/usr/bin/env python3
"""Train teacher-free CUB-200 DeiT-Ti baselines at 224x224."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from methods.LG import runtime as lg_runtime  # noqa: E402
from methods.Ours import core as ours_core  # noqa: E402


DEFAULT_STUDENT_EPOCHS = 300
NUM_CLASSES = 200
LR = 5e-4
MIN_LR = 5e-6
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 20
WARMUP_FACTOR = 0.001
DROP_PATH_RATE = 0.1
SEED = 1
PROFILE_BATCH_SIZE = {
    "lg_official_b128": 128,
    "ours_current_b64": 64,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        required=True,
        choices=tuple(PROFILE_BATCH_SIZE),
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--timing-run", action="store_true")
    modes.add_argument("--full-run", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=Path("./data/cub200"))
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--student-epochs",
        type=int,
        default=DEFAULT_STUDENT_EPOCHS,
        help="Planned training horizon; timing mode still executes two epochs.",
    )
    parser.add_argument(
        "--student-pretrained",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Initialize DeiT-Ti from timm's explicit ImageNet-1K weights.",
    )
    raw_args = sys.argv[1:] if argv is None else argv
    return parser.parse_args(
        argument for argument in raw_args if argument.strip()
    )


def loader_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        dataset="cub200",
        data_dir=args.data_dir,
        batch_size=PROFILE_BATCH_SIZE[args.profile],
        eval_batch_size=200,
        num_workers=args.num_workers,
        seed=SEED,
        smoke=False,
        smoke_train_samples=1024,
        smoke_test_samples=512,
        flowers_split_policy="trainval_test_best",
        eval_interpolation=(
            "bilinear"
            if args.profile == "lg_official_b128"
            else "bicubic"
        ),
    )


def official_student_parameter_groups(
    student: torch.nn.Module,
) -> list[dict[str, Any]]:
    groups: dict[str, list[torch.nn.Parameter]] = {
        "head_no_decay": [],
        "head_decay": [],
        "body_no_decay": [],
        "body_decay": [],
    }
    skip_tokens = ("cls_token", "pos_embed", "distill_token")
    for name, parameter in student.named_parameters():
        if not parameter.requires_grad:
            continue
        is_head = name.startswith("head.")
        no_decay = (
            parameter.ndim == 1
            or name.endswith(".bias")
            or any(token in name for token in skip_tokens)
        )
        key = (
            "head_no_decay"
            if is_head and no_decay
            else "head_decay"
            if is_head
            else "body_no_decay"
            if no_decay
            else "body_decay"
        )
        groups[key].append(parameter)
    return [
        {
            "name": name,
            "params": parameters,
            "lr": LR,
            "weight_decay": 0.0 if name.endswith("no_decay") else WEIGHT_DECAY,
        }
        for name, parameters in groups.items()
    ]


def create_student_and_optimizer(
    args: argparse.Namespace,
    timm: Any,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.optim.AdamW, str]:
    if args.profile == "lg_official_b128":
        student = lg_runtime.create_student(
            timm,
            NUM_CLASSES,
            DROP_PATH_RATE,
            pretrained=args.student_pretrained,
        ).to(device)
        optimizer = torch.optim.AdamW(
            official_student_parameter_groups(student),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )
        optimizer_description = (
            "official_lg_groups no_decay=1d,bias,cls_token,pos_embed "
            "head_zero_init=True"
        )
    else:
        student = ours_core.create_ours_student(
            timm,
            "deit_ti",
            NUM_CLASSES,
            DROP_PATH_RATE,
            pretrained=args.student_pretrained,
        ).to(device)
        optimizer = torch.optim.AdamW(
            student.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )
        optimizer_description = (
            "ours_current_single_group all_student_parameters "
            "head_zero_init=False"
        )
    return student, optimizer, optimizer_description


def train_one_epoch(
    student: torch.nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
) -> tuple[float, float]:
    student.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with lg_runtime.autocast_context(False):
            logits = student(images)
            loss = F.cross_entropy(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_size = targets.size(0)
        total += batch_size
        total_loss += float(loss.detach()) * batch_size
        correct += lg_runtime.top1_correct(logits.detach(), targets)
    return (
        total_loss / max(1, total),
        100.0 * correct / max(1, total),
    )


def atomic_json_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def train(args: argparse.Namespace) -> None:
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.student_epochs <= 0:
        raise ValueError("--student-epochs must be positive")
    lg_runtime.install_signal_handlers()
    lg_runtime.seed_everything(SEED)
    torch.backends.cudnn.benchmark = False
    timm = lg_runtime.ensure_timm()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actual_epochs = 2 if args.timing_run else args.student_epochs
    mode = "timing" if args.timing_run else "full"
    batch_size = PROFILE_BATCH_SIZE[args.profile]
    run_name = args.run_name or (
        f"vanilla_cub200_224_{args.profile}_{actual_epochs}ep_seed1"
    )
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "student_best.pt"
    latest_path = run_dir / "student_latest.pt"
    summary_path = run_dir / "summary.json"
    prepared_args = loader_args(args)

    lg_runtime.log("=" * 80)
    lg_runtime.log("VANILLA CUB-200 DEIT-TI / CE-ONLY / 224 x 224")
    lg_runtime.log("=" * 80)
    lg_runtime.log(
        f"[MODE] mode={mode} actual_epochs={actual_epochs} "
        f"planned_epochs={args.student_epochs}"
    )
    lg_runtime.log(
        f"[VANILLA_PROTOCOL] profile={args.profile} teacher=none "
        f"guidance=none loss=CE-only "
        f"student_pretrained={args.student_pretrained} "
        f"student_pretrained_source="
        f"{lg_runtime.STUDENT_PRETRAINED_SOURCES['deit_ti'] if args.student_pretrained else 'none'} "
        f"student_input=224 batch={batch_size} seed={SEED}"
    )
    lg_runtime.log(
        "[AUGMENT] official_lg_strong color_jitter=0.4 "
        "auto_augment=rand-m9-mstd0.5-inc1 random_erasing=0.25/pixel/1 "
        "train_interpolation=bicubic eval_resize=direct224_"
        f"{prepared_args.eval_interpolation}"
    )
    lg_runtime.log(f"[PATH] data_dir={args.data_dir.resolve()}")
    lg_runtime.log(f"[PATH] run_dir={run_dir.resolve()}")

    train_loader, test_loader, final_test_loader = (
        lg_runtime.build_alg_loaders_with_final_test(
            prepared_args,
            device,
            timm,
        )
    )
    if final_test_loader is not None:
        raise RuntimeError("CUB-200 vanilla expects one official test loader")
    student, optimizer, optimizer_description = create_student_and_optimizer(
        args,
        timm,
        device,
    )
    lg_runtime.log(
        f"[MODEL] student=deit_tiny_patch16_224 params="
        f"{lg_runtime.count_parameters(student):,}"
    )
    lg_runtime.log(f"[OPTIMIZER] {optimizer_description}")
    scheduler = lg_runtime.create_scheduler(
        optimizer,
        planned_epochs=args.student_epochs,
        lr=LR,
        min_lr=MIN_LR,
        warmup_epochs=WARMUP_EPOCHS,
        warmup_factor=WARMUP_FACTOR,
    )
    scaler = lg_runtime.create_grad_scaler(False)

    best_accuracy = 0.0
    latest_accuracy = 0.0
    epoch_times: list[float] = []
    start = time.time()
    for epoch in range(1, actual_epochs + 1):
        epoch_start = time.time()
        epoch_lr = optimizer.param_groups[0]["lr"]
        loss, train_accuracy = train_one_epoch(
            student,
            train_loader,
            optimizer,
            scaler,
            device,
        )
        latest_accuracy = lg_runtime.evaluate(
            student,
            test_loader,
            device,
            False,
        )
        epoch_seconds = time.time() - epoch_start
        epoch_times.append(epoch_seconds)
        saved_best = latest_accuracy >= best_accuracy
        best_accuracy = max(best_accuracy, latest_accuracy)
        payload = {
            "model": student.state_dict(),
            "epoch": epoch,
            "accuracy": latest_accuracy,
            "best_accuracy": best_accuracy,
            "method": "Vanilla",
            "dataset": "cub200",
            "student": "deit_ti",
            "student_pretrained": bool(args.student_pretrained),
            "student_pretrained_source": (
                lg_runtime.STUDENT_PRETRAINED_SOURCES["deit_ti"]
                if args.student_pretrained
                else None
            ),
            "input_resolution": 224,
            "profile": args.profile,
            "batch_size": batch_size,
            "loss": "cross_entropy_only",
            "mode": mode,
        }
        lg_runtime.atomic_torch_save(payload, latest_path)
        if saved_best:
            lg_runtime.atomic_torch_save(payload, best_path)
        elapsed = time.time() - start
        average_epoch = sum(epoch_times) / len(epoch_times)
        summary = {
            "status": "complete" if epoch == actual_epochs else "running",
            "mode": mode,
            "method": "Vanilla",
            "dataset": "cub200",
            "profile": args.profile,
            "student": "deit_ti",
            "student_pretrained": bool(args.student_pretrained),
            "student_pretrained_source": (
                lg_runtime.STUDENT_PRETRAINED_SOURCES["deit_ti"]
                if args.student_pretrained
                else None
            ),
            "teacher": None,
            "loss": "cross_entropy_only",
            "input_resolution": 224,
            "batch_size": batch_size,
            "completed_epoch": epoch,
            "actual_epochs": actual_epochs,
            "planned_epochs": args.student_epochs,
            "latest_top1": latest_accuracy,
            "best_top1": best_accuracy,
            "avg_epoch_seconds": average_epoch,
            "estimated_planned_seconds": average_epoch * args.student_epochs,
            "estimated_planned_human": lg_runtime.format_duration(
                average_epoch * args.student_epochs
            ),
            "elapsed_seconds": elapsed,
            "paths": {
                "best": str(best_path.resolve()),
                "latest": str(latest_path.resolve()),
            },
        }
        atomic_json_save(summary, summary_path)
        lg_runtime.log(
            f"[VANILLA][{args.profile}][{epoch:03d}/{actual_epochs:03d}] "
            f"loss={loss:.4f} train_acc={train_accuracy:.2f}% "
            f"val_acc={latest_accuracy:.2f}% best={best_accuracy:.2f}% "
            f"lr={epoch_lr:.8g} time={epoch_seconds:.1f}s "
            f"avg_epoch={average_epoch:.1f}s "
            f"est_planned={summary['estimated_planned_human']}"
            + (" saved_best" if saved_best else "")
        )
        scheduler.step(epoch)

    lg_runtime.log("=" * 80)
    lg_runtime.log(
        f"[FINAL_RESULT] vanilla_profile={args.profile} "
        f"best_top1={best_accuracy:.2f}%"
    )
    lg_runtime.log(
        f"[TIMING] avg_epoch={summary['avg_epoch_seconds']:.1f}s "
        f"planned_epochs={args.student_epochs} "
        f"estimated_total={summary['estimated_planned_human']}"
    )
    lg_runtime.log(f"[FINAL_RESULT] summary={summary_path.resolve()}")
    lg_runtime.log("[DONE] Vanilla CE-only training completed successfully.")


def main() -> None:
    try:
        train(parse_args())
    except Exception as error:
        lg_runtime.log(f"[FATAL] {type(error).__name__}: {error}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
