#!/usr/bin/env python3
"""Run matched Ours v1 lambda=0.25 controls on Flowers-102 and Chaoyang."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
POD_LIMIT_SECONDS = 600 * 60
ACTIVE_LAMBDA = 0.25
REFERENCE_LAMBDA = 0.5


@dataclass(frozen=True)
class Task:
    dataset: str
    display_name: str
    script: Path
    reference_summary: Path
    reference_best_top1: float


TASKS = (
    Task(
        dataset="flowers102",
        display_name="Flowers-102",
        script=Path("methods/Ours/flowers102/train.py"),
        reference_summary=Path(
            "results/Ours/flowers102/"
            "researcher_sync_v1_300ep_seed1/run_summary.json"
        ),
        reference_best_top1=74.80891201821434,
    ),
    Task(
        dataset="chaoyang",
        display_name="Chaoyang",
        script=Path("methods/Ours/chaoyang/train.py"),
        reference_summary=Path(
            "results/Ours/chaoyang/"
            "cifar100_locked_b64_v1_300ep_seed1/run_summary.json"
        ),
        reference_best_top1=81.11266947171576,
    ),
)


LOCKED_ARGUMENTS: dict[str, Any] = {
    "student": "deit_ti",
    "batch_size": 64,
    "eval_batch_size": 200,
    "image_size": 224,
    "eval_resize_mode": "direct",
    "seed": 1,
    "lr": 0.0005,
    "min_lr": 0.000005,
    "weight_decay": 0.05,
    "warmup_epochs": 20,
    "warmup_factor": 0.001,
    "label_smoothing": 0.0,
    "drop_path_rate": 0.1,
    "beta_schedule": "alg",
    "beta_on": 2.5,
    "alg_threshold": -0.02,
    "alg_smoothing_window": 50,
    "alg_warmup_epochs": 20,
    "teacher_image_size": 32,
    "grid_resize_mode": "larger",
    "base_protocol": "lg_official",
    "feature_grid": 14,
    "num_heads": 4,
    "deform_kernel_size": 5,
    "amp": False,
}


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


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--timing-run",
        action="store_true",
        help="Run two full-data epochs per task with the 300-epoch schedule.",
    )
    mode.add_argument(
        "--full-run",
        action="store_true",
        help="Run both lambda=0.25 controls for 300 epochs.",
    )
    parser.add_argument("--flowers-data-dir", type=Path, default=Path("./data"))
    parser.add_argument(
        "--chaoyang-data-dir",
        type=Path,
        default=Path("/app/data/chaoyang"),
    )
    parser.add_argument(
        "--teacher-root",
        type=Path,
        default=REPOSITORY_ROOT / "teachers/checkpoints",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args(argv)


def values_match(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        return isinstance(actual, (int, float)) and math.isclose(
            float(actual), expected, rel_tol=0.0, abs_tol=1e-12
        )
    return actual == expected


def load_and_validate_reference(task: Task) -> dict[str, Any]:
    path = REPOSITORY_ROOT / task.reference_summary
    if not path.is_file():
        raise FileNotFoundError(f"Missing lambda=0.5 reference summary: {path}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    arguments = summary.get("args", {})
    checks = {
        "status": summary.get("status") == "complete",
        "method": summary.get("method") == "Ours",
        "dataset": summary.get("dataset") == task.dataset,
        "planned_epochs": summary.get("planned_epochs") == 300,
        "student_epochs": arguments.get("student_epochs") == 300,
        "reference_lambda": values_match(
            arguments.get("fusion_ratio"), REFERENCE_LAMBDA
        ),
        "reference_best_top1": values_match(
            summary.get("best_top1"), task.reference_best_top1
        ),
    }
    for name, expected in LOCKED_ARGUMENTS.items():
        checks[f"locked_{name}"] = values_match(arguments.get(name), expected)
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            f"{task.display_name} lambda=0.5 reference failed audit: "
            + ", ".join(failed)
        )
    teacher = summary.get("teacher", {})
    if not isinstance(teacher.get("sha256"), str) or not teacher["sha256"]:
        raise RuntimeError(
            f"{task.display_name} reference has no teacher SHA-256"
        )
    if not isinstance(summary.get("source_snippet_sha256"), str):
        raise RuntimeError(
            f"{task.display_name} reference has no Ours source SHA-256"
        )
    return summary


def run_name(task: Task, timing_run: bool) -> str:
    mode = "timing_2ep" if timing_run else "300ep"
    return f"ours_v1_{task.dataset}_lambda_0p25_b64_{mode}_seed1"


def protocol_name(task: Task) -> str:
    return f"{task.dataset}_deit_ti_ours_lambda_0p25_transfer_v1"


def build_command(
    task: Task,
    args: argparse.Namespace,
    output_root: Path,
) -> tuple[list[str], Path]:
    task_output = output_root / task.dataset
    name = run_name(task, args.timing_run)
    data_dir = (
        args.flowers_data_dir
        if task.dataset == "flowers102"
        else args.chaoyang_data_dir
    )
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / task.script),
        "--data-dir",
        str(data_dir),
        "--teacher-root",
        str(args.teacher_root),
        "--output-dir",
        str(task_output),
        "--run-name",
        name,
        "--protocol-name",
        protocol_name(task),
        "--student-epochs",
        "300",
        "--batch-size",
        "64",
        "--eval-batch-size",
        "200",
        "--image-size",
        "224",
        "--eval-resize-mode",
        "direct",
        "--flowers-split-policy",
        "trainval_test_best",
        "--seed",
        "1",
        "--lr",
        "0.0005",
        "--min-lr",
        "0.000005",
        "--weight-decay",
        "0.05",
        "--warmup-epochs",
        "20",
        "--warmup-factor",
        "0.001",
        "--label-smoothing",
        "0.0",
        "--drop-path-rate",
        "0.1",
        "--fusion-ratio",
        "0.25",
        "--beta-schedule",
        "alg",
        "--beta-on",
        "2.5",
        "--alg-threshold",
        "-0.02",
        "--alg-smoothing-window",
        "50",
        "--alg-warmup-epochs",
        "20",
        "--teacher-image-size",
        "32",
        "--grid-resize-mode",
        "larger",
        "--base-protocol",
        "lg_official",
        "--feature-grid",
        "14",
        "--num-heads",
        "4",
        "--deform-kernel-size",
        "5",
        "--no-student-pretrained",
        "--no-amp",
        "--num-workers",
        str(args.num_workers),
    ]
    if args.timing_run:
        command.append("--timing-run")
    return command, task_output / name


def validate_candidate(
    summary: dict[str, Any],
    task: Task,
    reference: dict[str, Any],
    *,
    timing_run: bool,
) -> None:
    arguments = summary.get("args", {})
    expected_epochs = 2 if timing_run else 300
    checks = {
        "status": summary.get("status") == "complete",
        "method": summary.get("method") == "Ours",
        "dataset": summary.get("dataset") == task.dataset,
        "planned_epochs": summary.get("planned_epochs") == 300,
        "student_epochs": arguments.get("student_epochs") == expected_epochs,
        "active_lambda": values_match(
            arguments.get("fusion_ratio"), ACTIVE_LAMBDA
        ),
        "source_sha256": summary.get("source_snippet_sha256")
        == reference.get("source_snippet_sha256"),
        "teacher_sha256": summary.get("teacher", {}).get("sha256")
        == reference.get("teacher", {}).get("sha256"),
    }
    for name, expected in LOCKED_ARGUMENTS.items():
        checks[f"locked_{name}"] = values_match(arguments.get(name), expected)
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            f"{task.display_name} lambda=0.25 result failed audit: "
            + ", ".join(failed)
        )
    for metric in ("best_top1", "latest_top1"):
        if not isinstance(summary.get(metric), (int, float)):
            raise RuntimeError(
                f"{task.display_name} result is missing numeric {metric}"
            )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")

    default_root = (
        Path("/tmp/ours_v1_lambda_0p25_flowers_chaoyang_timing")
        if args.timing_run
        else Path(
            "/app/output/"
            "ours_v1_lambda_0p25_flowers_chaoyang_300ep_seed1"
        )
    )
    output_root = (args.output_dir or default_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "sequence_status.json"
    summary_path = output_root / "sequence_summary.json"
    references: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    sequence_start = time.time()

    log("=" * 88)
    log("OURS V1 CROSS-DATASET LAMBDA TRANSFER: FLOWERS-102 + CHAOYANG")
    log("=" * 88)
    log(f"[MODE] timing_run={args.timing_run} full_run={args.full_run}")
    log("[SEQUENCE] Flowers-102 -> Chaoyang")
    log(
        "[PROTOCOL_LOCK] only_change=lambda reference_lambda=0.5 "
        "active_lambda=0.25"
    )
    log(
        "[PROTOCOL_LOCK] student=DeiT-Ti scratch input=224 "
        "train/eval_batch=64/200 epochs=300 seed=1 fp32=True"
    )
    log(
        "[PROTOCOL_LOCK] loss=CE+beta(e)*(lambda*L_fuse+"
        "(1-lambda)*L_align) active_feature_loss="
        "0.25*L_fuse+0.75*L_align"
    )
    log(f"[PATH] flowers_data_dir={args.flowers_data_dir.resolve()}")
    log(f"[PATH] chaoyang_data_dir={args.chaoyang_data_dir.resolve()}")
    log(f"[PATH] teacher_root={args.teacher_root.resolve()}")
    log(f"[PATH] output_root={output_root}")

    try:
        for task in TASKS:
            reference = load_and_validate_reference(task)
            references[task.dataset] = reference
            log(
                f"[REFERENCE_CHECK][{task.display_name}] status=PASS "
                f"lambda=0.5 best_top1={task.reference_best_top1:.2f}% "
                f"teacher_sha256={reference['teacher']['sha256']} "
                f"source_sha256={reference['source_snippet_sha256']}"
            )

        for order, task in enumerate(TASKS, start=1):
            reference = references[task.dataset]
            command, run_dir = build_command(task, args, output_root)
            record: dict[str, Any] = {
                "order": order,
                "dataset": task.dataset,
                "display_name": task.display_name,
                "lambda": ACTIVE_LAMBDA,
                "reference_lambda": REFERENCE_LAMBDA,
                "reference_best_top1": task.reference_best_top1,
                "reference_summary": str(
                    REPOSITORY_ROOT / task.reference_summary
                ),
                "status": "running",
                "run_dir": str(run_dir),
                "command": command,
            }
            records.append(record)
            atomic_json(
                status_path,
                {
                    "status": "running",
                    "mode": "timing" if args.timing_run else "full",
                    "records": records,
                },
            )
            log("=" * 88)
            log(
                f"[SEQUENCE][{order}/{len(TASKS)}] START "
                f"dataset={task.display_name} lambda=0.25"
            )
            log(
                f"[SEQUENCE][{order}/{len(TASKS)}] "
                f"command={' '.join(command)}"
            )
            task_start = time.time()
            subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)
            record["elapsed_seconds"] = time.time() - task_start

            task_summary_path = run_dir / "summary.json"
            if not task_summary_path.is_file():
                raise FileNotFoundError(
                    f"Completed task has no summary: {task_summary_path}"
                )
            task_summary = json.loads(
                task_summary_path.read_text(encoding="utf-8")
            )
            validate_candidate(
                task_summary,
                task,
                reference,
                timing_run=args.timing_run,
            )
            best_top1 = float(task_summary["best_top1"])
            latest_top1 = float(task_summary["latest_top1"])
            record.update(
                {
                    "status": "complete",
                    "summary": str(task_summary_path),
                    "best_top1": best_top1,
                    "latest_top1": latest_top1,
                    "delta_vs_lambda_0p5_pp": (
                        None
                        if args.timing_run
                        else best_top1 - task.reference_best_top1
                    ),
                    "guidance_stop_epoch": task_summary.get(
                        "guidance_controller", {}
                    ).get("stop_epoch"),
                    "avg_epoch_seconds": task_summary.get(
                        "avg_epoch_seconds"
                    ),
                    "estimated_planned_seconds": task_summary.get(
                        "estimated_planned_seconds"
                    ),
                    "estimated_planned_human": task_summary.get(
                        "estimated_planned_human"
                    ),
                }
            )
            atomic_json(
                status_path,
                {
                    "status": (
                        "complete" if order == len(TASKS) else "running"
                    ),
                    "mode": "timing" if args.timing_run else "full",
                    "records": records,
                },
            )
            log(
                f"[SEQUENCE][{order}/{len(TASKS)}] DONE "
                f"dataset={task.display_name} lambda=0.25 "
                f"best_top1={best_top1:.2f}% latest_top1={latest_top1:.2f}%"
            )
    except Exception as error:
        if records and records[-1]["status"] == "running":
            records[-1]["status"] = "failed"
            records[-1]["error"] = f"{type(error).__name__}: {error}"
        atomic_json(
            status_path,
            {
                "status": "failed",
                "mode": "timing" if args.timing_run else "full",
                "records": records,
            },
        )
        raise

    elapsed_seconds = time.time() - sequence_start
    estimated_full_seconds = sum(
        float(record.get("estimated_planned_seconds") or 0.0)
        for record in records
    )
    pod_limit_passed = estimated_full_seconds < POD_LIMIT_SECONDS
    payload = {
        "status": "complete",
        "mode": "timing" if args.timing_run else "full",
        "protocol": "ours_v1_lambda_0p25_flowers_chaoyang_transfer_v1",
        "planned_epochs_each": 300,
        "task_order": [task.dataset for task in TASKS],
        "only_change": "lambda",
        "reference_lambda": REFERENCE_LAMBDA,
        "active_lambda": ACTIVE_LAMBDA,
        "completed_tasks": len(records),
        "total_tasks": len(TASKS),
        "elapsed_seconds": elapsed_seconds,
        "elapsed_human": format_duration(elapsed_seconds),
        "estimated_full_seconds": estimated_full_seconds,
        "estimated_full_human": format_duration(estimated_full_seconds),
        "pod_limit_seconds": POD_LIMIT_SECONDS,
        "pod_limit_passed": pod_limit_passed,
        "records": records,
    }
    atomic_json(status_path, payload)
    atomic_json(summary_path, payload)

    log("=" * 88)
    for record in records:
        comparison = (
            "timing_run_no_accuracy_comparison"
            if args.timing_run
            else (
                f"reference_lambda_0p5={record['reference_best_top1']:.2f}% "
                f"delta={record['delta_vs_lambda_0p5_pp']:+.2f}pp"
            )
        )
        log(
            f"[FINAL_RESULT][{record['display_name']}][lambda=0.25] "
            f"best_top1={record['best_top1']:.2f}% "
            f"latest_top1={record['latest_top1']:.2f}% {comparison}"
        )
    log(
        f"[POD_LIMIT_CHECK] status={'PASS' if pod_limit_passed else 'FAIL'} "
        f"limit=10h estimated={format_duration(estimated_full_seconds)}"
    )
    log(f"[SEQUENCE_DONE] completed_tasks={len(records)}/{len(TASKS)}")
    log(f"[FINAL_RESULT] sequence_summary={summary_path}")
    log("[DONE] Flowers-102 and Chaoyang lambda=0.25 tasks completed.")
    by_dataset = {record["dataset"]: record for record in records}
    log(
        "[FINAL_TOP1_SUMMARY_LAMBDA_0P25] "
        f"Flowers102={by_dataset['flowers102']['best_top1']:.2f}% "
        f"Chaoyang={by_dataset['chaoyang']['best_top1']:.2f}%"
    )


def cli_main() -> None:
    try:
        main()
    except Exception as error:
        log("=" * 88)
        log(f"[FATAL] {type(error).__name__}: {error}")
        traceback.print_exc()
        log("[FATAL] Cross-dataset lambda=0.25 sequence did not complete.")
        raise


if __name__ == "__main__":
    cli_main()
