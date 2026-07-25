#!/usr/bin/env python3
"""Run Ours v1 batch-128 lambda=0 and lambda=0.25 sequentially."""

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


REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
POD_LIMIT_SECONDS = 600 * 60


@dataclass(frozen=True)
class Task:
    lambda_value: float
    lambda_cli: str
    lambda_name: str


TASKS = (
    Task(lambda_value=0.0, lambda_cli="0", lambda_name="lambda_0"),
    Task(lambda_value=0.25, lambda_cli="0.25", lambda_name="lambda_0p25"),
)


def log(message: str = "") -> None:
    print(message, flush=True)


def format_duration(seconds: float) -> str:
    rounded = int(round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--timing-run", action="store_true")
    mode.add_argument("--full-run", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument(
        "--teacher-root",
        type=Path,
        default=REPOSITORY_ROOT / "teachers/checkpoints",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def build_command(
    task: Task,
    args: argparse.Namespace,
    output_root: Path,
) -> tuple[list[str], Path]:
    task_output = output_root / task.lambda_name
    mode_name = "timing_2ep" if args.timing_run else "300ep"
    run_name = (
        f"ours_v1_cifar100_batch128_{task.lambda_name}_"
        f"{mode_name}_seed1"
    )
    command = [
        sys.executable,
        str(Path(__file__).with_name("train.py")),
        "--lambda-value",
        task.lambda_cli,
        "--student-epochs",
        "300",
        "--batch-size",
        "128",
        "--data-dir",
        str(args.data_dir),
        "--teacher-root",
        str(args.teacher_root),
        "--output-dir",
        str(task_output),
        "--run-name",
        run_name,
        "--num-workers",
        str(args.num_workers),
    ]
    if args.timing_run:
        command.append("--timing-run")
    return command, task_output / run_name


def validate_summary(
    summary: dict[str, Any],
    task: Task,
    *,
    timing_run: bool,
) -> None:
    arguments = summary.get("args", {})
    expected_executed_epochs = 2 if timing_run else 300
    checks = {
        "method": summary.get("method") == "Ours",
        "batch_size": arguments.get("batch_size") == 128,
        "student_epochs": (
            arguments.get("student_epochs") == expected_executed_epochs
        ),
        "planned_epochs": summary.get("planned_epochs") == 300,
        "seed": arguments.get("seed") == 1,
        "fusion_ratio": isinstance(
            arguments.get("fusion_ratio"), (int, float)
        )
        and math.isclose(
            float(arguments["fusion_ratio"]),
            task.lambda_value,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            f"Completed summary failed protocol audit: {', '.join(failed)}"
        )


def main() -> None:
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")

    output_root = (
        args.output_dir
        or (
            Path("/tmp/ours_v1_cifar100_batch128_lambda_0_0p25_timing")
            if args.timing_run
            else Path(
                "/app/output/"
                "ours_v1_cifar100_batch128_lambda_0_0p25_300ep_seed1"
            )
        )
    ).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "sequence_status.json"
    summary_path = output_root / "sequence_summary.json"
    records: list[dict[str, Any]] = []
    sequence_start = time.time()

    log("=" * 80)
    log("OURS V1 CIFAR-100 BATCH 128: LAMBDA 0 -> LAMBDA 0.25")
    log("=" * 80)
    log(f"[MODE] timing_run={args.timing_run} full_run={args.full_run}")
    log("[SEQUENCE] lambda_0 -> lambda_0p25")
    log("[REFERENCE] batch128 lambda=0.5 best_top1=82.60%")
    log("[PROTOCOL_LOCK] only_change=lambda")
    log("[PROTOCOL_LOCK] epochs=300 train/eval_batch=128/200 seed=1 FP32")
    log("[PROTOCOL_LOCK] Ours v1 researcher_sync_v1 grids=32/16/14")
    log(f"[PATH] output_root={output_root}")

    try:
        for order, task in enumerate(TASKS, start=1):
            command, run_dir = build_command(task, args, output_root)
            record: dict[str, Any] = {
                "order": order,
                "lambda": task.lambda_value,
                "lambda_name": task.lambda_name,
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
            log("=" * 80)
            log(
                f"[SEQUENCE][{order}/{len(TASKS)}] START "
                f"lambda={task.lambda_value:g}"
            )
            task_start = time.time()
            subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)
            record["elapsed_seconds"] = time.time() - task_start

            task_summary_path = run_dir / "summary.json"
            if not task_summary_path.is_file():
                raise FileNotFoundError(
                    f"Completed task has no summary: {task_summary_path}"
                )
            task_summary = json.loads(task_summary_path.read_text(encoding="utf-8"))
            validate_summary(
                task_summary,
                task,
                timing_run=args.timing_run,
            )
            record.update(
                {
                    "status": "complete",
                    "summary": str(task_summary_path),
                    "best_top1": task_summary.get("best_top1"),
                    "latest_top1": task_summary.get("latest_top1"),
                    "avg_epoch_seconds": task_summary.get("avg_epoch_seconds"),
                    "estimated_planned_seconds": task_summary.get(
                        "estimated_planned_seconds"
                    ),
                    "estimated_planned_human": task_summary.get(
                        "estimated_planned_human"
                    ),
                }
            )
            log(
                f"[SEQUENCE][{order}/{len(TASKS)}] DONE "
                f"lambda={task.lambda_value:g} "
                f"best={record['best_top1']} latest={record['latest_top1']}"
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
        "protocol": "ours_v1_cifar100_batch128_lambda_0_0p25_v1",
        "reference_batch128_lambda_0p5_top1": 82.60,
        "planned_epochs_each": 300,
        "task_order": [task.lambda_value for task in TASKS],
        "only_change": "lambda",
        "elapsed_seconds": elapsed_seconds,
        "elapsed_human": format_duration(elapsed_seconds),
        "estimated_full_seconds": estimated_full_seconds,
        "estimated_full_human": format_duration(estimated_full_seconds),
        "pod_limit_seconds": POD_LIMIT_SECONDS,
        "pod_limit_passed": pod_limit_passed,
        "records": records,
    }
    atomic_json(status_path, {**payload, "records": records})
    atomic_json(summary_path, payload)

    log("=" * 80)
    for record in records:
        log(
            f"[FINAL_RESULT][lambda={float(record['lambda']):g}] "
            f"best_top1={record['best_top1']} "
            f"latest_top1={record['latest_top1']} "
            f"summary={record['summary']}"
        )
    log(
        f"[POD_LIMIT_CHECK] status={'PASS' if pod_limit_passed else 'FAIL'} "
        f"limit=10h estimated={format_duration(estimated_full_seconds)}"
    )
    log(f"[FINAL_RESULT] sequence_summary={summary_path}")
    log("[DONE] Ours v1 batch-128 lambda sequence completed successfully.")


def cli_main() -> None:
    try:
        main()
    except Exception as error:
        log("=" * 80)
        log(f"[FATAL] {type(error).__name__}: {error}")
        traceback.print_exc()
        log("[FATAL] Lambda sequence did not complete.")
        raise


if __name__ == "__main__":
    cli_main()
