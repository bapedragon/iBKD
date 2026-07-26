#!/usr/bin/env python3
"""Time the complete 36-task CUB-200 Table-1 experiment matrix."""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.metadata
import json
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from methods.table1_cub200.backbones import BACKBONES  # noqa: E402
from teachers.verify_checkpoints import DEFAULT_CHECKPOINT_ROOT  # noqa: E402


PLANNED_EPOCHS = 300
POD_LIMIT_SECONDS = 600 * 60
REQUIRED_IMPORTS = ("timm", "einops", "fvcore", "iopath", "yacs")


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


def bootstrap_dependencies() -> None:
    missing: list[str] = []
    for module_name in REQUIRED_IMPORTS:
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError:
            missing.append(module_name)
    if not missing:
        import timm

        if timm.__version__ == "1.0.27":
            log("[BOOT] required dependencies already available")
            return
        missing.append(f"timm=={timm.__version__}")
    requirements = REPOSITORY_ROOT / "requirements.txt"
    log(
        f"[BOOT] installing pinned requirements; missing_or_mismatched={missing}"
    )
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "-r",
            str(requirements),
        ],
        cwd=REPOSITORY_ROOT,
    )
    importlib.invalidate_caches()
    installed_timm = importlib.metadata.version("timm")
    if installed_timm != "1.0.27":
        raise RuntimeError(
            f"Dependency bootstrap expected timm=1.0.27, found {installed_timm}"
        )
    log("[BOOT] pinned requirements installed successfully")


@dataclass(frozen=True)
class Task:
    index: int
    kind: str
    student_key: str | None
    method: str
    batch_size: int
    run_name: str
    command: tuple[str, ...]
    summary_path: Path

    @property
    def label(self) -> str:
        if self.kind == "teacher":
            return "Teacher/ResNet56-32/b128"
        assert self.student_key is not None
        display = BACKBONES[self.student_key].display_name
        return f"{display}/{self.method.upper()}/b{self.batch_size}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing-run", action="store_true", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("./data/cub200"))
    parser.add_argument(
        "--teacher-root",
        type=Path,
        default=DEFAULT_CHECKPOINT_ROOT,
        help=(
            "Completed ResNet56-32 teacher used by all student timing tasks. "
            "The separate two-epoch teacher timing output is never used as guidance."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./outputs/table1_cub200_timing"),
    )
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def build_tasks(args: argparse.Namespace) -> list[Task]:
    output_root = args.output_dir
    tasks: list[Task] = []
    teacher_run_name = "table1_cub200_teacher_resnet56_32_b128_timing_2ep_seed1"
    teacher_output = output_root / "teacher"
    teacher_summary = teacher_output / teacher_run_name / "summary.json"
    teacher_command = (
        sys.executable,
        str(REPOSITORY_ROOT / "teachers" / "train_teacher_cub200.py"),
        "--timing-run",
        "--data-dir",
        str(args.data_dir),
        "--output-dir",
        str(teacher_output),
        "--run-name",
        teacher_run_name,
        "--num-workers",
        str(args.num_workers),
    )
    tasks.append(
        Task(
            index=1,
            kind="teacher",
            student_key=None,
            method="teacher",
            batch_size=128,
            run_name=teacher_run_name,
            command=teacher_command,
            summary_path=teacher_summary,
        )
    )

    combinations = (
        ("vanilla", 128),
        ("lg", 128),
        ("alg", 128),
        ("ours", 64),
        ("ours", 128),
    )
    index = 2
    for student_key in BACKBONES:
        for method, batch_size in combinations:
            run_name = (
                f"table1_cub200_{student_key}_{method}_"
                f"b{batch_size}_timing_2ep_seed1"
            )
            student_output = output_root / "students"
            summary_path = student_output / run_name / "summary.json"
            command = (
                sys.executable,
                str(
                    REPOSITORY_ROOT
                    / "methods"
                    / "table1_cub200"
                    / "train.py"
                ),
                "--timing-run",
                "--student",
                student_key,
                "--method",
                method,
                "--batch-size",
                str(batch_size),
                "--data-dir",
                str(args.data_dir),
                "--teacher-root",
                str(args.teacher_root),
                "--output-dir",
                str(student_output),
                "--run-name",
                run_name,
                "--num-workers",
                str(args.num_workers),
            )
            tasks.append(
                Task(
                    index=index,
                    kind="student",
                    student_key=student_key,
                    method=method,
                    batch_size=batch_size,
                    run_name=run_name,
                    command=command,
                    summary_path=summary_path,
                )
            )
            index += 1
    if len(tasks) != 36:
        raise AssertionError(f"Expected 36 timing tasks, built {len(tasks)}")
    return tasks


def load_summary(task: Task) -> dict[str, Any]:
    if not task.summary_path.is_file():
        raise RuntimeError(
            f"{task.label} completed without summary {task.summary_path}"
        )
    payload = json.loads(task.summary_path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise RuntimeError(f"{task.label} summary is not complete")
    return payload


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = (
        "index",
        "kind",
        "student",
        "method",
        "batch_size",
        "avg_epoch_seconds",
        "estimated_planned_seconds",
        "estimated_planned_human",
        "status",
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def run(args: argparse.Namespace) -> None:
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    bootstrap_dependencies()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(args)
    status_path = args.output_dir / "sequence_status.json"
    summary_path = args.output_dir / "timing_summary.json"
    csv_path = args.output_dir / "timing_summary.csv"
    rows: list[dict[str, Any]] = []
    suite_start = time.time()

    log("=" * 96)
    log("CUB-200 TABLE-1 COMPLETE TIMING SUITE")
    log("=" * 96)
    log(
        "[TASK_COUNT] requested_previous=29 corrected_with_ours_b64_b128=36 "
        "formula=1_teacher+7_backbones*5_student_settings"
    )
    log(
        "[MATRIX] per_backbone=Vanilla128,LG128,ALG128,Ours64,Ours128 "
        "planned_epochs_each=300 timing_epochs_each=2"
    )
    log(
        "[TEACHER_POLICY] teacher timing is measured separately; all student "
        "tasks load the completed shared ResNet56-32 checkpoint, never the "
        "two-epoch timing checkpoint."
    )

    for task in tasks:
        log("-" * 96)
        log(f"[TASK_START] {task.index:02d}/36 {task.label}")
        task_start = time.time()
        status = {
            "status": "running",
            "completed_tasks": len(rows),
            "total_tasks": len(tasks),
            "active_task": task.label,
            "active_index": task.index,
            "rows": rows,
        }
        atomic_json_save(status, status_path)
        try:
            subprocess.run(task.command, cwd=REPOSITORY_ROOT, check=True)
            payload = load_summary(task)
        except Exception as error:
            atomic_json_save(
                {
                    "status": "failed",
                    "completed_tasks": len(rows),
                    "total_tasks": len(tasks),
                    "failed_task": task.label,
                    "failed_index": task.index,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "rows": rows,
                },
                status_path,
            )
            log(
                f"[TASK_FAILED] {task.index:02d}/36 {task.label} "
                f"completed_before_failure={len(rows)}"
            )
            raise
        estimated = float(payload["estimated_planned_seconds"])
        average_epoch = float(payload["avg_epoch_seconds"])
        row = {
            "index": task.index,
            "kind": task.kind,
            "student": (
                "ResNet56-32"
                if task.student_key is None
                else BACKBONES[task.student_key].display_name
            ),
            "method": task.method,
            "batch_size": task.batch_size,
            "avg_epoch_seconds": average_epoch,
            "estimated_planned_seconds": estimated,
            "estimated_planned_human": format_duration(estimated),
            "status": "complete",
        }
        rows.append(row)
        write_csv(rows, csv_path)
        running_estimate = sum(
            float(item["estimated_planned_seconds"]) for item in rows
        )
        log(
            f"[TASK_DONE] {task.index:02d}/36 {task.label} "
            f"avg_epoch={average_epoch:.2f}s "
            f"estimated_300={format_duration(estimated)} "
            f"timing_elapsed={format_duration(time.time() - task_start)} "
            f"estimated_completed_subset={format_duration(running_estimate)}"
        )

    teacher_estimate = float(rows[0]["estimated_planned_seconds"])
    student_estimate = sum(
        float(item["estimated_planned_seconds"]) for item in rows[1:]
    )
    total_estimate = teacher_estimate + student_estimate
    pod_pass = total_estimate <= POD_LIMIT_SECONDS
    suite_elapsed = time.time() - suite_start
    final_summary = {
        "status": "complete",
        "task_count": len(rows),
        "formula": "1 teacher + 7 backbones * 5 settings = 36",
        "student_settings": [
            "Vanilla-b128",
            "LG-b128",
            "ALG-b128",
            "Ours-b64",
            "Ours-b128",
        ],
        "planned_epochs_per_task": PLANNED_EPOCHS,
        "timing_epochs_per_task": 2,
        "teacher_estimated_seconds": teacher_estimate,
        "teacher_estimated_human": format_duration(teacher_estimate),
        "students_only_estimated_seconds": student_estimate,
        "students_only_estimated_human": format_duration(student_estimate),
        "total_with_teacher_estimated_seconds": total_estimate,
        "total_with_teacher_estimated_human": format_duration(total_estimate),
        "reuse_existing_teacher_estimated_seconds": student_estimate,
        "reuse_existing_teacher_estimated_human": format_duration(student_estimate),
        "pod_limit_seconds": POD_LIMIT_SECONDS,
        "pod_limit_minutes": 600,
        "pod_limit_status": "PASS" if pod_pass else "FAIL",
        "timing_suite_elapsed_seconds": suite_elapsed,
        "timing_suite_elapsed_human": format_duration(suite_elapsed),
        "rows": rows,
        "paths": {
            "csv": str(csv_path.resolve()),
            "status": str(status_path.resolve()),
        },
    }
    atomic_json_save(final_summary, summary_path)
    atomic_json_save(
        {
            "status": "complete",
            "completed_tasks": len(rows),
            "total_tasks": len(tasks),
            "summary": str(summary_path.resolve()),
        },
        status_path,
    )

    log("=" * 96)
    log("[FINAL_TIMING_TABLE]")
    for row in rows:
        log(
            f"  {int(row['index']):02d}. {row['student']} / "
            f"{str(row['method']).upper()} / b{row['batch_size']}: "
            f"{row['estimated_planned_human']}"
        )
    log(
        f"[FINAL_TOTAL_ESTIMATE] tasks=36 teacher={format_duration(teacher_estimate)} "
        f"students35={format_duration(student_estimate)} "
        f"with_teacher={format_duration(total_estimate)} "
        f"reuse_teacher={format_duration(student_estimate)}"
    )
    log(
        f"[POD_LIMIT_CHECK] limit=600m estimated="
        f"{total_estimate / 60:.1f}m status={'PASS' if pod_pass else 'FAIL'}"
    )
    log(f"[FINAL_RESULT] summary={summary_path.resolve()}")
    log("[SEQUENCE_DONE] completed_tasks=36/36")


def main() -> None:
    # Some Issue wrappers have historically injected whitespace-only argv.
    sys.argv = [value for value in sys.argv if value.strip()]
    try:
        run(parse_args())
    except Exception as error:
        log(f"[FATAL] {type(error).__name__}: {error}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
