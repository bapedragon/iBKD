#!/usr/bin/env python3
"""Train the Table-1 CUB ResNet56-32 teacher, then DeiT-Ti LG and ALG."""

from __future__ import annotations

import argparse
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


PLANNED_EPOCHS = 300
POD_LIMIT_SECONDS = 600 * 60
TIMING_ESTIMATE_SECONDS = (36 * 60 + 54) + (65 * 60 + 34) + (71 * 60 + 30)
REQUIRED_IMPORTS = ("timm", "einops", "fvcore", "iopath", "yacs")
TEACHER_RUN_NAME = (
    "table1_cub200_teacher_resnet56_32_b128_full_300ep_seed1"
)
LG_RUN_NAME = "table1_cub200_deit_ti_lg_b128_full_300ep_seed1"
ALG_RUN_NAME = "table1_cub200_deit_ti_alg_b128_full_300ep_seed1"


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
    name: str
    command: tuple[str, ...]
    summary_path: Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-run", action="store_true", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/app/output/table1_cub200_deit_lg_alg_full_seed1"),
    )
    parser.add_argument("--num-workers", type=int, default=4)
    raw_args = sys.argv[1:] if argv is None else argv
    return parser.parse_args(
        argument for argument in raw_args if argument.strip()
    )


def build_tasks(args: argparse.Namespace) -> tuple[list[Task], Path]:
    teacher_output = args.output_dir / "teacher"
    teacher_root = teacher_output / TEACHER_RUN_NAME
    students_output = args.output_dir / "students"

    teacher = Task(
        name="Teacher",
        command=(
            sys.executable,
            str(REPOSITORY_ROOT / "teachers" / "train_teacher_cub200.py"),
            "--data-dir",
            str(args.data_dir),
            "--output-dir",
            str(teacher_output),
            "--run-name",
            TEACHER_RUN_NAME,
            "--num-workers",
            str(args.num_workers),
        ),
        summary_path=teacher_root / "summary.json",
    )

    students: list[Task] = []
    for display_name, method, run_name in (
        ("LG", "lg", LG_RUN_NAME),
        ("ALG", "alg", ALG_RUN_NAME),
    ):
        students.append(
            Task(
                name=display_name,
                command=(
                    sys.executable,
                    str(
                        REPOSITORY_ROOT
                        / "methods"
                        / "table1_cub200"
                        / "train.py"
                    ),
                    "--full-run",
                    "--student",
                    "deit_ti",
                    "--method",
                    method,
                    "--batch-size",
                    "128",
                    "--data-dir",
                    str(args.data_dir),
                    "--teacher-root",
                    str(teacher_root),
                    "--output-dir",
                    str(students_output),
                    "--run-name",
                    run_name,
                    "--num-workers",
                    str(args.num_workers),
                ),
                summary_path=students_output / run_name / "summary.json",
            )
        )
    return [teacher, *students], teacher_root


def validate_teacher(
    summary: dict[str, Any],
    teacher_root: Path,
) -> dict[str, Any]:
    expected_summary = {
        "status": "complete",
        "mode": "full",
        "dataset": "cub200",
        "pretrained": False,
        "completed_epoch": PLANNED_EPOCHS,
        "planned_epochs": PLANNED_EPOCHS,
    }
    mismatches = {
        key: {"expected": value, "actual": summary.get(key)}
        for key, value in expected_summary.items()
        if summary.get(key) != value
    }
    manifest_path = teacher_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    teacher_spec = manifest["teachers"]["cub200"]
    expected_spec = {
        "selected_kind": "best",
        "num_classes": 200,
        "input_resolution": 32,
        "pretrained": False,
    }
    mismatches.update(
        {
            f"manifest.{key}": {"expected": value, "actual": teacher_spec.get(key)}
            for key, value in expected_spec.items()
            if teacher_spec.get(key) != value
        }
    )
    checkpoint_path = teacher_root / teacher_spec["checkpoint"]
    if not checkpoint_path.is_file():
        mismatches["manifest.checkpoint"] = {
            "expected": str(checkpoint_path),
            "actual": "missing",
        }
    if mismatches:
        raise RuntimeError(
            "Table-1 CUB teacher validation failed: "
            f"{json.dumps(mismatches, sort_keys=True)}"
        )
    return teacher_spec


def validate_student(
    name: str,
    summary: dict[str, Any],
    teacher_spec: dict[str, Any],
) -> None:
    expected = {
        "status": "complete",
        "mode": "full_300ep",
        "dataset": "cub200",
        "student_key": "deit_ti",
        "student": "DeiT-Ti",
        "method": name.lower(),
        "batch_size": 128,
        "actual_epochs": PLANNED_EPOCHS,
        "planned_epochs": PLANNED_EPOCHS,
    }
    mismatches = {
        key: {"expected": value, "actual": summary.get(key)}
        for key, value in expected.items()
        if summary.get(key) != value
    }
    used_teacher = summary.get("teacher", {})
    for key in ("checkpoint", "sha256", "input_resolution", "pretrained"):
        if used_teacher.get(key) != teacher_spec.get(key):
            mismatches[f"teacher.{key}"] = {
                "expected": teacher_spec.get(key),
                "actual": used_teacher.get(key),
            }
    if mismatches:
        raise RuntimeError(
            f"{name} Table-1 CUB validation failed: "
            f"{json.dumps(mismatches, sort_keys=True)}"
        )


def run_task(
    task: Task,
    *,
    index: int,
    status: dict[str, Any],
    status_path: Path,
) -> tuple[dict[str, Any], float]:
    log("-" * 96)
    log(f"[TASK_START] {index}/3 {task.name}")
    log(f"[COMMAND][{task.name}] {' '.join(task.command)}")
    started = time.time()
    try:
        subprocess.run(task.command, cwd=REPOSITORY_ROOT, check=True)
        summary = json.loads(task.summary_path.read_text(encoding="utf-8"))
    except Exception as error:
        status.update(
            {
                "status": "failed",
                "failed_task": task.name,
                "error": f"{type(error).__name__}: {error}",
                "finished_at_unix": time.time(),
            }
        )
        status["tasks"].append(
            {
                "name": task.name,
                "status": "failed",
                "summary": str(task.summary_path),
            }
        )
        atomic_json_save(status, status_path)
        raise

    elapsed = time.time() - started
    return summary, elapsed


def main() -> None:
    try:
        args = parse_args()
        if args.num_workers < 0:
            raise ValueError("--num-workers must be non-negative")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        bootstrap_dependencies()
        tasks, teacher_root = build_tasks(args)
        if [task.name for task in tasks] != ["Teacher", "LG", "ALG"]:
            raise AssertionError("Expected the locked Teacher -> LG -> ALG sequence")

        status_path = args.output_dir / "sequence_status.json"
        status: dict[str, Any] = {
            "status": "running",
            "protocol": "table1_cub200_resnet56_32_deit_ti_lg_alg",
            "completed_tasks": 0,
            "total_tasks": 3,
            "tasks": [],
            "started_at_unix": time.time(),
        }
        atomic_json_save(status, status_path)

        log("=" * 96)
        log("TABLE-1 CUB-200: RESNET56-32 TEACHER -> DEIT-TI LG -> DEIT-TI ALG")
        log("=" * 96)
        log(
            "[PROTOCOL_LOCK_TABLE1_CUB200_DEIT_LG_ALG] "
            "teacher=ResNet56 scratch input=32 batch=128 epochs=300 "
            "students=DeiT-Ti scratch input=224 methods=LG,ALG "
            "batch=128 epochs=300 seed=1 fp32=True"
        )
        log(
            "[TASK_COUNT_TABLE1_CUB200_DEIT_LG_ALG] "
            "total=3 order=Teacher,LG,ALG"
        )
        log(
            "[POD_LIMIT_CHECK_TABLE1_CUB200_DEIT_LG_ALG] "
            f"limit=600m estimated={format_duration(TIMING_ESTIMATE_SECONDS)} "
            f"status={'PASS' if TIMING_ESTIMATE_SECONDS <= POD_LIMIT_SECONDS else 'FAIL'}"
        )

        summaries: dict[str, dict[str, Any]] = {}
        teacher_spec: dict[str, Any] | None = None
        for index, task in enumerate(tasks, start=1):
            summary, elapsed = run_task(
                task,
                index=index,
                status=status,
                status_path=status_path,
            )
            try:
                if task.name == "Teacher":
                    teacher_spec = validate_teacher(summary, teacher_root)
                    log(
                        "[TEACHER_IDENTITY_CHECK_TABLE1_CUB200] "
                        "status=PASS model=ResNet56 pretrained=False input=32 "
                        f"sha256={teacher_spec['sha256']}"
                    )
                else:
                    assert teacher_spec is not None
                    validate_student(task.name, summary, teacher_spec)
                    log(
                        f"[STUDENT_IDENTITY_CHECK_TABLE1_CUB200][{task.name}] "
                        "status=PASS student=DeiT-Ti pretrained=False input=224 "
                        "batch=128 teacher_input=32"
                    )
            except Exception as error:
                status.update(
                    {
                        "status": "failed",
                        "failed_task": f"{task.name}:validation",
                        "error": f"{type(error).__name__}: {error}",
                        "finished_at_unix": time.time(),
                    }
                )
                status["tasks"].append(
                    {
                        "name": task.name,
                        "status": "validation_failed",
                        "summary": str(task.summary_path),
                    }
                )
                atomic_json_save(status, status_path)
                raise

            summaries[task.name] = summary
            status["completed_tasks"] += 1
            status["tasks"].append(
                {
                    "name": task.name,
                    "status": "complete",
                    "summary": str(task.summary_path),
                    "elapsed_seconds": elapsed,
                }
            )
            atomic_json_save(status, status_path)
            log(
                f"[TASK_DONE] {index}/3 {task.name} "
                f"elapsed={format_duration(elapsed)}"
            )

        teacher_top1 = float(summaries["Teacher"]["best_top1"])
        lg_top1 = float(summaries["LG"]["best_top1"])
        alg_top1 = float(summaries["ALG"]["best_top1"])
        final_summary = {
            "status": "complete",
            "protocol": "table1_cub200_resnet56_32_deit_ti_lg_alg",
            "completed_tasks": 3,
            "total_tasks": 3,
            "best_top1": {
                "Teacher": teacher_top1,
                "LG": lg_top1,
                "ALG": alg_top1,
            },
            "teacher_root": str(teacher_root),
            "summaries": {
                name: str(task.summary_path)
                for name, task in zip(("Teacher", "LG", "ALG"), tasks)
            },
            "timing_estimate_seconds": TIMING_ESTIMATE_SECONDS,
            "timing_estimate_human": format_duration(TIMING_ESTIMATE_SECONDS),
        }
        final_summary_path = args.output_dir / "final_summary.json"
        atomic_json_save(final_summary, final_summary_path)
        status.update(
            {
                "status": "complete",
                "finished_at_unix": time.time(),
                "final_summary": str(final_summary_path),
            }
        )
        atomic_json_save(status, status_path)

        log("=" * 96)
        log(
            "[FINAL_TOP1_SUMMARY_TABLE1_CUB200_DEIT_LG_ALG] "
            f"Teacher={teacher_top1:.2f}% "
            f"LG={lg_top1:.2f}% ALG={alg_top1:.2f}%"
        )
        log(
            "[SEQUENCE_DONE_TABLE1_CUB200_DEIT_LG_ALG] "
            "completed_tasks=3/3"
        )
        log(f"[FINAL_RESULT] summary={final_summary_path.resolve()}")
    except Exception as error:
        log(f"[FATAL] {type(error).__name__}: {error}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
