#!/usr/bin/env python3
"""Train Table-1 CUB DeiT-Ti Ours at batch 64 and batch 128."""

from __future__ import annotations

import argparse
import hashlib
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

from methods.table1_cub200.teacher_contract import (  # noqa: E402
    TABLE1_TEACHER_BUILD,
    TABLE1_TEACHER_ROOT,
    TABLE1_TEACHER_SHA256,
    validate_table1_teacher_spec,
)


PLANNED_EPOCHS = 300
POD_LIMIT_SECONDS = 600 * 60
TIMING_ESTIMATE_SECONDS = (74 * 60 + 39) + (69 * 60 + 17)
REQUIRED_IMPORTS = ("timm", "einops", "fvcore", "iopath", "yacs")
RUN_NAMES = {
    64: "table1_cub200_deit_ti_ours_b64_full_300ep_seed1",
    128: "table1_cub200_deit_ti_ours_b128_full_300ep_seed1",
}
COMPLETED_RESULT_SUMMARIES = {
    "LG": (
        REPOSITORY_ROOT
        / "results"
        / "LG"
        / "cub200"
        / "table1_cub200_deit_ti_lg_b128_full_300ep_seed1"
        / "run_summary.json"
    ),
    "ALG": (
        REPOSITORY_ROOT
        / "results"
        / "ALG"
        / "cub200"
        / "table1_cub200_deit_ti_alg_b128_full_300ep_seed1"
        / "run_summary.json"
    ),
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


def atomic_json_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    batch_size: int
    command: tuple[str, ...]
    summary_path: Path
    checkpoint_path: Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-run", action="store_true", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--teacher-root",
        type=Path,
        default=TABLE1_TEACHER_ROOT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/app/output/table1_cub200_deit_ours_full_seed1"),
    )
    parser.add_argument("--num-workers", type=int, default=4)
    raw_args = sys.argv[1:] if argv is None else argv
    return parser.parse_args(
        argument for argument in raw_args if argument.strip()
    )


def load_and_validate_teacher(teacher_root: Path) -> dict[str, Any]:
    manifest_path = teacher_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    teacher_spec = manifest["teachers"]["cub200"]
    validate_table1_teacher_spec(teacher_spec)
    checkpoint_path = teacher_root / teacher_spec["checkpoint"]
    actual_hash = sha256_file(checkpoint_path)
    if actual_hash != TABLE1_TEACHER_SHA256:
        raise RuntimeError(
            "CUB-200 Table-1 teacher checkpoint hash mismatch: "
            f"expected={TABLE1_TEACHER_SHA256} actual={actual_hash}"
        )
    return teacher_spec


def load_completed_results(
    teacher_spec: dict[str, Any],
) -> dict[str, float]:
    best_top1: dict[str, float] = {}
    for name, summary_path in COMPLETED_RESULT_SUMMARIES.items():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        expected = {
            "status": "complete",
            "mode": "full_300ep",
            "dataset": "cub200",
            "student_key": "deit_ti",
            "student": "DeiT-Ti",
            "method": name,
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
        for key in (
            "checkpoint",
            "sha256",
            "epoch",
            "top1",
            "num_classes",
            "input_resolution",
            "recipe_name",
            "pretrained",
        ):
            if used_teacher.get(key) != teacher_spec.get(key):
                mismatches[f"teacher.{key}"] = {
                    "expected": teacher_spec.get(key),
                    "actual": used_teacher.get(key),
                }
        if mismatches:
            raise RuntimeError(
                f"Completed {name} result validation failed: "
                f"{json.dumps(mismatches, sort_keys=True)}"
            )
        best_top1[name] = float(summary["best_top1"])
    return best_top1


def build_tasks(args: argparse.Namespace) -> list[Task]:
    students_output = args.output_dir / "students"
    tasks: list[Task] = []
    for batch_size in (64, 128):
        run_name = RUN_NAMES[batch_size]
        command = (
            sys.executable,
            str(REPOSITORY_ROOT / "methods" / "table1_cub200" / "train.py"),
            "--full-run",
            "--student",
            "deit_ti",
            "--method",
            "ours",
            "--batch-size",
            str(batch_size),
            "--data-dir",
            str(args.data_dir),
            "--teacher-root",
            str(args.teacher_root),
            "--output-dir",
            str(students_output),
            "--run-name",
            run_name,
            "--num-workers",
            str(args.num_workers),
        )
        tasks.append(
            Task(
                name=f"OursB{batch_size}",
                batch_size=batch_size,
                command=command,
                summary_path=students_output / run_name / "summary.json",
                checkpoint_path=students_output / run_name / "student_best.pt",
            )
        )
    return tasks


def validate_student_summary(
    task: Task,
    summary: dict[str, Any],
    teacher_spec: dict[str, Any],
) -> None:
    expected = {
        "status": "complete",
        "mode": "full_300ep",
        "dataset": "cub200",
        "student_key": "deit_ti",
        "student": "DeiT-Ti",
        "method": "ours",
        "batch_size": task.batch_size,
        "actual_epochs": PLANNED_EPOCHS,
        "planned_epochs": PLANNED_EPOCHS,
        "optimizer_contract": "ours_single_group_all_parameters_decay_0.05",
    }
    mismatches = {
        key: {"expected": value, "actual": summary.get(key)}
        for key, value in expected.items()
        if summary.get(key) != value
    }
    used_teacher = summary.get("teacher", {})
    for key in (
        "checkpoint",
        "sha256",
        "epoch",
        "top1",
        "num_classes",
        "input_resolution",
        "recipe_name",
        "pretrained",
    ):
        if used_teacher.get(key) != teacher_spec.get(key):
            mismatches[f"teacher.{key}"] = {
                "expected": teacher_spec.get(key),
                "actual": used_teacher.get(key),
            }
    adapter = summary.get("ours_adapter", {})
    expected_adapter = {
        "all_blocks": True,
        "output_channels": 192,
        "output_grid": 14,
        "batch": task.batch_size,
    }
    for key, value in expected_adapter.items():
        if adapter.get(key) != value:
            mismatches[f"ours_adapter.{key}"] = {
                "expected": value,
                "actual": adapter.get(key),
            }
    if mismatches:
        raise RuntimeError(
            f"{task.name} Table-1 CUB validation failed: "
            f"{json.dumps(mismatches, sort_keys=True)}"
        )


def validate_student_checkpoint(
    task: Task,
    summary: dict[str, Any],
    teacher_spec: dict[str, Any],
) -> None:
    import torch

    payload = torch.load(
        task.checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    expected = {
        "student_key": "deit_ti",
        "method": "ours",
        "batch_size": task.batch_size,
    }
    mismatches = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    used_teacher = payload.get("teacher", {})
    for key in (
        "checkpoint",
        "sha256",
        "epoch",
        "top1",
        "num_classes",
        "input_resolution",
        "recipe_name",
        "pretrained",
    ):
        if used_teacher.get(key) != teacher_spec.get(key):
            mismatches[f"teacher.{key}"] = {
                "expected": teacher_spec.get(key),
                "actual": used_teacher.get(key),
            }
    if abs(
        float(payload.get("best_accuracy", -1.0))
        - float(summary["best_top1"])
    ) > 1e-8:
        mismatches["best_accuracy"] = {
            "expected": float(summary["best_top1"]),
            "actual": payload.get("best_accuracy"),
        }
    if mismatches:
        raise RuntimeError(
            f"{task.name} checkpoint validation failed: "
            f"{json.dumps(mismatches, sort_keys=True)}"
        )


def mark_failed(
    status: dict[str, Any],
    status_path: Path,
    *,
    task_name: str,
    error: Exception,
    phase: str,
    summary_path: Path,
) -> None:
    status.update(
        {
            "status": "failed",
            "failed_task": f"{task_name}:{phase}",
            "error": f"{type(error).__name__}: {error}",
            "finished_at_unix": time.time(),
        }
    )
    status["tasks"].append(
        {
            "name": task_name,
            "status": f"{phase}_failed",
            "summary": str(summary_path),
        }
    )
    atomic_json_save(status, status_path)


def main() -> None:
    try:
        args = parse_args()
        if args.num_workers < 0:
            raise ValueError("--num-workers must be non-negative")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        bootstrap_dependencies()
        teacher_spec = load_and_validate_teacher(args.teacher_root)
        completed_results = load_completed_results(teacher_spec)
        tasks = build_tasks(args)
        if [task.name for task in tasks] != ["OursB64", "OursB128"]:
            raise AssertionError("Expected the locked OursB64 -> OursB128 sequence")

        status_path = args.output_dir / "sequence_status.json"
        status: dict[str, Any] = {
            "status": "running",
            "protocol": "table1_cub200_deit_ti_ours_b64_b128",
            "completed_tasks": 0,
            "total_tasks": 2,
            "tasks": [],
            "started_at_unix": time.time(),
        }
        atomic_json_save(status, status_path)

        log("=" * 96)
        log("TABLE-1 CUB-200: DEIT-TI OURS B64 -> DEIT-TI OURS B128")
        log("=" * 96)
        log(
            "[PROTOCOL_LOCK_TABLE1_CUB200_DEIT_OURS] "
            "teacher=ResNet56 scratch input=32 build=543 fixed=True "
            "student=DeiT-Ti scratch input=224 method=Ours "
            "batches=64,128 epochs=300 seed=1 fp32=True"
        )
        log(
            "[FIXED_TEACHER_CHECK_TABLE1_CUB200] "
            f"status=PASS build={TABLE1_TEACHER_BUILD} "
            f"epoch={teacher_spec['epoch']} top1={teacher_spec['top1']:.2f}% "
            f"input={teacher_spec['input_resolution']} "
            f"sha256={teacher_spec['sha256']}"
        )
        log(
            "[PREVIOUS_RESULT_CHECK_TABLE1_CUB200_DEIT] "
            f"status=PASS LG={completed_results['LG']:.2f}% "
            f"ALG={completed_results['ALG']:.2f}% "
            f"teacher_sha256={teacher_spec['sha256']}"
        )
        log(
            "[TASK_COUNT_TABLE1_CUB200_DEIT_OURS] "
            "total=2 order=OursB64,OursB128"
        )
        log(
            "[POD_LIMIT_CHECK_TABLE1_CUB200_DEIT_OURS] "
            f"limit=600m estimated={format_duration(TIMING_ESTIMATE_SECONDS)} "
            f"status={'PASS' if TIMING_ESTIMATE_SECONDS <= POD_LIMIT_SECONDS else 'FAIL'}"
        )

        summaries: dict[str, dict[str, Any]] = {}
        for index, task in enumerate(tasks, start=1):
            log("-" * 96)
            log(f"[TASK_START] {index}/2 {task.name}")
            log(f"[COMMAND][{task.name}] {' '.join(task.command)}")
            started = time.time()
            try:
                subprocess.run(task.command, cwd=REPOSITORY_ROOT, check=True)
                summary = json.loads(
                    task.summary_path.read_text(encoding="utf-8")
                )
            except Exception as error:
                mark_failed(
                    status,
                    status_path,
                    task_name=task.name,
                    error=error,
                    phase="execution",
                    summary_path=task.summary_path,
                )
                raise
            elapsed = time.time() - started
            try:
                validate_student_summary(task, summary, teacher_spec)
                validate_student_checkpoint(task, summary, teacher_spec)
            except Exception as error:
                mark_failed(
                    status,
                    status_path,
                    task_name=task.name,
                    error=error,
                    phase="validation",
                    summary_path=task.summary_path,
                )
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
                f"[STUDENT_IDENTITY_CHECK_TABLE1_CUB200][{task.name}] "
                "status=PASS student=DeiT-Ti pretrained=False input=224 "
                f"batch={task.batch_size} teacher_input=32 "
                f"teacher_sha256={teacher_spec['sha256']}"
            )
            log(
                f"[TASK_DONE] {index}/2 {task.name} "
                f"elapsed={format_duration(elapsed)}"
            )

        b64_top1 = float(summaries["OursB64"]["best_top1"])
        b128_top1 = float(summaries["OursB128"]["best_top1"])
        final_summary = {
            "status": "complete",
            "protocol": "table1_cub200_deit_ti_ours_b64_b128",
            "completed_tasks": 2,
            "total_tasks": 2,
            "teacher": teacher_spec,
            "best_top1": {
                "Teacher": float(teacher_spec["top1"]),
                "LG": completed_results["LG"],
                "ALG": completed_results["ALG"],
                "OursB64": b64_top1,
                "OursB128": b128_top1,
            },
            "summaries": {
                task.name: str(task.summary_path) for task in tasks
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
            "[FINAL_TOP1_SUMMARY_TABLE1_CUB200_DEIT] "
            f"Teacher={float(teacher_spec['top1']):.2f}% "
            f"LG={completed_results['LG']:.2f}% "
            f"ALG={completed_results['ALG']:.2f}% "
            f"OursB64={b64_top1:.2f}% OursB128={b128_top1:.2f}%"
        )
        log(
            "[SEQUENCE_DONE_TABLE1_CUB200_DEIT_OURS] "
            "completed_tasks=2/2"
        )
        log(f"[FINAL_RESULT] summary={final_summary_path.resolve()}")
    except Exception as error:
        log(f"[FATAL] {type(error).__name__}: {error}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
