#!/usr/bin/env python3
"""Train all five Table-1 CUB settings for one student backbone."""

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

from methods.table1_cub200.backbones import BACKBONES  # noqa: E402
from methods.table1_cub200.teacher_contract import (  # noqa: E402
    TABLE1_TEACHER_BUILD,
    TABLE1_TEACHER_ROOT,
    TABLE1_TEACHER_SHA256,
    validate_table1_teacher_spec,
)


PLANNED_EPOCHS = 300
POD_LIMIT_SECONDS = 600 * 60
REQUIRED_IMPORTS = ("timm", "einops", "fvcore", "iopath", "yacs")
COMBINATIONS = (
    ("Vanilla", "vanilla", 128),
    ("LG", "lg", 128),
    ("ALG", "alg", 128),
    ("OursB64", "ours", 64),
    ("OursB128", "ours", 128),
)
TIMING_ESTIMATES_SECONDS = {
    "deit_ti": 5 * 3600 + 51 * 60 + 42,
    "convit_ti": 6 * 3600 + 7 * 60 + 30,
    "cvt_13": 7 * 3600 + 40,
    "pit_ti": 6 * 3600 + 5 * 60 + 41,
    "pvtv2_b0": 6 * 3600 + 14 * 60 + 53,
    "t2t_vit_7": 5 * 3600 + 43 * 60 + 46,
    "t2t_vit_14": 6 * 3600 + 23 * 60 + 54,
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
    missing_or_mismatched: list[str] = []
    for module_name in REQUIRED_IMPORTS:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            missing_or_mismatched.append(module_name)
            continue
        if module_name == "timm" and module.__version__ != "1.0.27":
            missing_or_mismatched.append(f"timm=={module.__version__}")
    if not missing_or_mismatched:
        log("[BOOT] complete Table-1 runtime dependencies already available")
        return

    requirements = REPOSITORY_ROOT / "requirements.txt"
    log(
        "[BOOT] installing complete pinned Table-1 requirements; "
        f"missing_or_mismatched={missing_or_mismatched}"
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
    unresolved: list[str] = []
    for module_name in REQUIRED_IMPORTS:
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError:
            unresolved.append(module_name)
    if unresolved:
        raise RuntimeError(
            f"Table-1 dependency bootstrap incomplete: {unresolved}"
        )
    installed_timm = importlib.metadata.version("timm")
    if installed_timm != "1.0.27":
        raise RuntimeError(
            f"Expected timm=1.0.27 after bootstrap, found {installed_timm}"
        )
    log("[BOOT] complete pinned Table-1 requirements installed successfully")


@dataclass(frozen=True)
class Task:
    name: str
    method: str
    batch_size: int
    run_name: str
    command: tuple[str, ...]
    summary_path: Path
    checkpoint_path: Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-run", action="store_true", required=True)
    parser.add_argument("--student", choices=tuple(BACKBONES), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--teacher-root",
        type=Path,
        default=TABLE1_TEACHER_ROOT,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    raw_args = sys.argv[1:] if argv is None else argv
    return parser.parse_args(
        argument for argument in raw_args if argument.strip()
    )


def load_and_validate_teacher(teacher_root: Path) -> dict[str, Any]:
    manifest = json.loads(
        (teacher_root / "manifest.json").read_text(encoding="utf-8")
    )
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


def build_tasks(args: argparse.Namespace) -> list[Task]:
    students_output = args.output_dir / "students"
    tasks: list[Task] = []
    for name, method, batch_size in COMBINATIONS:
        run_name = (
            f"table1_cub200_{args.student}_{method}_"
            f"b{batch_size}_full_300ep_seed1"
        )
        command = (
            sys.executable,
            str(REPOSITORY_ROOT / "methods" / "table1_cub200" / "train.py"),
            "--full-run",
            "--student",
            args.student,
            "--method",
            method,
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
        run_dir = students_output / run_name
        tasks.append(
            Task(
                name=name,
                method=method,
                batch_size=batch_size,
                run_name=run_name,
                command=command,
                summary_path=run_dir / "summary.json",
                checkpoint_path=run_dir / "student_best.pt",
            )
        )
    return tasks


def validate_teacher_reference(
    used_teacher: dict[str, Any] | None,
    teacher_spec: dict[str, Any],
    *,
    vanilla: bool,
    mismatches: dict[str, Any],
) -> None:
    if vanilla:
        if used_teacher is not None:
            mismatches["teacher"] = {
                "expected": None,
                "actual": used_teacher,
            }
        return
    if not isinstance(used_teacher, dict):
        mismatches["teacher"] = {
            "expected": teacher_spec,
            "actual": used_teacher,
        }
        return
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


def validate_summary(
    task: Task,
    summary: dict[str, Any],
    *,
    student_key: str,
    teacher_spec: dict[str, Any],
) -> None:
    expected_optimizer = (
        "ours_single_group_all_parameters_decay_0.05"
        if task.method == "ours"
        else "official_lg_no_decay_exclusions"
    )
    expected = {
        "status": "complete",
        "mode": "full_300ep",
        "dataset": "cub200",
        "student_key": student_key,
        "student": BACKBONES[student_key].display_name,
        "method": task.method,
        "batch_size": task.batch_size,
        "actual_epochs": PLANNED_EPOCHS,
        "planned_epochs": PLANNED_EPOCHS,
        "optimizer_contract": expected_optimizer,
    }
    mismatches = {
        key: {"expected": value, "actual": summary.get(key)}
        for key, value in expected.items()
        if summary.get(key) != value
    }
    validate_teacher_reference(
        summary.get("teacher"),
        teacher_spec,
        vanilla=task.method == "vanilla",
        mismatches=mismatches,
    )
    if task.method == "ours":
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
    elif summary.get("ours_adapter") is not None:
        mismatches["ours_adapter"] = {
            "expected": None,
            "actual": summary.get("ours_adapter"),
        }
    if mismatches:
        raise RuntimeError(
            f"{task.name} summary validation failed: "
            f"{json.dumps(mismatches, sort_keys=True)}"
        )


def validate_checkpoint(
    task: Task,
    summary: dict[str, Any],
    *,
    student_key: str,
    teacher_spec: dict[str, Any],
) -> None:
    import torch

    payload = torch.load(
        task.checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    expected = {
        "student_key": student_key,
        "method": task.method,
        "batch_size": task.batch_size,
    }
    mismatches = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    validate_teacher_reference(
        payload.get("teacher"),
        teacher_spec,
        vanilla=task.method == "vanilla",
        mismatches=mismatches,
    )
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
    task: Task,
    phase: str,
    error: Exception,
) -> None:
    status.update(
        {
            "status": "failed",
            "failed_task": f"{task.name}:{phase}",
            "error": f"{type(error).__name__}: {error}",
            "finished_at_unix": time.time(),
        }
    )
    status["tasks"].append(
        {
            "name": task.name,
            "status": f"{phase}_failed",
            "summary": str(task.summary_path),
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
        tasks = build_tasks(args)
        expected_order = [name for name, _, _ in COMBINATIONS]
        if [task.name for task in tasks] != expected_order:
            raise AssertionError(f"Expected locked order {expected_order}")

        display_name = BACKBONES[args.student].display_name
        protocol = f"table1_cub200_{args.student}_all_five"
        status_path = args.output_dir / "sequence_status.json"
        status: dict[str, Any] = {
            "status": "running",
            "protocol": protocol,
            "student_key": args.student,
            "student": display_name,
            "completed_tasks": 0,
            "total_tasks": 5,
            "tasks": [],
            "started_at_unix": time.time(),
        }
        atomic_json_save(status, status_path)

        timing_estimate = TIMING_ESTIMATES_SECONDS[args.student]
        log("=" * 96)
        log(f"TABLE-1 CUB-200: {display_name.upper()} ALL FIVE SETTINGS")
        log("=" * 96)
        log(
            "[PROTOCOL_LOCK_TABLE1_CUB200_BACKBONE_ALL] "
            f"student={display_name} scratch input=224 epochs=300 "
            "order=VanillaB128,LG,ALG,OursB64,OursB128 "
            "teacher=Build543-ResNet56-scratch-32 guided_only=True "
            "seed=1 fp32=True"
        )
        log(
            "[FIXED_TEACHER_CHECK_TABLE1_CUB200] "
            f"status=PASS build={TABLE1_TEACHER_BUILD} "
            f"epoch={teacher_spec['epoch']} top1={teacher_spec['top1']:.2f}% "
            f"input={teacher_spec['input_resolution']} "
            f"sha256={teacher_spec['sha256']}"
        )
        log(
            "[POD_LIMIT_CHECK_TABLE1_CUB200_BACKBONE_ALL] "
            f"student={display_name} limit=600m "
            f"estimated={format_duration(timing_estimate)} "
            f"status={'PASS' if timing_estimate <= POD_LIMIT_SECONDS else 'FAIL'}"
        )

        summaries: dict[str, dict[str, Any]] = {}
        for index, task in enumerate(tasks, start=1):
            log("-" * 96)
            log(f"[TASK_START] {index}/5 {display_name}/{task.name}")
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
                    task=task,
                    phase="execution",
                    error=error,
                )
                raise
            elapsed = time.time() - started
            try:
                validate_summary(
                    task,
                    summary,
                    student_key=args.student,
                    teacher_spec=teacher_spec,
                )
                validate_checkpoint(
                    task,
                    summary,
                    student_key=args.student,
                    teacher_spec=teacher_spec,
                )
            except Exception as error:
                mark_failed(
                    status,
                    status_path,
                    task=task,
                    phase="validation",
                    error=error,
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
            teacher_label = (
                "none"
                if task.method == "vanilla"
                else teacher_spec["sha256"]
            )
            log(
                "[STUDENT_IDENTITY_CHECK_TABLE1_CUB200] "
                f"status=PASS student={display_name} method={task.name} "
                f"batch={task.batch_size} teacher={teacher_label}"
            )
            log(
                f"[TASK_DONE] {index}/5 {display_name}/{task.name} "
                f"elapsed={format_duration(elapsed)}"
            )

        best_top1 = {
            "Teacher": float(teacher_spec["top1"]),
            **{
                task.name: float(summaries[task.name]["best_top1"])
                for task in tasks
            },
        }
        final_summary = {
            "status": "complete",
            "protocol": protocol,
            "student_key": args.student,
            "student": display_name,
            "completed_tasks": 5,
            "total_tasks": 5,
            "teacher": teacher_spec,
            "best_top1": best_top1,
            "summaries": {
                task.name: str(task.summary_path) for task in tasks
            },
            "timing_estimate_seconds": timing_estimate,
            "timing_estimate_human": format_duration(timing_estimate),
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
            "[FINAL_TOP1_SUMMARY_TABLE1_CUB200_BACKBONE_ALL] "
            f"student={display_name} "
            f"Teacher={best_top1['Teacher']:.2f}% "
            f"Vanilla={best_top1['Vanilla']:.2f}% "
            f"LG={best_top1['LG']:.2f}% "
            f"ALG={best_top1['ALG']:.2f}% "
            f"OursB64={best_top1['OursB64']:.2f}% "
            f"OursB128={best_top1['OursB128']:.2f}%"
        )
        log(
            "[SEQUENCE_DONE_TABLE1_CUB200_BACKBONE_ALL] "
            f"student={display_name} completed_tasks=5/5"
        )
        log(f"[FINAL_RESULT] summary={final_summary_path.resolve()}")
    except Exception as error:
        log(f"[FATAL] {type(error).__name__}: {error}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
