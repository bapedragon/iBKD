#!/usr/bin/env python3
"""Run the paired CUB-200 experiment with both backbones from scratch."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
POD_LIMIT_SECONDS = 600 * 60
PROTOCOL_FAMILY = "cub200_resnet50_deit_ti_224_both_scratch_100ep"
TEACHER_EPOCHS = 200
STUDENT_EPOCHS = 100
TEACHER_RUN_NAME = "teacher_cub200_resnet50_224_scratch_200ep_seed1"
RUN_NAMES = {
    "VanillaB128": "vanilla_cub200_deit_ti_scratch_b128_100ep_seed1",
    "LG": "lg_cub200_both_scratch_b128_100ep_seed1",
    "ALG": "alg_cub200_both_scratch_b128_100ep_seed1",
    "OursB64": "ours_cub200_both_scratch_b64_100ep_seed1",
    "OursB128": "ours_cub200_both_scratch_b128_100ep_seed1",
}
METHOD_SCRIPTS = {
    "LG": "methods/LG/cub200_resnet50_224_scratch/train.py",
    "ALG": "methods/ALG/cub200_resnet50_224_scratch/train.py",
    "OursB64": "methods/Ours/cub200_resnet50_224_scratch/train.py",
    "OursB128": "methods/Ours/cub200_resnet50_224_scratch/train.py",
}
EXPECTED_BATCH_SIZE = {
    "VanillaB128": 128,
    "LG": 128,
    "ALG": 128,
    "OursB64": 64,
    "OursB128": 128,
}
FINAL_RESULT_ORDER = (
    "teacher",
    "VanillaB128",
    "LG",
    "ALG",
    "OursB64",
    "OursB128",
)


def log(message: str = "") -> None:
    print(message, flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--timing-run", action="store_true")
    modes.add_argument("--full-run", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=Path("./data/cub200"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/app/output/cub200_both_scratch_100ep_all_seed1"),
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-download", action="store_true")
    raw_args = sys.argv[1:] if argv is None else argv
    return parser.parse_args(
        argument for argument in raw_args if argument.strip()
    )


def atomic_json_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def collect_best_top1(
    summaries: dict[str, dict[str, Any]],
) -> dict[str, float]:
    results: dict[str, float] = {}
    for name in FINAL_RESULT_ORDER:
        if name not in summaries:
            raise KeyError(f"Missing completed summary for {name}")
        if "best_top1" not in summaries[name]:
            raise KeyError(f"Missing best_top1 in {name} summary")
        results[name] = float(summaries[name]["best_top1"])
    return results


def format_final_top1_summary(best_top1: dict[str, float]) -> str:
    return (
        "[FINAL_TOP1_SUMMARY_224_BOTH_SCRATCH_100EP] "
        f"Teacher={best_top1['teacher']:.2f}% "
        f"VanillaB128={best_top1['VanillaB128']:.2f}% "
        f"LG={best_top1['LG']:.2f}% "
        f"ALG={best_top1['ALG']:.2f}% "
        f"OursB64={best_top1['OursB64']:.2f}% "
        f"OursB128={best_top1['OursB128']:.2f}%"
    )


def validate_teacher_summary(summary: dict[str, Any]) -> None:
    expected = {
        "pretrained": False,
        "pretrained_source": None,
        "input_resolution": 224,
        "model_name": "resnet50_cub200_scratch_224",
        "protocol_family": "cub200_resnet50_224_scratch",
        "planned_epochs": TEACHER_EPOCHS,
    }
    mismatches = {
        key: {"expected": value, "actual": summary.get(key)}
        for key, value in expected.items()
        if summary.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "Scratch teacher identity validation failed: "
            f"{json.dumps(mismatches, sort_keys=True)}"
        )


def validate_student_summary(
    name: str,
    summary: dict[str, Any],
) -> None:
    expected = {
        "student_pretrained": False,
        "student_pretrained_source": None,
        "input_resolution": 224,
        "batch_size": EXPECTED_BATCH_SIZE[name],
        "planned_epochs": STUDENT_EPOCHS,
    }
    mismatches = {
        key: {"expected": value, "actual": summary.get(key)}
        for key, value in expected.items()
        if summary.get(key) != value
    }
    if name == "VanillaB128":
        if summary.get("teacher") is not None:
            mismatches["teacher"] = {
                "expected": None,
                "actual": summary.get("teacher"),
            }
    else:
        teacher = summary.get("teacher", {})
        if teacher.get("pretrained") is not False:
            mismatches["teacher.pretrained"] = {
                "expected": False,
                "actual": teacher.get("pretrained"),
            }
        if teacher.get("input_resolution") != 224:
            mismatches["teacher.input_resolution"] = {
                "expected": 224,
                "actual": teacher.get("input_resolution"),
            }
    if mismatches:
        raise RuntimeError(
            f"{name} both-scratch identity validation failed: "
            f"{json.dumps(mismatches, sort_keys=True)}"
        )


def run_tracked_task(
    *,
    name: str,
    command: list[str],
    summary_path: Path,
    status: dict[str, Any],
    status_path: Path,
) -> dict[str, Any]:
    log(f"[COMMAND][{name}] {' '.join(command)}")
    try:
        subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as error:
        status.update(
            {
                "status": "failed",
                "failed_task": name,
                "error": f"{type(error).__name__}: {error}",
                "finished_at_unix": time.time(),
            }
        )
        status["tasks"].append({"name": name, "status": "failed"})
        atomic_json_save(status, status_path)
        raise
    status["tasks"].append(
        {"name": name, "status": "complete", "summary": str(summary_path)}
    )
    status["completed_tasks"] += 1
    atomic_json_save(status, status_path)
    return summary


def main() -> None:
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    timing = bool(args.timing_run)
    mode = "timing" if timing else "full"
    suffix = "_timing_2ep" if timing else ""
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "sequence_status.json"
    teacher_name = TEACHER_RUN_NAME + suffix
    teacher_output = output_root / "teacher_resnet50_224_scratch"
    teacher_root = teacher_output / teacher_name
    status: dict[str, Any] = {
        "status": "running",
        "mode": mode,
        "protocol_family": PROTOCOL_FAMILY,
        "controlled_pair": {
            "reference": (
                "cub200_resnet50_deit_ti_224_both_imagenet_pretrained"
            ),
            "only_changed_factor": "teacher_and_student_initialization",
        },
        "separate_from": [
            "cub200_resnet56_32_scratch",
            "cub200_common_transfer_resnet50_224",
            "cub200_resnet50_224_scratch_300ep",
            "cub200_resnet50_deit_ti_224_both_imagenet_pretrained",
        ],
        "completed_tasks": 0,
        "total_tasks": 6,
        "tasks": [],
        "started_at_unix": time.time(),
        "protocols": {
            "teacher": "random-init ResNet50, 224px, 200 epochs",
            "VanillaB128": (
                "random-init DeiT-Ti, official-LG CE-only base, batch 128"
            ),
            "LG": (
                "random-init DeiT-Ti, official LG mechanics, batch 128"
            ),
            "ALG": (
                "random-init DeiT-Ti, ALG paper controller on official LG, "
                "batch 128"
            ),
            "OursB64": (
                "random-init DeiT-Ti, unchanged Ours protocol, batch 64"
            ),
            "OursB128": (
                "random-init DeiT-Ti, unchanged Ours except batch 128"
            ),
        },
    }
    atomic_json_save(status, status_path)

    log("=" * 104)
    log(
        "CUB-200 PAIRED SCRATCH: TEACHER -> VANILLA128 -> LG -> ALG -> "
        "OURS64 -> OURS128"
    )
    log("=" * 104)
    log(f"[MODE] {mode}")
    log(
        f"[PROTOCOL_FAMILY] {PROTOCOL_FAMILY} "
        "separate_from=both_imagenet_pretrained,resnet50_224_scratch_300ep,"
        "teacher_only_transfer,resnet56_32_scratch"
    )
    log(
        "[CONTROLLED_PAIR] "
        "reference=cub200_resnet50_deit_ti_224_both_imagenet_pretrained "
        "only_changed_factor=teacher_and_student_initialization"
    )
    log(
        "[RESOLUTION_LOCK] teacher_input=224 all_student_inputs=224 "
        "no_32px_teacher=True"
    )
    log(
        "[PRETRAINING_LOCK] teacher_pretrained=False "
        "all_students_pretrained=False "
        "teacher_source=none student_source=none"
    )
    log(
        f"[EPOCH_LOCK] teacher_planned_epochs={TEACHER_EPOCHS} "
        f"all_students_planned_epochs={STUDENT_EPOCHS}"
    )
    log(
        "[SEQUENCE] Teacher -> VanillaB128 -> LG -> ALG -> OursB64 -> "
        "OursB128"
    )
    log(
        "[METHOD_LOCK] LG=official_LG_mechanics "
        "ALG=paper_controller_on_official_LG "
        "OursB64=unchanged_current_Ours OursB128=batch_only_ablation"
    )
    log(
        "[FEATURE_CONTRACT] teacher=ResNet50 stages=(layer2,layer3,layer4) "
        "channels=(512,1024,2048) grids=(28,14,7)"
    )
    log(f"[PATH] data_dir={args.data_dir.resolve()}")
    log(f"[PATH] output_root={output_root}")

    teacher_command = [
        sys.executable,
        "teachers/train_teacher_cub200_resnet50_224.py",
        "--data-dir",
        str(args.data_dir),
        "--output-dir",
        str(teacher_output),
        "--run-name",
        teacher_name,
        "--initialization",
        "scratch",
        "--num-workers",
        str(args.num_workers),
    ]
    if timing:
        teacher_command.append("--timing-run")
    if args.no_download:
        teacher_command.append("--no-download")

    summaries: dict[str, dict[str, Any]] = {}
    summaries["teacher"] = run_tracked_task(
        name="teacher",
        command=teacher_command,
        summary_path=teacher_root / "summary.json",
        status=status,
        status_path=status_path,
    )
    validate_teacher_summary(summaries["teacher"])
    log(
        "[TEACHER_IDENTITY_CHECK] status=PASS pretrained=False "
        "source=none input=224"
    )

    task_names = ("VanillaB128", "LG", "ALG", "OursB64", "OursB128")
    for name in task_names:
        run_name = RUN_NAMES[name] + suffix
        if name == "VanillaB128":
            task_output = output_root / "Vanilla" / "scratch_b128"
            command = [
                sys.executable,
                "methods/Vanilla/cub200_224/train.py",
                "--profile",
                "lg_official_b128",
                "--no-student-pretrained",
                "--student-epochs",
                str(STUDENT_EPOCHS),
                "--timing-run" if timing else "--full-run",
                "--data-dir",
                str(args.data_dir),
                "--output-dir",
                str(task_output),
                "--run-name",
                run_name,
                "--num-workers",
                str(args.num_workers),
            ]
        else:
            task_output = output_root / name / "both_scratch_100ep"
            command = [
                sys.executable,
                METHOD_SCRIPTS[name],
                "--data-dir",
                str(args.data_dir),
                "--teacher-root",
                str(teacher_root),
                "--output-dir",
                str(task_output),
                "--run-name",
                run_name,
                "--num-workers",
                str(args.num_workers),
                "--no-student-pretrained",
                "--student-epochs",
                str(STUDENT_EPOCHS),
            ]
            if name in {"OursB64", "OursB128"}:
                command.extend(
                    ("--batch-size", str(EXPECTED_BATCH_SIZE[name]))
                )
            if timing:
                command.extend(
                    ("--timing-run", "--allow-teacher-runtime-gap")
                )
        summaries[name] = run_tracked_task(
            name=name,
            command=command,
            summary_path=task_output / run_name / "summary.json",
            status=status,
            status_path=status_path,
        )
        validate_student_summary(name, summaries[name])
        log(
            f"[STUDENT_IDENTITY_CHECK][{name}] status=PASS "
            "pretrained=False input=224 "
            f"batch={EXPECTED_BATCH_SIZE[name]}"
        )

    estimated_seconds = sum(
        float(summary["estimated_planned_seconds"])
        for summary in summaries.values()
    )
    best_top1 = collect_best_top1(summaries)
    margin = POD_LIMIT_SECONDS - estimated_seconds
    limit_status = "PASS" if margin > 0 else "FAIL"
    status.update(
        {
            "status": "complete",
            "best_top1": best_top1,
            "estimated_planned_seconds": estimated_seconds,
            "pod_limit_seconds": POD_LIMIT_SECONDS,
            "pod_limit_status": limit_status,
            "finished_at_unix": time.time(),
        }
    )
    atomic_json_save(status, status_path)
    log("=" * 104)
    log("[SEQUENCE_DONE_224_BOTH_SCRATCH_100EP] completed_tasks=6/6")
    log(
        f"[POD_LIMIT_CHECK_224_BOTH_SCRATCH_100EP] status={limit_status} "
        f"estimated_minutes={estimated_seconds / 60:.1f} "
        f"limit_minutes={POD_LIMIT_SECONDS / 60:.0f} "
        f"{'margin' if margin >= 0 else 'over_by'}_minutes="
        f"{abs(margin) / 60:.1f}"
    )
    if timing and limit_status == "FAIL":
        log(
            "[NEXT_ACTION_224_BOTH_SCRATCH_100EP] Split full tasks across "
            "Issues and reuse only a completed full teacher."
        )
    log(f"[FINAL_RESULT_224_BOTH_SCRATCH_100EP] sequence_status={status_path}")
    log(format_final_top1_summary(best_top1))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        log(
            f"[FATAL_224_BOTH_SCRATCH_100EP] "
            f"{type(error).__name__}: {error}"
        )
        raise
