#!/usr/bin/env python3
"""Audit and time canonical ALG on Flowers-102, Chaoyang, and CIFAR-100."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from methods.ALG.chaoyang.train import PROTOCOL_DEFAULTS as CHAOYANG_DEFAULTS
from methods.ALG.cifar100.train import PROTOCOL_DEFAULTS as CIFAR100_DEFAULTS
from methods.ALG.flowers102.train import PROTOCOL_DEFAULTS as FLOWERS102_DEFAULTS
from methods.LG.official_lg import OFFICIAL_LG_COMMIT
from methods.LG.runtime import ALG_PAPER_DOI, PAPER_GUIDANCE_STOP_EPOCH, REFERENCE_TOP1


@dataclass(frozen=True)
class TimingTask:
    dataset: str
    script: Path
    defaults: tuple[tuple[str, str], ...]
    data_arg: str
    run_name: str


TASKS = (
    TimingTask(
        dataset="flowers102",
        script=REPOSITORY_ROOT / "methods/ALG/flowers102/train.py",
        defaults=FLOWERS102_DEFAULTS,
        data_arg="flowers102_data_dir",
        run_name="alg_paper_official_lg_flowers102_timing_2ep",
    ),
    TimingTask(
        dataset="chaoyang",
        script=REPOSITORY_ROOT / "methods/ALG/chaoyang/train.py",
        defaults=CHAOYANG_DEFAULTS,
        data_arg="chaoyang_data_dir",
        run_name="alg_paper_official_lg_chaoyang_timing_2ep",
    ),
    TimingTask(
        dataset="cifar100",
        script=REPOSITORY_ROOT / "methods/ALG/cifar100/train.py",
        defaults=CIFAR100_DEFAULTS,
        data_arg="cifar100_data_dir",
        run_name="alg_paper_official_lg_cifar100_timing_2ep",
    ),
)

EXPECTED_SHARED_DEFAULTS = {
    "--student-epochs": "300",
    "--batch-size": "128",
    "--eval-batch-size": "200",
    "--lr": "0.0005",
    "--min-lr": "0.000005",
    "--weight-decay": "0.05",
    "--warmup-epochs": "20",
    "--warmup-factor": "0.001",
    "--label-smoothing": "0.0",
    "--drop-path-rate": "0.1",
    "--teacher-image-size": "32",
    "--beta": "2.5",
    "--alg-threshold": "-0.02",
    "--alg-smoothing-window": "50",
    "--alg-warmup-epochs": "0",
    "--alg-stop-comparison": "paper_ge",
    "--alg-derivative-mode": "paper_equations",
    "--base-protocol": "lg_official",
    "--eval-resize-mode": "direct",
    "--eval-interpolation": "bilinear",
    "--seed": "1",
}


def log(message: str) -> None:
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


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timing-run",
        action="store_true",
        help="Required safety flag. Each dataset runs for 2 epochs only.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--teacher-root",
        type=Path,
        default=REPOSITORY_ROOT / "teachers/checkpoints",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "outputs/alg_official_three_dataset_timing",
    )
    parser.add_argument(
        "--cifar100-data-dir",
        type=Path,
        default=REPOSITORY_ROOT / "data",
    )
    parser.add_argument(
        "--flowers102-data-dir",
        type=Path,
        default=REPOSITORY_ROOT / "data",
    )
    parser.add_argument(
        "--chaoyang-data-dir",
        type=Path,
        default=Path("/app/data/chaoyang"),
    )
    args = parser.parse_args()
    if not args.timing_run:
        parser.error("--timing-run is required; this runner must not launch full runs")
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    return args


def audit_wrapper(task: TimingTask) -> dict[str, str]:
    defaults = dict(task.defaults)
    mismatches = {
        option: f"expected={expected} actual={defaults.get(option)}"
        for option, expected in EXPECTED_SHARED_DEFAULTS.items()
        if defaults.get(option) != expected
    }
    expected_protocol = f"{task.dataset}_deit_ti_alg_paper_official_lg_v1"
    if defaults.get("--protocol-name") != expected_protocol:
        mismatches["--protocol-name"] = (
            f"expected={expected_protocol} "
            f"actual={defaults.get('--protocol-name')}"
        )
    if task.dataset == "flowers102":
        expected_split = "trainval_test_best"
        if defaults.get("--flowers-split-policy") != expected_split:
            mismatches["--flowers-split-policy"] = (
                f"expected={expected_split} "
                f"actual={defaults.get('--flowers-split-policy')}"
            )
    if mismatches:
        details = "; ".join(f"{key}: {value}" for key, value in mismatches.items())
        raise RuntimeError(f"{task.dataset} wrapper audit failed: {details}")
    return defaults


def audit_all() -> None:
    log("=" * 72)
    log("CANONICAL ALG / OFFICIAL LG BASE / THREE-DATASET TIMING")
    log("=" * 72)
    log(
        f"[SOURCE_LOCK] alg_paper_doi={ALG_PAPER_DOI} "
        f"official_lg_commit={OFFICIAL_LG_COMMIT}"
    )
    log(
        "[ISOLATION] method=ALG ours_module=False ours_loss=False "
        "ours_protocol=False"
    )
    log(
        "[PROTOCOL_LOCK] epochs=300 train_batch=128 eval_batch=200 "
        "optimizer=AdamW lr=0.0005 min_lr=0.000005 weight_decay=0.05 "
        "warmup=20 warmup_factor=0.001 fp32=True seed=1"
    )
    log(
        "[ALG_LOCK] loss=CE+2.5*LG_while_active tau=-0.02 window=50 "
        "controller_warmup=0 equations=ALG_10_to_19"
    )
    log(
        "[LG_LOCK] student=DeiT-Ti-224 teacher=ResNet56-32 "
        "teacher_stages=(0,1,2) student_blocks=(0,6,11) "
        "projection=1x1 alignment=larger_grid_bilinear "
        "train_interpolation=bicubic eval_interpolation=bilinear"
    )
    for task in TASKS:
        audit_wrapper(task)
        reference = REFERENCE_TOP1[task.dataset]["alg"]
        stop_epoch = PAPER_GUIDANCE_STOP_EPOCH[task.dataset]
        log(
            f"[WRAPPER_AUDIT] dataset={task.dataset} status=PASS "
            f"paper_deit_top1={reference:.2f}% "
            f"paper_guidance_stop_epoch={stop_epoch}"
        )


def data_dir_for(args: argparse.Namespace, task: TimingTask) -> Path:
    return Path(getattr(args, task.data_arg))


def validate_child_summary(task: TimingTask, summary: dict[str, Any]) -> None:
    public = summary.get("args", {})
    expected = {
        "method": "ALG",
        "dataset": task.dataset,
        "planned_epochs": 300,
        "student_epochs": 2,
        "official_lg_commit": OFFICIAL_LG_COMMIT,
        "alg_paper_doi": ALG_PAPER_DOI,
    }
    actual = {
        "method": summary.get("method"),
        "dataset": summary.get("dataset"),
        "planned_epochs": summary.get("planned_epochs"),
        "student_epochs": summary.get("student_epochs"),
        "official_lg_commit": summary.get("official_lg_commit"),
        "alg_paper_doi": summary.get("alg_paper_doi"),
    }
    mismatches = [
        f"{key}: expected={value!r} actual={actual.get(key)!r}"
        for key, value in expected.items()
        if actual.get(key) != value
    ]
    runtime_expected = {
        "batch_size": 128,
        "eval_batch_size": 200,
        "lr": 0.0005,
        "min_lr": 0.000005,
        "weight_decay": 0.05,
        "warmup_epochs": 20,
        "warmup_factor": 0.001,
        "label_smoothing": 0.0,
        "drop_path_rate": 0.1,
        "teacher_image_size": 32,
        "beta": 2.5,
        "alg_threshold": -0.02,
        "alg_smoothing_window": 50,
        "alg_warmup_epochs": 0,
        "alg_stop_comparison": "paper_ge",
        "alg_derivative_mode": "paper_equations",
        "base_protocol": "lg_official",
        "eval_resize_mode": "direct",
        "eval_interpolation": "bilinear",
        "seed": 1,
        "amp": False,
    }
    mismatches.extend(
        f"args.{key}: expected={value!r} actual={public.get(key)!r}"
        for key, value in runtime_expected.items()
        if public.get(key) != value
    )
    if task.dataset == "flowers102":
        if public.get("flowers_split_policy") != "trainval_test_best":
            mismatches.append(
                "args.flowers_split_policy: expected='trainval_test_best' "
                f"actual={public.get('flowers_split_policy')!r}"
            )
    if mismatches:
        raise RuntimeError(
            f"{task.dataset} runtime audit failed: " + "; ".join(mismatches)
        )


def run_task(
    args: argparse.Namespace,
    task: TimingTask,
) -> dict[str, Any]:
    data_dir = data_dir_for(args, task)
    command = [
        sys.executable,
        "-u",
        str(task.script),
        "--timing-run",
        "--num-workers",
        str(args.num_workers),
        "--data-dir",
        str(data_dir),
        "--teacher-root",
        str(args.teacher_root),
        "--output-dir",
        str(args.output_dir),
        "--run-name",
        task.run_name,
    ]
    log("-" * 72)
    log(f"[TIMING_START] dataset={task.dataset}")
    log(f"[TIMING_COMMAND] {' '.join(command)}")
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    started = time.time()
    result = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
    )
    wall_seconds = time.time() - started
    if result.returncode != 0:
        raise RuntimeError(
            f"{task.dataset} timing run failed with exit code {result.returncode}"
        )

    summary_path = args.output_dir / task.run_name / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"{task.dataset} summary was not created: {summary_path}"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    validate_child_summary(task, summary)
    average_epoch = float(summary["avg_epoch_seconds"])
    estimated_seconds = float(summary["estimated_planned_seconds"])
    record = {
        "dataset": task.dataset,
        "status": "complete",
        "paper_deit_top1": REFERENCE_TOP1[task.dataset]["alg"],
        "paper_guidance_stop_epoch": PAPER_GUIDANCE_STOP_EPOCH[task.dataset],
        "timing_top1": summary["best_top1"],
        "avg_epoch_seconds": average_epoch,
        "estimated_300_seconds": estimated_seconds,
        "estimated_300_human": summary["estimated_planned_human"],
        "timing_wall_seconds": wall_seconds,
        "timing_wall_human": format_duration(wall_seconds),
        "run_name": task.run_name,
        "summary": str(summary_path.resolve()),
    }
    log(
        f"[TIMING_RESULT] dataset={task.dataset} status=PASS "
        f"avg_epoch={average_epoch:.1f}s "
        f"estimated_300={summary['estimated_planned_human']} "
        f"timing_top1={summary['best_top1']:.2f}%"
    )
    return record


def main() -> None:
    args = parse_args()
    audit_all()
    if not args.teacher_root.is_dir():
        raise FileNotFoundError(f"Teacher root not found: {args.teacher_root}")
    if not args.chaoyang_data_dir.is_dir():
        raise FileNotFoundError(
            "Chaoyang mount not found. Expected the dataset under "
            f"{args.chaoyang_data_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = args.output_dir / "three_dataset_timing_summary.json"
    records: list[dict[str, Any]] = []
    started = time.time()
    try:
        for task in TASKS:
            records.append(run_task(args, task))
    except Exception as error:
        atomic_write_json(
            combined_path,
            {
                "status": "failed",
                "method": "ALG",
                "official_lg_commit": OFFICIAL_LG_COMMIT,
                "alg_paper_doi": ALG_PAPER_DOI,
                "completed": records,
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise

    estimated_total = sum(record["estimated_300_seconds"] for record in records)
    elapsed = time.time() - started
    payload = {
        "status": "complete",
        "method": "ALG",
        "scope": ["flowers102", "chaoyang", "cifar100"],
        "official_lg_commit": OFFICIAL_LG_COMMIT,
        "alg_paper_doi": ALG_PAPER_DOI,
        "ours_protocol_used": False,
        "runs": records,
        "estimated_three_full_runs_seconds": estimated_total,
        "estimated_three_full_runs_human": format_duration(estimated_total),
        "timing_sequence_elapsed_seconds": elapsed,
        "timing_sequence_elapsed_human": format_duration(elapsed),
    }
    atomic_write_json(combined_path, payload)
    log("=" * 72)
    for record in records:
        log(
            f"[TIMING_SUMMARY] dataset={record['dataset']} "
            f"avg_epoch={record['avg_epoch_seconds']:.1f}s "
            f"estimated_300={record['estimated_300_human']}"
        )
    log(
        f"[TIMING_TOTAL] estimated_three_full_runs="
        f"{format_duration(estimated_total)}"
    )
    log(f"[FINAL_RESULT] combined_summary={combined_path.resolve()}")
    log("[DONE] Three-dataset canonical ALG timing completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        log("=" * 72)
        log(f"[FATAL] {type(error).__name__}: {error}")
        raise
