#!/usr/bin/env python3
"""Run canonical ALG full training on Flowers-102, Chaoyang, and CIFAR-100."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from methods.ALG.run_three_dataset_timing import (
    TASKS,
    atomic_write_json,
    audit_all,
    format_duration,
    validate_child_summary,
)
from methods.LG.official_lg import OFFICIAL_LG_COMMIT
from methods.LG.runtime import (
    ALG_PAPER_DOI,
    PAPER_GUIDANCE_STOP_EPOCH,
    REFERENCE_TOP1,
)


FULL_EPOCHS = 300


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-run",
        action="store_true",
        help="Required safety flag. Runs all three datasets for 300 epochs.",
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
        default=Path("/app/output/alg_official_three_dataset_full_300ep"),
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
    if not args.full_run:
        parser.error("--full-run is required to launch the three 300-epoch runs")
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    return args


def full_run_name(timing_name: str) -> str:
    suffix = "_timing_2ep"
    if not timing_name.endswith(suffix):
        raise ValueError(f"Unexpected timing run name: {timing_name}")
    return timing_name[: -len(suffix)] + "_full_300ep"


def data_dir_for(args: argparse.Namespace, data_arg: str) -> Path:
    return Path(getattr(args, data_arg))


def run_task(args: argparse.Namespace, task: Any) -> dict[str, Any]:
    run_name = full_run_name(task.run_name)
    data_dir = data_dir_for(args, task.data_arg)
    command = [
        sys.executable,
        "-u",
        str(task.script),
        "--student-epochs",
        str(FULL_EPOCHS),
        "--num-workers",
        str(args.num_workers),
        "--data-dir",
        str(data_dir),
        "--teacher-root",
        str(args.teacher_root),
        "--output-dir",
        str(args.output_dir),
        "--run-name",
        run_name,
    ]
    log("-" * 72)
    log(f"[FULL_START] dataset={task.dataset} epochs={FULL_EPOCHS}")
    log(f"[FULL_COMMAND] {' '.join(command)}")
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
            f"{task.dataset} full run failed with exit code {result.returncode}"
        )

    run_dir = args.output_dir / run_name
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"{task.dataset} summary was not created: {summary_path}"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    validate_child_summary(task, summary, student_epochs=FULL_EPOCHS)
    if summary.get("status") != "complete":
        raise RuntimeError(
            f"{task.dataset} summary is not complete: {summary.get('status')!r}"
        )

    record = {
        "dataset": task.dataset,
        "status": "complete",
        "epochs": FULL_EPOCHS,
        "best_top1": summary["best_top1"],
        "latest_top1": summary["latest_top1"],
        "paper_deit_top1": REFERENCE_TOP1[task.dataset]["alg"],
        "observed_guidance_stop_epoch": summary["controller"].get("stop_epoch"),
        "paper_guidance_stop_epoch": PAPER_GUIDANCE_STOP_EPOCH[task.dataset],
        "elapsed_seconds": wall_seconds,
        "elapsed_human": format_duration(wall_seconds),
        "run_name": run_name,
        "best_checkpoint": str((run_dir / "student_best.pt").resolve()),
        "latest_checkpoint": str((run_dir / "student_latest.pt").resolve()),
        "summary": str(summary_path.resolve()),
    }
    log(
        f"[FULL_RESULT] dataset={task.dataset} status=PASS "
        f"best_top1={summary['best_top1']:.2f}% "
        f"latest_top1={summary['latest_top1']:.2f}% "
        f"elapsed={format_duration(wall_seconds)}"
    )
    return record


def write_progress(
    path: Path,
    *,
    status: str,
    records: list[dict[str, Any]],
    started: float,
    error: str | None = None,
) -> None:
    payload = {
        "status": status,
        "method": "ALG",
        "scope": [task.dataset for task in TASKS],
        "epochs_per_dataset": FULL_EPOCHS,
        "official_lg_commit": OFFICIAL_LG_COMMIT,
        "alg_paper_doi": ALG_PAPER_DOI,
        "ours_protocol_used": False,
        "completed": records,
        "elapsed_seconds": time.time() - started,
        "elapsed_human": format_duration(time.time() - started),
    }
    if error is not None:
        payload["error"] = error
    atomic_write_json(path, payload)


def main() -> None:
    args = parse_args()
    audit_all("FULL TRAINING")
    if not args.teacher_root.is_dir():
        raise FileNotFoundError(f"Teacher root not found: {args.teacher_root}")
    if not args.chaoyang_data_dir.is_dir():
        raise FileNotFoundError(
            "Chaoyang mount not found. Expected the dataset under "
            f"{args.chaoyang_data_dir}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = args.output_dir / "three_dataset_full_summary.json"
    records: list[dict[str, Any]] = []
    started = time.time()
    try:
        for task in TASKS:
            records.append(run_task(args, task))
            write_progress(
                combined_path,
                status="running",
                records=records,
                started=started,
            )
    except Exception as error:
        write_progress(
            combined_path,
            status="failed",
            records=records,
            started=started,
            error=f"{type(error).__name__}: {error}",
        )
        raise

    write_progress(
        combined_path,
        status="complete",
        records=records,
        started=started,
    )
    log("=" * 72)
    for record in records:
        log(
            f"[FINAL_DATASET] dataset={record['dataset']} "
            f"best_top1={record['best_top1']:.2f}% "
            f"latest_top1={record['latest_top1']:.2f}% "
            f"checkpoint={record['best_checkpoint']}"
        )
    log(f"[FINAL_RESULT] combined_summary={combined_path.resolve()}")
    log(
        f"[FINAL_RESULT] elapsed={format_duration(time.time() - started)} "
        f"completed={len(records)}/{len(TASKS)}"
    )
    log("[DONE] Three-dataset canonical ALG full training completed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        log("=" * 72)
        log(f"[FATAL] {type(error).__name__}: {error}")
        raise
