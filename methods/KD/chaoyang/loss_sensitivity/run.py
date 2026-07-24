#!/usr/bin/env python3
"""Run Chaoyang KD variants B and C sequentially with isolated outputs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
TRAIN_ENTRYPOINT = REPOSITORY_ROOT / "methods/KD/chaoyang/train.py"
VERIFY_ENTRYPOINT = REPOSITORY_ROOT / "teachers/verify_checkpoints.py"
PLANNED_EPOCHS = 300

# Variant A (T=4, KD=0.9) already exists in the result archive.  Only T and
# KD weight change below; CE weight is always 1 - KD weight.
VARIANTS = (
    {
        "id": "B",
        "temperature": 2.0,
        "kd_weight": 0.5,
        "protocol_name": "chaoyang_kd_T2_ce050_kd050_300ep_seed42_v1",
    },
    {
        "id": "C",
        "temperature": 2.0,
        "kd_weight": 0.25,
        "protocol_name": "chaoyang_kd_T2_ce075_kd025_300ep_seed42_v1",
    },
)


def log(message: str = "") -> None:
    print(message, flush=True)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--timing-run",
        action="store_true",
        help="Run two full-data epochs for B and C and estimate 300 epochs.",
    )
    mode.add_argument(
        "--full-run",
        action="store_true",
        help="Run B and C for the locked 300-epoch Chaoyang protocol.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data/chaoyang"))
    parser.add_argument(
        "--teacher-root",
        type=Path,
        default=REPOSITORY_ROOT / "teachers/checkpoints",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./outputs/kd_chaoyang_loss_sensitivity"),
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def run_name(variant: dict[str, Any], timing_run: bool, seed: int) -> str:
    ce_weight = 1.0 - float(variant["kd_weight"])
    suffix = "timing_2ep" if timing_run else f"full_{PLANNED_EPOCHS}ep"
    return (
        f"variant_{variant['id'].lower()}_T{float(variant['temperature']):g}_"
        f"ce{ce_weight:.2f}_kd{float(variant['kd_weight']):.2f}_"
        f"{suffix}_seed{seed}"
    )


def build_command(
    args: argparse.Namespace,
    variant: dict[str, Any],
    output_root: Path,
) -> tuple[str, list[str]]:
    name = run_name(variant, args.timing_run, args.seed)
    command = [
        sys.executable,
        str(TRAIN_ENTRYPOINT),
        "--protocol-name",
        str(variant["protocol_name"]),
        "--data-dir",
        str(args.data_dir),
        "--teacher-root",
        str(args.teacher_root),
        "--output-dir",
        str(output_root),
        "--run-name",
        name,
        "--student-epochs",
        str(PLANNED_EPOCHS),
        "--batch-size",
        "64",
        "--num-workers",
        str(args.num_workers),
        "--image-size",
        "224",
        "--lr",
        "0.0005",
        "--weight-decay",
        "0.05",
        "--warmup-epochs",
        "5",
        "--temperature",
        str(variant["temperature"]),
        "--kd-weight",
        str(variant["kd_weight"]),
        "--label-smoothing",
        "0.1",
        "--seed",
        str(args.seed),
    ]
    if args.timing_run:
        command.append("--timing-run")
    return name, command


def main() -> None:
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")

    output_root = args.output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "status.json"
    combined_summary_path = output_root / "summary.json"
    records: list[dict[str, Any]] = []
    sequence_start = time.time()

    log("=" * 80)
    log("CHAOYANG KD LOSS SENSITIVITY / VARIANT B -> VARIANT C")
    log("=" * 80)
    log(
        f"[MODE] timing_run={args.timing_run} full_run={args.full_run} "
        f"planned_epochs={PLANNED_EPOCHS}"
    )
    log(f"[PATH] data_dir={args.data_dir.expanduser().resolve()}")
    log(f"[PATH] teacher_root={args.teacher_root.expanduser().resolve()}")
    log(f"[PATH] output_root={output_root}")
    log(
        "[PROTOCOL_LOCK] Same as completed Chaoyang KD A: ResNet56 teacher, "
        "DeiT-Ti student, official train/test split, student/teacher inputs "
        "224/32, AdamW, lr=5e-4, weight_decay=0.05, cosine, batch=64, "
        "warmup=5, label_smoothing=0.1, seed fixed by --seed."
    )
    log(
        "[ONLY_CHANGED] B: T=2 CE=0.50 KD=0.50; "
        "C: T=2 CE=0.75 KD=0.25."
    )
    log(
        "[OUTPUT] B and C use independent run directories. A later failure "
        "does not overwrite an earlier completed result."
    )
    if args.timing_run:
        log(
            "[TIMING_NOTE] Two-epoch Top-1 is only a runtime/smoke check; "
            "it is not comparable to the completed 300-epoch A result."
        )

    verify_command = [
        sys.executable,
        str(VERIFY_ENTRYPOINT),
        "--dataset",
        "chaoyang",
        "--checkpoint-root",
        str(args.teacher_root),
    ]
    log(f"[PRECHECK] command={' '.join(verify_command)}")
    subprocess.run(verify_command, cwd=REPOSITORY_ROOT, check=True)

    try:
        for index, variant in enumerate(VARIANTS, start=1):
            name, command = build_command(args, variant, output_root)
            run_dir = output_root / name
            ce_weight = 1.0 - float(variant["kd_weight"])
            record: dict[str, Any] = {
                "order": index,
                "variant": variant["id"],
                "temperature": variant["temperature"],
                "ce_weight": ce_weight,
                "kd_weight": variant["kd_weight"],
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
                f"[SEQUENCE][{index}/{len(VARIANTS)}] START "
                f"variant={variant['id']} T={variant['temperature']} "
                f"CE={ce_weight:.2f} KD={float(variant['kd_weight']):.2f}"
            )
            log(f"[SEQUENCE] command={' '.join(command)}")
            run_start = time.time()
            subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)
            record["elapsed_seconds"] = time.time() - run_start
            run_summary_path = run_dir / "summary.json"
            if not run_summary_path.is_file():
                raise FileNotFoundError(
                    f"Variant {variant['id']} completed without summary: "
                    f"{run_summary_path}"
                )
            run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
            record["status"] = "complete"
            record["summary"] = str(run_summary_path)
            record["best_top1"] = run_summary["best_top1"]
            record["latest_top1"] = run_summary["latest_top1"]
            atomic_json(
                status_path,
                {
                    "status": (
                        "complete" if index == len(VARIANTS) else "running"
                    ),
                    "mode": "timing" if args.timing_run else "full",
                    "records": records,
                },
            )
            log(
                f"[VARIANT_RESULT] id={variant['id']} "
                f"best_top1={float(record['best_top1']):.2f}% "
                f"latest_top1={float(record['latest_top1']):.2f}% "
                f"summary={run_summary_path}"
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

    combined_summary = {
        "status": "complete",
        "method": "KD",
        "dataset": "chaoyang",
        "experiment": "loss_sensitivity_B_C",
        "mode": "timing" if args.timing_run else "full",
        "planned_epochs": PLANNED_EPOCHS,
        "protocol_lock": {
            "batch_size": 64,
            "warmup_epochs": 5,
            "student_input": 224,
            "teacher_input": 32,
            "optimizer": "AdamW",
            "lr": 0.0005,
            "weight_decay": 0.05,
            "label_smoothing": 0.1,
            "seed": args.seed,
        },
        "records": records,
        "elapsed_seconds": time.time() - sequence_start,
    }
    atomic_json(combined_summary_path, combined_summary)
    log("=" * 80)
    log("[FINAL_COMPARISON] Chaoyang KD B/C sequence completed.")
    for record in records:
        log(
            f"[FINAL_COMPARISON] variant={record['variant']} "
            f"T={record['temperature']} CE={record['ce_weight']:.2f} "
            f"KD={record['kd_weight']:.2f} "
            f"best_top1={float(record['best_top1']):.2f}%"
        )
    log(f"[FINAL_COMPARISON] combined_summary={combined_summary_path}")
    log("[DONE] Both isolated KD variants completed; resources may be released.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        log(f"[FATAL] {type(error).__name__}: {error}")
        traceback.print_exc()
        log("[FATAL] KD loss-sensitivity sequence did not complete.")
        raise
