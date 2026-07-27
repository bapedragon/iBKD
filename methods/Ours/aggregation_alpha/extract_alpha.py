"""Extract stage-wise Ours V1 aggregation coefficients from best checkpoints.

This script performs no training and never runs the student or teacher models.
It reads the trusted, repository-owned Ours V1 checkpoints on CPU, applies the
same softmax used by ``TransformerAggregationPooling``, and cross-checks the
result against each run's saved ``aggregation_weights`` summary field.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch


NUM_TEACHER_STAGES = 3
NUM_STUDENT_BLOCKS = 12
ALPHA_ATOL = 1e-6


@dataclass(frozen=True)
class RunSpec:
    dataset: str
    display_name: str
    relative_dir: str


DEFAULT_RUNS = (
    RunSpec(
        dataset="cifar100",
        display_name="CIFAR-100",
        relative_dir="results/Ours/cifar100/researcher_sync_v1_300ep_seed1",
    ),
    RunSpec(
        dataset="flowers102",
        display_name="Flowers-102",
        relative_dir="results/Ours/flowers102/researcher_sync_v1_300ep_seed1",
    ),
    RunSpec(
        dataset="chaoyang",
        display_name="Chaoyang",
        relative_dir="results/Ours/chaoyang/cifar100_locked_b64_v1_300ep_seed1",
    ),
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description=(
            "Extract softmax-normalized 3x12 aggregation alpha matrices from "
            "the three selected Ours V1 best checkpoints."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=repo_root,
        help="IBAM_KD_H200_V2 repository root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "outputs" / "ours_v1_aggregation_alpha",
        help="Directory for JSON, CSV, and Markdown outputs.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_trusted_checkpoint(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: expected a dictionary checkpoint")
    return payload


def validate_checkpoint_identity(payload: dict[str, Any], spec: RunSpec) -> None:
    expected = {
        "method": "Ours",
        "student": "deit_ti",
        "dataset": spec.dataset,
    }
    for field, expected_value in expected.items():
        actual = payload.get(field)
        if actual != expected_value:
            raise ValueError(
                f"{spec.display_name}: checkpoint {field}={actual!r}, "
                f"expected {expected_value!r}"
            )


def compute_alpha(payload: dict[str, Any], spec: RunSpec) -> torch.Tensor:
    ours_state = payload.get("ours")
    if not isinstance(ours_state, dict):
        raise KeyError(f"{spec.display_name}: missing checkpoint['ours']")
    raw_weights = ours_state.get("aggregation.weights")
    if not isinstance(raw_weights, torch.Tensor):
        raise KeyError(
            f"{spec.display_name}: missing tensor "
            "checkpoint['ours']['aggregation.weights']"
        )
    expected_shape = (NUM_TEACHER_STAGES, NUM_STUDENT_BLOCKS)
    if tuple(raw_weights.shape) != expected_shape:
        raise ValueError(
            f"{spec.display_name}: aggregation.weights shape "
            f"{tuple(raw_weights.shape)}, expected {expected_shape}"
        )
    alpha = torch.softmax(raw_weights.detach().to(dtype=torch.float64), dim=-1)
    if not torch.isfinite(alpha).all():
        raise ValueError(f"{spec.display_name}: alpha contains non-finite values")
    if bool((alpha < 0).any()):
        raise ValueError(f"{spec.display_name}: alpha contains negative values")
    expected_sums = torch.ones(NUM_TEACHER_STAGES, dtype=alpha.dtype)
    if not torch.allclose(alpha.sum(dim=-1), expected_sums, atol=ALPHA_ATOL, rtol=0):
        raise ValueError(f"{spec.display_name}: alpha rows do not sum to one")
    return alpha


def validate_summary(
    summary: dict[str, Any],
    alpha: torch.Tensor,
    spec: RunSpec,
) -> float:
    if summary.get("dataset") != spec.dataset:
        raise ValueError(
            f"{spec.display_name}: summary dataset={summary.get('dataset')!r}, "
            f"expected {spec.dataset!r}"
        )
    saved = summary.get("aggregation_weights")
    if not isinstance(saved, list):
        raise KeyError(f"{spec.display_name}: summary lacks aggregation_weights")
    saved_alpha = torch.tensor(saved, dtype=alpha.dtype)
    if tuple(saved_alpha.shape) != tuple(alpha.shape):
        raise ValueError(
            f"{spec.display_name}: summary alpha shape {tuple(saved_alpha.shape)}, "
            f"expected {tuple(alpha.shape)}"
        )
    max_abs_diff = float(torch.max(torch.abs(alpha - saved_alpha)))
    if not math.isfinite(max_abs_diff) or max_abs_diff > ALPHA_ATOL:
        raise ValueError(
            f"{spec.display_name}: checkpoint/summary alpha mismatch "
            f"(max_abs_diff={max_abs_diff:.3e})"
        )
    return max_abs_diff


def top_three(alpha_row: Sequence[float]) -> list[dict[str, float | int]]:
    ranked = sorted(
        enumerate(alpha_row, start=1),
        key=lambda item: item[1],
        reverse=True,
    )[:3]
    return [
        {"student_block": block, "alpha": float(value)}
        for block, value in ranked
    ]


def extract_run(repo_root: Path, spec: RunSpec) -> dict[str, Any]:
    run_dir = repo_root / spec.relative_dir
    checkpoint_path = run_dir / "student_best.pt"
    summary_path = run_dir / "run_summary.json"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)

    payload = load_trusted_checkpoint(checkpoint_path)
    validate_checkpoint_identity(payload, spec)
    alpha_tensor = compute_alpha(payload, spec)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    max_abs_diff = validate_summary(summary, alpha_tensor, spec)
    alpha = alpha_tensor.tolist()

    stages = []
    for stage_index, stage_alpha in enumerate(alpha, start=1):
        stages.append(
            {
                "teacher_stage": stage_index,
                "alpha": stage_alpha,
                "top3": top_three(stage_alpha),
                "sum": float(sum(stage_alpha)),
            }
        )

    return {
        "dataset": spec.dataset,
        "display_name": spec.display_name,
        "checkpoint": str(checkpoint_path.relative_to(repo_root)),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": int(payload["epoch"]),
        "checkpoint_accuracy": float(payload["accuracy"]),
        "summary_best_top1": float(summary["best_top1"]),
        "summary_crosscheck_max_abs_diff": max_abs_diff,
        "shape": [NUM_TEACHER_STAGES, NUM_STUDENT_BLOCKS],
        "stages": stages,
    }


def write_json(output_dir: Path, document: dict[str, Any]) -> Path:
    path = output_dir / "aggregation_alpha.json"
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return path


def write_csv(output_dir: Path, runs: Sequence[dict[str, Any]]) -> Path:
    path = output_dir / "aggregation_alpha.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "dataset",
                "teacher_stage",
                "student_block",
                "alpha",
                "checkpoint_epoch",
                "checkpoint_accuracy",
                "checkpoint_sha256",
            ]
        )
        for run in runs:
            for stage in run["stages"]:
                for block_index, alpha in enumerate(stage["alpha"], start=1):
                    writer.writerow(
                        [
                            run["dataset"],
                            stage["teacher_stage"],
                            block_index,
                            f"{alpha:.10f}",
                            run["checkpoint_epoch"],
                            f"{run['checkpoint_accuracy']:.10f}",
                            run["checkpoint_sha256"],
                        ]
                    )
    return path


def write_markdown(output_dir: Path, runs: Sequence[dict[str, Any]]) -> Path:
    lines = [
        "# Ours V1 learnable-aggregation alpha",
        "",
        "Each row is a teacher stage. `B1`–`B12` use one-based student-block "
        "indexing. Values are `softmax(aggregation.weights)` from the selected "
        "best checkpoint; each row sums to one.",
        "",
    ]
    block_headers = [f"B{index}" for index in range(1, NUM_STUDENT_BLOCKS + 1)]
    for run in runs:
        lines.extend(
            [
                f"## {run['display_name']}",
                "",
                f"- Best checkpoint epoch: {run['checkpoint_epoch']}",
                f"- Checkpoint accuracy: {run['checkpoint_accuracy']:.4f}%",
                f"- Checkpoint SHA-256: `{run['checkpoint_sha256']}`",
                "",
                "| Teacher stage | "
                + " | ".join(block_headers)
                + " | Top-3 blocks |",
                "|---:|"
                + "|".join(["---:" for _ in block_headers])
                + "|---|",
            ]
        )
        for stage in run["stages"]:
            alpha_cells = [f"{value:.6f}" for value in stage["alpha"]]
            top3 = ", ".join(
                f"B{item['student_block']}={item['alpha']:.4f}"
                for item in stage["top3"]
            )
            lines.append(
                f"| {stage['teacher_stage']} | "
                + " | ".join(alpha_cells)
                + f" | {top3} |"
            )
        lines.append("")
    path = output_dir / "aggregation_alpha.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        "[OURS_V1_ALPHA_PROTOCOL] "
        "method=OursV1 datasets=cifar100,flowers102,chaoyang "
        "source=student_best.pt normalization=softmax shape=3x12 "
        "device=cpu training=False"
    )
    runs = []
    for spec in DEFAULT_RUNS:
        run = extract_run(repo_root, spec)
        runs.append(run)
        print(
            f"[ALPHA_CHECK][{spec.dataset}] status=PASS "
            f"epoch={run['checkpoint_epoch']} "
            f"top1={run['checkpoint_accuracy']:.4f}% "
            f"shape=3x12 row_sums=1 "
            f"summary_max_abs_diff={run['summary_crosscheck_max_abs_diff']:.3e}"
        )
        for stage in run["stages"]:
            top3 = ",".join(
                f"B{item['student_block']}={item['alpha']:.6f}"
                for item in stage["top3"]
            )
            print(
                f"[ALPHA_TOP3][{spec.dataset}] "
                f"teacher_stage={stage['teacher_stage']} {top3}"
            )

    document = {
        "schema_version": 1,
        "method": "OursV1",
        "scope": "selected working-paper DeiT-Ti runs only",
        "alpha_definition": "softmax(ours['aggregation.weights'], dim=-1)",
        "indexing": {
            "teacher_stage": "one-based",
            "student_block": "one-based",
        },
        "runs": runs,
    }
    json_path = write_json(output_dir, document)
    csv_path = write_csv(output_dir, runs)
    markdown_path = write_markdown(output_dir, runs)
    for path in (json_path, csv_path, markdown_path):
        print(f"[ALPHA_OUTPUT] {path}")
    print(
        "[ALPHA_EXTRACTION_DONE] "
        "status=PASS datasets=3 matrices=3 values=108 "
        "training=False ours_v2=False"
    )


if __name__ == "__main__":
    main()
