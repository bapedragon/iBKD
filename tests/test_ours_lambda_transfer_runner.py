from __future__ import annotations

import argparse
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from methods.Ours.lambda_transfer import run_flowers_chaoyang as runner
from methods.Ours.lambda_transfer.run_flowers_chaoyang import (
    ACTIVE_LAMBDA,
    POD_LIMIT_SECONDS,
    REFERENCE_LAMBDA,
    TASKS,
    build_command,
    format_duration,
    load_and_validate_reference,
)


class OursLambdaTransferRunnerTest(unittest.TestCase):
    def test_reference_summaries_are_complete_and_protocol_locked(self) -> None:
        self.assertEqual(ACTIVE_LAMBDA, 0.25)
        self.assertEqual(REFERENCE_LAMBDA, 0.5)
        self.assertEqual(
            [task.dataset for task in TASKS], ["flowers102", "chaoyang"]
        )
        for task in TASKS:
            summary = load_and_validate_reference(task)
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["args"]["fusion_ratio"], 0.5)
            self.assertEqual(summary["best_top1"], task.reference_best_top1)

    def test_full_commands_change_lambda_and_isolate_outputs(self) -> None:
        args = argparse.Namespace(
            timing_run=False,
            full_run=True,
            flowers_data_dir=Path("/flowers"),
            chaoyang_data_dir=Path("/chaoyang"),
            teacher_root=Path("/teachers"),
            num_workers=4,
        )
        output_root = Path("/tmp/ours-lambda-transfer-test")
        run_dirs: list[Path] = []
        for task in TASKS:
            command, run_dir = build_command(task, args, output_root)
            run_dirs.append(run_dir)
            self.assertNotIn("--timing-run", command)
            self.assertEqual(command[command.index("--fusion-ratio") + 1], "0.25")
            self.assertEqual(command[command.index("--batch-size") + 1], "64")
            self.assertEqual(
                command[command.index("--student-epochs") + 1], "300"
            )
            self.assertEqual(command[command.index("--seed") + 1], "1")
            self.assertIn("--no-amp", command)
            self.assertIn(task.dataset, str(run_dir))
        self.assertNotEqual(run_dirs[0], run_dirs[1])

    def test_timing_command_preserves_300_epoch_plan(self) -> None:
        args = argparse.Namespace(
            timing_run=True,
            full_run=False,
            flowers_data_dir=Path("/flowers"),
            chaoyang_data_dir=Path("/chaoyang"),
            teacher_root=Path("/teachers"),
            num_workers=2,
        )
        output_root = Path("/tmp/ours-lambda-transfer-timing-test")
        for task in TASKS:
            command, _ = build_command(task, args, output_root)
            self.assertIn("--timing-run", command)
            self.assertEqual(
                command[command.index("--student-epochs") + 1], "300"
            )
            self.assertEqual(command[command.index("--num-workers") + 1], "2")

    def test_issue_uses_renamed_repository_and_requires_both_values(self) -> None:
        issue = (
            Path(__file__).resolve().parents[1]
            / "methods/Ours/lambda_transfer/H200_ISSUE.md"
        ).read_text(encoding="utf-8")
        self.assertIn("https://github.com/bapedragon/iBKD.git", issue)
        self.assertIn("--full-run --num-workers 4", issue)
        self.assertIn("only_change=lambda", issue)
        self.assertIn("[SEQUENCE_DONE] completed_tasks=2/2", issue)
        self.assertIn(
            "[FINAL_TOP1_SUMMARY_LAMBDA_0P25] "
            "Flowers102=...% Chaoyang=...%",
            issue,
        )

    def test_duration_and_reference_json_are_valid(self) -> None:
        self.assertEqual(POD_LIMIT_SECONDS, 600 * 60)
        self.assertEqual(format_duration(5757), "1h 35m 57s")
        for task in TASKS:
            reference_path = (
                Path(__file__).resolve().parents[1] / task.reference_summary
            )
            json.loads(reference_path.read_text(encoding="utf-8"))

    def test_full_sequence_last_line_contains_both_results(self) -> None:
        expected_best = {"flowers102": 75.25, "chaoyang": 82.5}

        def fake_run(command: list[str], *, cwd: Path, check: bool) -> None:
            self.assertEqual(cwd, runner.REPOSITORY_ROOT)
            self.assertTrue(check)
            script = command[1]
            dataset = "flowers102" if "flowers102" in script else "chaoyang"
            task = next(item for item in TASKS if item.dataset == dataset)
            candidate = copy.deepcopy(load_and_validate_reference(task))
            arguments = candidate["args"]
            arguments["fusion_ratio"] = ACTIVE_LAMBDA
            arguments["student_epochs"] = 300
            arguments["planned_epochs"] = 300
            arguments["flowers_split_policy"] = "trainval_test_best"
            candidate["best_top1"] = expected_best[dataset]
            candidate["latest_top1"] = expected_best[dataset] - 0.25
            candidate["planned_epochs"] = 300
            candidate["avg_epoch_seconds"] = 10.0
            candidate["estimated_planned_seconds"] = 3000.0
            candidate["estimated_planned_human"] = "50m 00s"
            output_dir = Path(command[command.index("--output-dir") + 1])
            name = command[command.index("--run-name") + 1]
            run_dir = output_dir / name
            run_dir.mkdir(parents=True)
            (run_dir / "summary.json").write_text(
                json.dumps(candidate), encoding="utf-8"
            )

        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "output"
            captured = io.StringIO()
            with patch.object(runner.subprocess, "run", side_effect=fake_run):
                with patch("sys.stdout", captured):
                    runner.main(
                        [
                            "--full-run",
                            "--flowers-data-dir",
                            "/flowers",
                            "--chaoyang-data-dir",
                            "/chaoyang",
                            "--teacher-root",
                            "/teachers",
                            "--output-dir",
                            str(output_root),
                        ]
                    )
            lines = captured.getvalue().strip().splitlines()
            self.assertEqual(
                lines[-1],
                "[FINAL_TOP1_SUMMARY_LAMBDA_0P25] "
                "Flowers102=75.25% Chaoyang=82.50%",
            )
            sequence = json.loads(
                (output_root / "sequence_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(sequence["completed_tasks"], 2)
            self.assertEqual(sequence["total_tasks"], 2)


if __name__ == "__main__":
    unittest.main()
