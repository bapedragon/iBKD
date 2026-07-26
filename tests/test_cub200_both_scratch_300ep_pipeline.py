from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from methods import run_cub200_both_scratch_100ep_all as shared_runner
from methods import run_cub200_both_scratch_300ep_all as locked_runner
from methods.run_cub200_both_scratch_100ep_all import (
    EXPECTED_BATCH_SIZE,
    FINAL_RESULT_ORDER,
    format_final_top1_summary,
    protocol_family_for,
    run_names_for,
    validate_student_summary,
    validate_teacher_summary,
)


STUDENT_EPOCHS = 300


class Cub200BothScratch300EpochPipelineTest(unittest.TestCase):
    def test_public_runner_locks_300_without_cli_override(self) -> None:
        original = shared_runner.LOCKED_STUDENT_EPOCHS
        try:
            with mock.patch.object(shared_runner, "main") as shared_main:
                locked_runner.main()
            shared_main.assert_called_once_with()
            self.assertEqual(shared_runner.LOCKED_STUDENT_EPOCHS, 300)
        finally:
            shared_runner.LOCKED_STUDENT_EPOCHS = original

    def test_300_epoch_identity_is_separate_and_explicit(self) -> None:
        self.assertEqual(
            protocol_family_for(STUDENT_EPOCHS),
            "cub200_resnet50_deit_ti_224_both_scratch_300ep",
        )
        run_names = run_names_for(STUDENT_EPOCHS)
        self.assertEqual(set(run_names), set(FINAL_RESULT_ORDER[1:]))
        for run_name in run_names.values():
            self.assertIn("300ep", run_name)
            self.assertNotIn("100ep", run_name)

    def test_compact_summary_uses_300_epoch_tag(self) -> None:
        results = {
            "teacher": 40.111,
            "VanillaB128": 41.222,
            "LG": 42.333,
            "ALG": 43.444,
            "OursB64": 44.555,
            "OursB128": 45.666,
        }
        self.assertEqual(
            format_final_top1_summary(results, STUDENT_EPOCHS),
            (
                "[FINAL_TOP1_SUMMARY_224_BOTH_SCRATCH_300EP] "
                "Teacher=40.11% VanillaB128=41.22% LG=42.33% "
                "ALG=43.44% OursB64=44.55% OursB128=45.67%"
            ),
        )

    def test_identity_checks_lock_teacher_200_and_students_300(self) -> None:
        teacher = {
            "pretrained": False,
            "pretrained_source": None,
            "input_resolution": 224,
            "model_name": "resnet50_cub200_scratch_224",
            "protocol_family": "cub200_resnet50_224_scratch",
            "planned_epochs": 200,
        }
        validate_teacher_summary(teacher)
        with self.assertRaisesRegex(RuntimeError, "teacher identity"):
            validate_teacher_summary(dict(teacher, planned_epochs=300))

        student = {
            "student_pretrained": False,
            "student_pretrained_source": None,
            "input_resolution": 224,
            "batch_size": 128,
            "planned_epochs": 300,
            "teacher": None,
        }
        validate_student_summary(
            "VanillaB128",
            student,
            student_epochs=STUDENT_EPOCHS,
        )
        with self.assertRaisesRegex(RuntimeError, "identity validation"):
            validate_student_summary(
                "VanillaB128",
                dict(student, planned_epochs=100),
                student_epochs=STUDENT_EPOCHS,
            )

    def test_main_changes_only_student_horizon(self) -> None:
        commands: dict[str, list[str]] = {}
        teacher_summary = {
            "pretrained": False,
            "pretrained_source": None,
            "input_resolution": 224,
            "model_name": "resnet50_cub200_scratch_224",
            "protocol_family": "cub200_resnet50_224_scratch",
            "planned_epochs": 200,
            "best_top1": 48.31,
            "estimated_planned_seconds": 1.0,
        }
        teacher_spec = {"pretrained": False, "input_resolution": 224}

        def fake_run_tracked_task(**kwargs: object) -> dict[str, object]:
            name = str(kwargs["name"])
            commands[name] = list(kwargs["command"])  # type: ignore[arg-type]
            if name == "teacher":
                return teacher_summary
            return {
                "student_pretrained": False,
                "student_pretrained_source": None,
                "input_resolution": 224,
                "batch_size": EXPECTED_BATCH_SIZE[name],
                "planned_epochs": STUDENT_EPOCHS,
                "teacher": None if name == "VanillaB128" else teacher_spec,
                "best_top1": 25.0,
                "estimated_planned_seconds": 1.0,
            }

        with tempfile.TemporaryDirectory() as temporary:
            args = argparse.Namespace(
                timing_run=True,
                full_run=False,
                data_dir=Path("./data/cub200"),
                output_dir=Path(temporary),
                num_workers=4,
                no_download=True,
                student_epochs=STUDENT_EPOCHS,
            )
            with (
                mock.patch.object(
                    shared_runner,
                    "parse_args",
                    return_value=args,
                ),
                mock.patch.object(
                    shared_runner,
                    "run_tracked_task",
                    side_effect=fake_run_tracked_task,
                ),
            ):
                shared_runner.main()

        teacher_command = commands["teacher"]
        self.assertNotIn("--student-epochs", teacher_command)
        initialization_index = teacher_command.index("--initialization")
        self.assertEqual(
            teacher_command[initialization_index + 1],
            "scratch",
        )
        for name in FINAL_RESULT_ORDER[1:]:
            command = commands[name]
            self.assertIn("--no-student-pretrained", command)
            epoch_index = command.index("--student-epochs")
            self.assertEqual(command[epoch_index + 1], "300")

    def test_issues_have_both_accounts_and_single_line_commands(self) -> None:
        root = (
            Path(__file__).resolve().parents[1]
            / "methods"
            / "cub200_both_scratch_300ep"
        )
        timing = (root / "H200_TIMING_ISSUE.md").read_text(encoding="utf-8")
        full = (root / "H200_FULL_ISSUE.md").read_text(encoding="utf-8")
        timing_command = (
            "python methods/run_cub200_both_scratch_300ep_all.py "
            "--timing-run --num-workers 4 --output-dir "
            "/app/output/cub200_both_scratch_300ep_timing_seed1"
        )
        full_command = (
            "python methods/run_cub200_both_scratch_300ep_all.py "
            "--full-run --num-workers 4 --output-dir "
            "/app/output/cub200_both_scratch_300ep_full_seed1"
        )
        for issue in (timing, full):
            self.assertIn("bapedragon", issue)
            self.assertIn("kau-aimslab", issue)
            self.assertIn("teacher_planned_epochs=200", issue)
            self.assertIn("all_students_planned_epochs=300", issue)
            self.assertIn("OursB64", issue)
            self.assertIn("OursB128", issue)
            self.assertNotIn("VanillaB64", issue)
        self.assertIn(timing_command, timing)
        self.assertIn(full_command, full)
        self.assertNotIn(f"{timing_command} \\", timing)
        self.assertNotIn(f"{full_command} \\", full)


if __name__ == "__main__":
    unittest.main()
