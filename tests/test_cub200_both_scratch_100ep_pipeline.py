from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from methods import run_cub200_both_scratch_100ep_all as scratch_runner
from methods.run_cub200_both_scratch_100ep_all import (
    EXPECTED_BATCH_SIZE,
    FINAL_RESULT_ORDER,
    METHOD_SCRIPTS,
    STUDENT_EPOCHS,
    collect_best_top1,
    format_final_top1_summary,
    parse_args as parse_runner_args,
    validate_student_summary,
    validate_teacher_summary,
)


class Cub200BothScratch100EpochPipelineTest(unittest.TestCase):
    def test_runner_is_exact_six_task_paired_control(self) -> None:
        self.assertEqual(STUDENT_EPOCHS, 100)
        self.assertEqual(
            FINAL_RESULT_ORDER,
            (
                "teacher",
                "VanillaB128",
                "LG",
                "ALG",
                "OursB64",
                "OursB128",
            ),
        )
        self.assertNotIn("VanillaB64", FINAL_RESULT_ORDER)
        self.assertEqual(EXPECTED_BATCH_SIZE["OursB64"], 64)
        self.assertEqual(EXPECTED_BATCH_SIZE["OursB128"], 128)
        self.assertEqual(
            METHOD_SCRIPTS["OursB64"],
            METHOD_SCRIPTS["OursB128"],
        )
        for script in METHOD_SCRIPTS.values():
            self.assertIn("scratch", script)
            self.assertNotIn("pretrained", script)

    def test_runner_compacts_results_and_ignores_whitespace(self) -> None:
        args = parse_runner_args(
            ["--timing-run", "   ", "--num-workers", "4"]
        )
        self.assertTrue(args.timing_run)
        results = collect_best_top1(
            {
                "teacher": {"best_top1": 10.111},
                "VanillaB128": {"best_top1": 20.222},
                "LG": {"best_top1": 30.333},
                "ALG": {"best_top1": 40.444},
                "OursB64": {"best_top1": 50.555},
                "OursB128": {"best_top1": 60.666},
            }
        )
        self.assertEqual(
            format_final_top1_summary(results),
            (
                "[FINAL_TOP1_SUMMARY_224_BOTH_SCRATCH_100EP] "
                "Teacher=10.11% VanillaB128=20.22% LG=30.33% "
                "ALG=40.44% OursB64=50.55% OursB128=60.67%"
            ),
        )

    def test_identity_checks_require_scratch_on_both_sides(self) -> None:
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
            validate_teacher_summary(dict(teacher, pretrained=True))

        student = {
            "student_pretrained": False,
            "student_pretrained_source": None,
            "input_resolution": 224,
            "batch_size": 128,
            "planned_epochs": 100,
            "teacher": None,
        }
        validate_student_summary("VanillaB128", student)
        with self.assertRaisesRegex(RuntimeError, "identity validation"):
            validate_student_summary(
                "VanillaB128",
                dict(student, student_pretrained=True),
            )

        guided = dict(
            student,
            teacher={"pretrained": False, "input_resolution": 224},
        )
        validate_student_summary("LG", guided)
        with self.assertRaisesRegex(RuntimeError, "identity validation"):
            validate_student_summary(
                "LG",
                dict(
                    guided,
                    teacher={"pretrained": True, "input_resolution": 224},
                ),
            )

    def test_main_explicitly_disables_every_student_pretraining(self) -> None:
        commands: dict[str, list[str]] = {}
        teacher_summary = {
            "pretrained": False,
            "pretrained_source": None,
            "input_resolution": 224,
            "model_name": "resnet50_cub200_scratch_224",
            "protocol_family": "cub200_resnet50_224_scratch",
            "planned_epochs": 200,
            "best_top1": 60.0,
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
                "planned_epochs": 100,
                "teacher": None if name == "VanillaB128" else teacher_spec,
                "best_top1": 61.0,
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
            )
            with (
                mock.patch.object(
                    scratch_runner,
                    "parse_args",
                    return_value=args,
                ),
                mock.patch.object(
                    scratch_runner,
                    "run_tracked_task",
                    side_effect=fake_run_tracked_task,
                ),
            ):
                scratch_runner.main()

        teacher_command = commands["teacher"]
        initialization_index = teacher_command.index("--initialization")
        self.assertEqual(
            teacher_command[initialization_index + 1],
            "scratch",
        )
        self.assertNotIn("--student-epochs", teacher_command)
        for name in FINAL_RESULT_ORDER[1:]:
            command = commands[name]
            self.assertIn("--no-student-pretrained", command)
            self.assertNotIn("--student-pretrained", command)
            self.assertEqual(command.count("--student-epochs"), 1)
            epoch_index = command.index("--student-epochs")
            self.assertEqual(command[epoch_index + 1], "100")

    def test_issues_have_both_accounts_and_single_line_commands(self) -> None:
        root = (
            Path(__file__).resolve().parents[1]
            / "methods"
            / "cub200_both_scratch_100ep"
        )
        timing = (root / "H200_TIMING_ISSUE.md").read_text(encoding="utf-8")
        full = (root / "H200_FULL_ISSUE.md").read_text(encoding="utf-8")
        timing_command = (
            "python methods/run_cub200_both_scratch_100ep_all.py "
            "--timing-run --num-workers 4 --output-dir "
            "/app/output/cub200_both_scratch_100ep_timing_seed1"
        )
        full_command = (
            "python methods/run_cub200_both_scratch_100ep_all.py "
            "--full-run --num-workers 4 --output-dir "
            "/app/output/cub200_both_scratch_100ep_full_seed1"
        )
        for issue in (timing, full):
            self.assertIn("bapedragon", issue)
            self.assertIn("kau-aimslab", issue)
            self.assertIn("teacher_pretrained=False", issue)
            self.assertIn("all_students_pretrained=False", issue)
            self.assertIn("OursB64", issue)
            self.assertIn("OursB128", issue)
            self.assertNotIn("VanillaB64", issue)
        self.assertIn(timing_command, timing)
        self.assertIn(full_command, full)
        self.assertNotIn(f"{timing_command} \\", timing)
        self.assertNotIn(f"{full_command} \\", full)


if __name__ == "__main__":
    unittest.main()
