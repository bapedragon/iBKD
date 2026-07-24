from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from methods.ALG.cub200_resnet50_224_scratch.train import (
    PROTOCOL_DEFAULTS as ALG_SCRATCH_DEFAULTS,
)
from methods.Ours.cub200_resnet50_224_scratch.train import (
    PROTOCOL_DEFAULTS as OURS_SCRATCH_DEFAULTS,
)
from methods.Vanilla.cub200_224.train import (
    PROFILE_BATCH_SIZE,
    loader_args,
    parse_args as parse_vanilla_args,
)
from methods.run_cub200_resnet50_224_scratch_all import (
    FINAL_RESULT_ORDER,
    METHOD_SCRIPTS,
    collect_best_top1,
    format_final_top1_summary,
    parse_args as parse_runner_args,
    validate_scratch_teacher_summary,
)
from teachers import verify_checkpoints
from teachers.train_teacher_cub200_resnet50_224 import (
    SCRATCH_MODEL_NAME,
    SCRATCH_PROTOCOL_FAMILY,
    SCRATCH_RECIPE_NAME,
    parse_args as parse_teacher_args,
)


class Cub200ResNet50224ScratchPipelineTest(unittest.TestCase):
    def test_teacher_scratch_identity_is_explicit_and_loadable(self) -> None:
        self.assertEqual(SCRATCH_MODEL_NAME, "resnet50_cub200_scratch_224")
        self.assertIn("scratch", SCRATCH_RECIPE_NAME)
        self.assertEqual(
            SCRATCH_PROTOCOL_FAMILY,
            "cub200_resnet50_224_scratch",
        )
        self.assertIn(
            SCRATCH_MODEL_NAME,
            verify_checkpoints.NAMED_MODEL_FACTORIES,
        )
        with mock.patch(
            "sys.argv",
            ["train.py", "--initialization", "scratch", "--timing-run"],
        ):
            args = parse_teacher_args()
        self.assertEqual(args.initialization, "scratch")
        self.assertTrue(args.timing_run)

    def test_wrappers_lock_scratch_protocol_identity_and_224_teacher(self) -> None:
        alg = dict(ALG_SCRATCH_DEFAULTS)
        ours = dict(OURS_SCRATCH_DEFAULTS)
        self.assertIn("scratch_teacher", alg["--protocol-name"])
        self.assertIn("scratch_teacher", ours["--protocol-name"])
        self.assertEqual(alg["--teacher-image-size"], "224")
        self.assertEqual(ours["--teacher-image-size"], "224")
        self.assertEqual(alg["--batch-size"], "128")
        self.assertEqual(ours["--batch-size"], "64")
        self.assertEqual(
            METHOD_SCRIPTS,
            {
                "LG": "methods/LG/cub200_resnet50_224_scratch/train.py",
                "ALG": "methods/ALG/cub200_resnet50_224_scratch/train.py",
                "Ours": "methods/Ours/cub200_resnet50_224_scratch/train.py",
            },
        )

    def test_vanilla_profiles_are_teacher_free_and_batch_matched(self) -> None:
        self.assertEqual(
            PROFILE_BATCH_SIZE,
            {
                "lg_official_b128": 128,
                "ours_current_b64": 64,
            },
        )
        args = parse_vanilla_args(
            [
                "--profile",
                "lg_official_b128",
                "--timing-run",
                "   ",
                "--num-workers",
                "4",
            ]
        )
        prepared = loader_args(args)
        self.assertEqual(prepared.dataset, "cub200")
        self.assertEqual(prepared.batch_size, 128)
        self.assertEqual(prepared.eval_interpolation, "bilinear")
        ours_args = parse_vanilla_args(
            [
                "--profile",
                "ours_current_b64",
                "--timing-run",
            ]
        )
        ours_prepared = loader_args(ours_args)
        self.assertEqual(ours_prepared.batch_size, 64)
        self.assertEqual(ours_prepared.eval_interpolation, "bicubic")

    def test_runner_compacts_all_six_results_and_ignores_whitespace(self) -> None:
        args = parse_runner_args(
            ["--timing-run", "   ", "--num-workers", "4"]
        )
        self.assertTrue(args.timing_run)
        self.assertEqual(
            FINAL_RESULT_ORDER,
            (
                "teacher",
                "VanillaB128",
                "LG",
                "ALG",
                "VanillaB64",
                "Ours",
            ),
        )
        results = collect_best_top1(
            {
                "teacher": {"best_top1": 10.111},
                "VanillaB128": {"best_top1": 20.222},
                "LG": {"best_top1": 30.333},
                "ALG": {"best_top1": 40.444},
                "VanillaB64": {"best_top1": 50.555},
                "Ours": {"best_top1": 60.666},
            }
        )
        self.assertEqual(
            format_final_top1_summary(results),
            (
                "[FINAL_TOP1_SUMMARY_224_SCRATCH] "
                "TeacherScratch224=10.11% VanillaB128=20.22% "
                "LG=30.33% ALG=40.44% VanillaB64=50.55% Ours=60.67%"
            ),
        )

    def test_runner_rejects_pretrained_or_mislabeled_teacher(self) -> None:
        valid = {
            "pretrained": False,
            "input_resolution": 224,
            "model_name": "resnet50_cub200_scratch_224",
            "protocol_family": "cub200_resnet50_224_scratch",
        }
        validate_scratch_teacher_summary(valid)
        invalid = dict(valid, pretrained=True)
        with self.assertRaisesRegex(
            RuntimeError,
            "Scratch teacher identity validation failed",
        ):
            validate_scratch_teacher_summary(invalid)

    def test_timing_issue_has_both_accounts_and_single_line_command(self) -> None:
        issue = (
            Path(__file__).resolve().parents[1]
            / "methods"
            / "cub200_resnet50_224_scratch"
            / "H200_TIMING_ISSUE.md"
        ).read_text(encoding="utf-8")
        command = (
            "python methods/run_cub200_resnet50_224_scratch_all.py "
            "--timing-run --num-workers 4 --output-dir "
            "/app/output/cub200_resnet50_224_scratch_all_timing_seed1"
        )
        self.assertIn("bapedragon", issue)
        self.assertIn("kau-aimslab", issue)
        self.assertIn(command, issue)
        self.assertNotIn(f"{command} \\", issue)
        self.assertIn("completed_tasks=6/6", issue)
        self.assertIn("teacher_pretrained=False", issue)


if __name__ == "__main__":
    unittest.main()
