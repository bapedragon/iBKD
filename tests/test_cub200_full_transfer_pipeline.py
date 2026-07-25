from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

import torch

from methods.ALG.cub200_resnet50_deit_ti_224_pretrained.train import (
    PROTOCOL_DEFAULTS as ALG_DEFAULTS,
)
from methods.KD.core import (
    STUDENT_PRETRAINED_MODELS,
    STUDENT_PRETRAINED_SOURCES,
)
from methods.LG import runtime as lg_runtime
from methods.Ours import core as ours_core
from methods.Ours.cub200_resnet50_deit_ti_224_pretrained.train import (
    PROTOCOL_DEFAULTS as OURS_DEFAULTS,
)
from methods.run_cub200_full_transfer_all import (
    EXPECTED_BATCH_SIZE,
    FINAL_RESULT_ORDER,
    METHOD_SCRIPTS,
    collect_best_top1,
    format_final_top1_summary,
    parse_args as parse_runner_args,
    validate_student_summary,
    validate_teacher_summary,
)


class ToyDeiT(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = torch.nn.Linear(4, 4)
        self.head = torch.nn.Linear(4, 200)


class Cub200FullTransferPipelineTest(unittest.TestCase):
    def test_explicit_deit_tiny_imagenet_source_is_locked(self) -> None:
        self.assertEqual(
            STUDENT_PRETRAINED_MODELS["deit_ti"],
            "deit_tiny_patch16_224.fb_in1k",
        )
        self.assertEqual(
            STUDENT_PRETRAINED_SOURCES["deit_ti"],
            "timm/deit_tiny_patch16_224.fb_in1k",
        )

    def test_lg_pretrained_creation_keeps_official_zero_head(self) -> None:
        fake_timm = mock.Mock()
        student = ToyDeiT()
        fake_timm.create_model.return_value = student
        result = lg_runtime.create_student(
            fake_timm,
            200,
            0.1,
            pretrained=True,
        )
        self.assertIs(result, student)
        fake_timm.create_model.assert_called_once_with(
            "deit_tiny_patch16_224.fb_in1k",
            pretrained=True,
            num_classes=200,
            drop_path_rate=0.1,
        )
        self.assertEqual(torch.count_nonzero(student.head.weight).item(), 0)
        self.assertEqual(torch.count_nonzero(student.head.bias).item(), 0)

    def test_ours_pretrained_creation_changes_only_initialization(self) -> None:
        fake_timm = mock.Mock()
        student = ToyDeiT()
        fake_timm.create_model.return_value = student
        result = ours_core.create_ours_student(
            fake_timm,
            "deit_ti",
            200,
            0.1,
            pretrained=True,
        )
        self.assertIs(result, student)
        fake_timm.create_model.assert_called_once_with(
            "deit_tiny_patch16_224.fb_in1k",
            pretrained=True,
            num_classes=200,
            drop_path_rate=0.1,
        )

    def test_existing_protocol_defaults_remain_scratch(self) -> None:
        with mock.patch(
            "sys.argv",
            ["train.py", "--dataset", "cub200"],
        ):
            lg_args = lg_runtime.parse_args()
        with mock.patch(
            "sys.argv",
            ["train.py", "--dataset", "cub200"],
        ):
            ours_args = ours_core.parse_args()
        self.assertFalse(lg_args.student_pretrained)
        self.assertFalse(ours_args.student_pretrained)

    def test_wrappers_preserve_method_protocols_and_batches(self) -> None:
        alg = dict(ALG_DEFAULTS)
        ours = dict(OURS_DEFAULTS)
        self.assertEqual(alg["--batch-size"], "128")
        self.assertEqual(alg["--base-protocol"], "lg_official")
        self.assertEqual(alg["--alg-warmup-epochs"], "0")
        self.assertEqual(alg["--alg-derivative-mode"], "paper_equations")
        self.assertEqual(ours["--batch-size"], "64")
        self.assertEqual(ours["--base-protocol"], "lg_official")
        self.assertEqual(ours["--teacher-image-size"], "224")
        self.assertIn("both_imagenet_pretrained", alg["--protocol-name"])
        self.assertIn("both_imagenet_pretrained", ours["--protocol-name"])

    def test_runner_order_has_one_vanilla_and_two_ours(self) -> None:
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

    def test_runner_compacts_all_results_and_ignores_whitespace(self) -> None:
        args = parse_runner_args(
            ["--timing-run", "   ", "--num-workers", "4"]
        )
        self.assertTrue(args.timing_run)
        results = collect_best_top1(
            {
                "teacher": {"best_top1": 80.111},
                "VanillaB128": {"best_top1": 81.222},
                "LG": {"best_top1": 82.333},
                "ALG": {"best_top1": 83.444},
                "OursB64": {"best_top1": 84.555},
                "OursB128": {"best_top1": 85.666},
            }
        )
        self.assertEqual(
            format_final_top1_summary(results),
            (
                "[FINAL_TOP1_SUMMARY_224_FULL_TRANSFER] Teacher=80.11% "
                "VanillaB128=81.22% LG=82.33% ALG=83.44% "
                "OursB64=84.56% OursB128=85.67%"
            ),
        )

    def test_identity_checks_reject_scratch_components(self) -> None:
        teacher = {
            "pretrained": True,
            "pretrained_source": (
                "torchvision.ResNet50_Weights.IMAGENET1K_V2"
            ),
            "input_resolution": 224,
            "model_name": "resnet50_cub200_imagenet1k_v2_224",
            "protocol_family": "cub200_common_transfer_resnet50_224",
        }
        validate_teacher_summary(teacher)
        with self.assertRaisesRegex(RuntimeError, "teacher identity"):
            validate_teacher_summary(dict(teacher, pretrained=False))

        student = {
            "student_pretrained": True,
            "student_pretrained_source": (
                "timm/deit_tiny_patch16_224.fb_in1k"
            ),
            "input_resolution": 224,
            "batch_size": 128,
            "teacher": None,
        }
        validate_student_summary("VanillaB128", student)
        with self.assertRaisesRegex(RuntimeError, "identity validation"):
            validate_student_summary(
                "VanillaB128",
                dict(student, student_pretrained=False),
            )

    def test_issues_show_both_accounts_and_single_line_commands(self) -> None:
        root = (
            Path(__file__).resolve().parents[1]
            / "methods"
            / "cub200_full_transfer"
        )
        timing = (root / "H200_TIMING_ISSUE.md").read_text(encoding="utf-8")
        full = (root / "H200_FULL_ISSUE.md").read_text(encoding="utf-8")
        timing_command = (
            "python methods/run_cub200_full_transfer_all.py --timing-run "
            "--num-workers 4 --output-dir "
            "/app/output/cub200_full_transfer_all_timing_seed1"
        )
        full_command = (
            "python methods/run_cub200_full_transfer_all.py --full-run "
            "--num-workers 4 --output-dir "
            "/app/output/cub200_full_transfer_all_full_seed1"
        )
        for issue in (timing, full):
            self.assertIn("bapedragon", issue)
            self.assertIn("kau-aimslab", issue)
            self.assertIn("all_students_pretrained=True", issue)
            self.assertIn("OursB64", issue)
            self.assertIn("OursB128", issue)
            self.assertNotIn("VanillaB64", issue)
        self.assertIn(timing_command, timing)
        self.assertIn(full_command, full)
        self.assertNotIn(f"{timing_command} \\", timing)
        self.assertNotIn(f"{full_command} \\", full)


if __name__ == "__main__":
    unittest.main()
