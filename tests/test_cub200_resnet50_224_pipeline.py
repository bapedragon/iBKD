from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn

from methods.ALG.cub200_resnet50_224.train import (
    PROTOCOL_DEFAULTS as ALG_224_DEFAULTS,
)
from methods.LG.official_lg import LocalityGuidance
from methods.LG import runtime as lg_runtime
from methods.Ours.cub200_resnet50_224.train import (
    PROTOCOL_DEFAULTS as OURS_224_DEFAULTS,
)
from methods.Ours import core as ours_core
from methods.Ours.ours import Ours
from methods.run_cub200_resnet50_224_lg_alg_ours import (
    STUDENT_SCRIPTS,
    collect_best_top1,
    format_final_top1_summary,
    parse_args as parse_runner_args,
)
from teachers import verify_checkpoints
from teachers.train_teacher_cub200_resnet50_224 import (
    FEATURE_CHANNELS,
    FEATURE_SPATIAL_SIZES,
    ResNet50CUB200,
)


class TinyNamedTeacher(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return torch.zeros(images.shape[0], 200) * self.weight


class Cub200ResNet50224PipelineTest(unittest.TestCase):
    def test_teacher_exposes_locked_late_stage_features(self) -> None:
        model = ResNet50CUB200(pretrained=False).eval()
        with torch.inference_mode():
            features = model.forward_features(torch.zeros(1, 3, 224, 224))
            logits = model(torch.zeros(1, 3, 224, 224))
        self.assertEqual(
            [tuple(feature.shape) for feature in features],
            [
                (1, channels, size, size)
                for channels, size in zip(
                    FEATURE_CHANNELS,
                    FEATURE_SPATIAL_SIZES,
                    strict=True,
                )
            ],
        )
        self.assertEqual(tuple(logits.shape), (1, 200))

    def test_lg_and_ours_accept_resnet50_feature_contract(self) -> None:
        teacher_features = [
            torch.randn(1, 512, 28, 28),
            torch.randn(1, 1024, 14, 14),
            torch.randn(1, 2048, 7, 7),
        ]
        lg_student = [torch.randn(1, 192, 14, 14) for _ in range(3)]
        lg = LocalityGuidance(teacher_channels=FEATURE_CHANNELS)
        lg_loss, aligned_student, _ = lg(lg_student, teacher_features)
        self.assertTrue(bool(torch.isfinite(lg_loss)))
        self.assertEqual(
            [tuple(feature.shape) for feature in aligned_student],
            [
                (1, 512, 28, 28),
                (1, 1024, 14, 14),
                (1, 2048, 14, 14),
            ],
        )

        ours = Ours(teacher_channels=FEATURE_CHANNELS, num_heads=4)
        ours_student = [torch.randn(1, 192, 14, 14) for _ in range(12)]
        with torch.no_grad():
            alignment, fusion, _, _, targets = ours(
                ours_student, teacher_features
            )
        self.assertTrue(bool(torch.isfinite(alignment + fusion)))
        self.assertEqual(
            [tuple(feature.shape) for feature in targets],
            [
                (1, 512, 28, 28),
                (1, 1024, 14, 14),
                (1, 2048, 14, 14),
            ],
        )

    def test_named_manifest_selects_resnet50_factory_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "teacher.pt"
            model = TinyNamedTeacher()
            payload = {
                "epoch": 2,
                "accuracy": 3.5,
                "dataset": "cub200",
                "num_classes": 200,
                "model_name": "resnet50_cub200_imagenet1k_v2_224",
                "model": model.state_dict(),
            }
            torch.save(payload, checkpoint)
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            manifest = {
                "version": 1,
                "teachers": {
                    "cub200": {
                        "selected_kind": "best",
                        "checkpoint": checkpoint.name,
                        "sha256": digest,
                        "epoch": 2,
                        "top1": 3.5,
                        "num_classes": 200,
                        "model_name": "resnet50_cub200_imagenet1k_v2_224",
                        "input_resolution": 224,
                        "feature_channels": list(FEATURE_CHANNELS),
                    }
                },
            }
            (root / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with mock.patch.dict(
                verify_checkpoints.NAMED_MODEL_FACTORIES,
                {
                    "resnet50_cub200_imagenet1k_v2_224": TinyNamedTeacher,
                },
            ):
                loaded, loaded_payload, spec = verify_checkpoints.load_teacher(
                    "cub200", checkpoint_root=root
                )
            self.assertIsInstance(loaded, TinyNamedTeacher)
            self.assertEqual(loaded_payload["epoch"], 2)
            self.assertEqual(spec["input_resolution"], 224)

    def test_224_wrappers_are_separate_and_locked(self) -> None:
        self.assertEqual(
            STUDENT_SCRIPTS,
            {
                "LG": "methods/LG/cub200_resnet50_224/train.py",
                "ALG": "methods/ALG/cub200_resnet50_224/train.py",
                "Ours": "methods/Ours/cub200_resnet50_224/train.py",
            },
        )
        alg = dict(ALG_224_DEFAULTS)
        ours = dict(OURS_224_DEFAULTS)
        self.assertEqual(alg["--teacher-image-size"], "224")
        self.assertEqual(ours["--teacher-image-size"], "224")
        self.assertIn("resnet50_224", alg["--protocol-name"])
        self.assertIn("resnet50_224", ours["--protocol-name"])

    def test_lg_runtime_accepts_only_cub_for_224_teacher(self) -> None:
        with mock.patch(
            "sys.argv",
            [
                "train.py",
                "--dataset",
                "cub200",
                "--teacher-image-size",
                "224",
            ],
        ):
            args = lg_runtime.parse_args()
        args.method = "LG"
        lg_runtime.finalize_args(args)
        self.assertEqual(args.teacher_image_size, 224)

        with mock.patch(
            "sys.argv",
            [
                "train.py",
                "--dataset",
                "cifar100",
                "--teacher-image-size",
                "224",
            ],
        ):
            args = lg_runtime.parse_args()
        args.method = "LG"
        with self.assertRaisesRegex(ValueError, "locked to CUB-200"):
            lg_runtime.finalize_args(args)

    def test_ours_224_requires_cub_and_imagenet_normalized_path(self) -> None:
        with mock.patch(
            "sys.argv",
            [
                "train.py",
                "--dataset",
                "cub200",
                "--teacher-image-size",
                "224",
                "--base-protocol",
                "lg_official",
            ],
        ):
            args = ours_core.parse_args()
        ours_core.finalize_args(args)

        args.base_protocol = "common"
        with self.assertRaisesRegex(ValueError, "lg_official"):
            ours_core.finalize_args(args)

    def test_runner_ignores_whitespace_and_compacts_224_results(self) -> None:
        args = parse_runner_args(["--timing-run", "   ", "--num-workers", "4"])
        self.assertTrue(args.timing_run)
        results = collect_best_top1(
            {
                "teacher": {"best_top1": 10.111},
                "LG": {"best_top1": 20.222},
                "ALG": {"best_top1": 30.333},
                "Ours": {"best_top1": 40.444},
            }
        )
        self.assertEqual(
            format_final_top1_summary(results),
            (
                "[FINAL_TOP1_SUMMARY_224] Teacher224=10.11% LG224=20.22% "
                "ALG224=30.33% Ours224=40.44%"
            ),
        )

    def test_issue_keeps_224_identity_and_both_accounts_copyable(self) -> None:
        issue = (
            Path(__file__).resolve().parents[1]
            / "methods"
            / "cub200_resnet50_224"
            / "H200_ISSUE.md"
        ).read_text(encoding="utf-8")
        self.assertIn("bapedragon", issue)
        self.assertIn("kau-aimslab", issue)
        self.assertIn("ImageNet-pretrained ResNet50-224 teacher", issue)
        self.assertIn("teacher_input=224 student_input=224", issue)
        timing_command = (
            "python methods/run_cub200_resnet50_224_lg_alg_ours.py "
            "--timing-run --num-workers 4 --output-dir "
            "/app/output/cub200_resnet50_224_lg_alg_ours_timing_seed1"
        )
        self.assertIn(timing_command, issue)
        self.assertNotIn(f"{timing_command} \\", issue)


if __name__ == "__main__":
    unittest.main()
