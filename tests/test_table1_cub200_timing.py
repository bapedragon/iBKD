from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

import torch

from methods.Ours.ours import Ours
from methods.table1_cub200.adapters import (
    OfficialLGFeatureLoss,
    OursAllBlockAdapter,
)
from methods.table1_cub200.backbones import BACKBONES, create_student
from methods.table1_cub200.run_timing import build_tasks


class Table1CUB200TimingTests(unittest.TestCase):
    def test_seven_official_backbone_contracts(self) -> None:
        expected = {
            "deit_ti": (12, (0, 6, 11)),
            "convit_ti": (12, (0, 6, 11)),
            "cvt_13": (13, (0, 6, 11)),
            "pit_ti": (12, (0, 6, 11)),
            "pvtv2_b0": (8, (0, 3, 7)),
            "t2t_vit_7": (7, (0, 3, 6)),
            "t2t_vit_14": (14, (0, 7, 13)),
        }
        self.assertEqual(set(BACKBONES), set(expected))
        for key, (depth, indexes) in expected.items():
            model, spec = create_student(key)
            self.assertEqual(len(model.feature_dims), depth)
            self.assertEqual(spec.selected_feature_indices, indexes)
            self.assertEqual(model.num_classes, 200)

    def test_complete_matrix_has_36_unique_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = argparse.Namespace(
                data_dir=root / "data",
                teacher_root=root / "teacher",
                output_dir=root / "output",
                num_workers=0,
            )
            tasks = build_tasks(args)
        self.assertEqual(len(tasks), 36)
        self.assertEqual(len({task.run_name for task in tasks}), 36)
        self.assertEqual(sum(task.kind == "teacher" for task in tasks), 1)
        self.assertEqual(sum(task.kind == "student" for task in tasks), 35)
        for student_key in BACKBONES:
            student_tasks = [
                task for task in tasks if task.student_key == student_key
            ]
            self.assertEqual(
                [(task.method, task.batch_size) for task in student_tasks],
                [
                    ("vanilla", 128),
                    ("lg", 128),
                    ("alg", 128),
                    ("ours", 64),
                    ("ours", 128),
                ],
            )

    def test_feature_adapters_are_finite_for_every_backbone(self) -> None:
        teacher_features = [
            torch.randn(2, 16, 32, 32),
            torch.randn(2, 32, 16, 16),
            torch.randn(2, 64, 8, 8),
        ]
        grid_cycle = (56, 28, 14, 7)
        for key, spec in BACKBONES.items():
            model, _ = create_student(key)
            student_features = [
                torch.randn(2, int(channels), grid_cycle[index % 4], grid_cycle[index % 4])
                for index, channels in enumerate(model.feature_dims)
            ]
            lg = OfficialLGFeatureLoss(
                model.feature_dims,
                spec.selected_feature_indices,
            )
            self.assertTrue(torch.isfinite(lg(student_features, teacher_features)))

            adapter = OursAllBlockAdapter(model.feature_dims)
            adapted = adapter(student_features)
            self.assertEqual(len(adapted), spec.depth)
            self.assertTrue(
                all(tuple(feature.shape[1:]) == (192, 14, 14) for feature in adapted)
            )
            ours = Ours(num_student_blocks=spec.depth)
            alignment, fusion, *_ = ours(adapted, teacher_features)
            self.assertTrue(torch.isfinite(alignment + fusion))


if __name__ == "__main__":
    unittest.main()
