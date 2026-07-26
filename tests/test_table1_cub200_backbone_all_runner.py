from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from methods.table1_cub200.run_backbone_all import (
    COMBINATIONS,
    TIMING_ESTIMATES_SECONDS,
    build_tasks,
    load_and_validate_teacher,
    validate_teacher_reference,
)
from methods.table1_cub200.teacher_contract import (
    TABLE1_TEACHER_ROOT,
    TABLE1_TEACHER_SHA256,
)


class Table1CUB200BackboneAllRunnerTests(unittest.TestCase):
    def test_convit_locked_five_task_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = argparse.Namespace(
                student="convit_ti",
                data_dir=root / "data" / "cub200",
                teacher_root=TABLE1_TEACHER_ROOT,
                output_dir=root / "output",
                num_workers=4,
            )
            tasks = build_tasks(args)

        self.assertEqual(
            [(task.name, task.method, task.batch_size) for task in tasks],
            list(COMBINATIONS),
        )
        self.assertEqual(len({task.run_name for task in tasks}), 5)
        for task in tasks:
            command = list(task.command)
            self.assertEqual(
                command[command.index("--student") + 1],
                "convit_ti",
            )
            self.assertEqual(
                command[command.index("--method") + 1],
                task.method,
            )
            self.assertEqual(
                command[command.index("--batch-size") + 1],
                str(task.batch_size),
            )
            self.assertEqual(
                command[command.index("--teacher-root") + 1],
                str(TABLE1_TEACHER_ROOT),
            )

    def test_fixed_teacher_is_hash_validated(self) -> None:
        teacher_spec = load_and_validate_teacher(TABLE1_TEACHER_ROOT)
        self.assertEqual(teacher_spec["sha256"], TABLE1_TEACHER_SHA256)
        self.assertEqual(teacher_spec["input_resolution"], 32)

    def test_vanilla_is_teacher_free_and_guided_runs_are_teacher_locked(
        self,
    ) -> None:
        teacher_spec = load_and_validate_teacher(TABLE1_TEACHER_ROOT)

        vanilla_mismatches: dict[str, object] = {}
        validate_teacher_reference(
            None,
            teacher_spec,
            vanilla=True,
            mismatches=vanilla_mismatches,
        )
        self.assertEqual(vanilla_mismatches, {})

        guided_mismatches: dict[str, object] = {}
        validate_teacher_reference(
            teacher_spec,
            teacher_spec,
            vanilla=False,
            mismatches=guided_mismatches,
        )
        self.assertEqual(guided_mismatches, {})

        wrong_vanilla: dict[str, object] = {}
        validate_teacher_reference(
            teacher_spec,
            teacher_spec,
            vanilla=True,
            mismatches=wrong_vanilla,
        )
        self.assertIn("teacher", wrong_vanilla)

        missing_guided: dict[str, object] = {}
        validate_teacher_reference(
            None,
            teacher_spec,
            vanilla=False,
            mismatches=missing_guided,
        )
        self.assertIn("teacher", missing_guided)

    def test_convit_estimate_fits_pod_limit(self) -> None:
        self.assertEqual(
            TIMING_ESTIMATES_SECONDS["convit_ti"],
            6 * 3600 + 7 * 60 + 30,
        )
        self.assertLess(
            TIMING_ESTIMATES_SECONDS["convit_ti"],
            600 * 60,
        )


if __name__ == "__main__":
    unittest.main()
