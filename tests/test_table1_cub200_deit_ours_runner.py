from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from methods.table1_cub200.run_deit_ours import (
    RUN_NAMES,
    TIMING_ESTIMATE_SECONDS,
    build_tasks,
    load_completed_results,
    load_and_validate_teacher,
)
from methods.table1_cub200.teacher_contract import (
    TABLE1_TEACHER_ROOT,
    TABLE1_TEACHER_SHA256,
)


class Table1CUB200DeiTOursRunnerTests(unittest.TestCase):
    def test_locked_two_task_sequence_and_batches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = argparse.Namespace(
                data_dir=root / "data" / "cub200",
                teacher_root=TABLE1_TEACHER_ROOT,
                output_dir=root / "output",
                num_workers=4,
            )
            tasks = build_tasks(args)

        self.assertEqual([task.name for task in tasks], ["OursB64", "OursB128"])
        self.assertEqual([task.batch_size for task in tasks], [64, 128])
        for task in tasks:
            command = list(task.command)
            self.assertIn("--full-run", command)
            self.assertEqual(command[command.index("--student") + 1], "deit_ti")
            self.assertEqual(command[command.index("--method") + 1], "ours")
            self.assertEqual(
                command[command.index("--batch-size") + 1],
                str(task.batch_size),
            )
            self.assertEqual(
                command[command.index("--teacher-root") + 1],
                str(TABLE1_TEACHER_ROOT),
            )
            self.assertEqual(
                command[command.index("--run-name") + 1],
                RUN_NAMES[task.batch_size],
            )
            self.assertEqual(
                task.checkpoint_path,
                root
                / "output"
                / "students"
                / RUN_NAMES[task.batch_size]
                / "student_best.pt",
            )

    def test_fixed_teacher_file_and_manifest_are_hash_validated(self) -> None:
        teacher_spec = load_and_validate_teacher(TABLE1_TEACHER_ROOT)
        self.assertEqual(teacher_spec["sha256"], TABLE1_TEACHER_SHA256)
        self.assertEqual(teacher_spec["input_resolution"], 32)
        self.assertIs(teacher_spec["pretrained"], False)

    def test_modified_teacher_manifest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = json.loads(
                (TABLE1_TEACHER_ROOT / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            manifest["teachers"]["cub200"]["sha256"] = "0" * 64
            (root / "manifest.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "build-543"):
                load_and_validate_teacher(root)

    def test_completed_lg_alg_results_share_the_fixed_teacher(self) -> None:
        teacher_spec = load_and_validate_teacher(TABLE1_TEACHER_ROOT)
        results = load_completed_results(teacher_spec)
        self.assertAlmostEqual(results["LG"], 44.511563686572316)
        self.assertAlmostEqual(results["ALG"], 47.704521919226785)

    def test_timing_estimate_fits_pod_limit(self) -> None:
        self.assertEqual(TIMING_ESTIMATE_SECONDS, 2 * 3600 + 23 * 60 + 56)
        self.assertLess(TIMING_ESTIMATE_SECONDS, 600 * 60)


if __name__ == "__main__":
    unittest.main()
