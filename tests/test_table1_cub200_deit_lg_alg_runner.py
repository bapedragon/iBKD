from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from methods.table1_cub200.run_deit_lg_alg import (
    ALG_RUN_NAME,
    LG_RUN_NAME,
    TEACHER_RUN_NAME,
    TIMING_ESTIMATE_SECONDS,
    build_tasks,
)


class Table1CUB200DeiTLGALGRunnerTests(unittest.TestCase):
    def test_locked_three_task_sequence_and_teacher_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = argparse.Namespace(
                data_dir=root / "data" / "cub200",
                output_dir=root / "output",
                num_workers=4,
            )
            tasks, teacher_root = build_tasks(args)

        self.assertEqual([task.name for task in tasks], ["Teacher", "LG", "ALG"])
        self.assertEqual(
            teacher_root,
            root / "output" / "teacher" / TEACHER_RUN_NAME,
        )
        self.assertEqual(tasks[0].summary_path, teacher_root / "summary.json")
        for task, method, run_name in (
            (tasks[1], "lg", LG_RUN_NAME),
            (tasks[2], "alg", ALG_RUN_NAME),
        ):
            command = list(task.command)
            self.assertIn("--full-run", command)
            self.assertEqual(command[command.index("--student") + 1], "deit_ti")
            self.assertEqual(command[command.index("--method") + 1], method)
            self.assertEqual(command[command.index("--batch-size") + 1], "128")
            self.assertEqual(
                command[command.index("--teacher-root") + 1],
                str(teacher_root),
            )
            self.assertEqual(command[command.index("--run-name") + 1], run_name)

    def test_timing_estimate_fits_pod_limit(self) -> None:
        self.assertEqual(TIMING_ESTIMATE_SECONDS, 2 * 3600 + 53 * 60 + 58)
        self.assertLess(TIMING_ESTIMATE_SECONDS, 600 * 60)


if __name__ == "__main__":
    unittest.main()
