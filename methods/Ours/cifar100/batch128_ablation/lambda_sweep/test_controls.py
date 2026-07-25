from __future__ import annotations

import unittest

from methods.Ours.cifar100.batch128_ablation.lambda_sweep.train import (
    extract_lambda_value,
    inject_locked_defaults,
)


def option_value(arguments: list[str], option: str) -> str:
    index = arguments.index(option)
    return arguments[index + 1]


class Batch128LambdaControlsTest(unittest.TestCase):
    def test_lambda_zero_changes_only_fusion_ratio(self) -> None:
        value, name, remaining = extract_lambda_value(["--lambda-value", "0"])
        injected = inject_locked_defaults(remaining, value, name)
        self.assertEqual(option_value(injected, "--batch-size"), "128")
        self.assertEqual(option_value(injected, "--student-epochs"), "300")
        self.assertEqual(option_value(injected, "--seed"), "1")
        self.assertEqual(option_value(injected, "--fusion-ratio"), "0.0")

    def test_lambda_quarter_changes_only_fusion_ratio(self) -> None:
        value, name, remaining = extract_lambda_value(
            ["--lambda-value=0.25"]
        )
        injected = inject_locked_defaults(remaining, value, name)
        self.assertEqual(option_value(injected, "--batch-size"), "128")
        self.assertEqual(option_value(injected, "--fusion-ratio"), "0.25")

    def test_other_lambda_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "only lambda 0 or 0.25"):
            extract_lambda_value(["--lambda-value", "0.5"])

    def test_conflicting_batch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "batch-size=128"):
            inject_locked_defaults(["--batch-size", "64"], 0.0, "0")


if __name__ == "__main__":
    unittest.main()
