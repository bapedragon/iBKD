from __future__ import annotations

import importlib
import unittest
from unittest import mock

from methods.table1_cub200 import train


class Table1CUB200RuntimeBootstrapTests(unittest.TestCase):
    def test_complete_runtime_dependency_contract(self) -> None:
        self.assertEqual(
            train.REQUIRED_RUNTIME_IMPORTS,
            ("timm", "einops", "fvcore", "iopath", "yacs"),
        )
        requirements = (
            train.REPOSITORY_ROOT / "requirements.txt"
        ).read_text(encoding="utf-8")
        for requirement in (
            "timm==1.0.27",
            "einops==0.8.1",
            "fvcore==0.1.5.post20221221",
            "iopath==0.1.10",
            "yacs==0.1.8",
        ):
            self.assertIn(requirement, requirements)

    def test_ready_runtime_does_not_invoke_pip(self) -> None:
        with mock.patch.object(train.subprocess, "check_call") as pip_install:
            train.bootstrap_dependencies()
        pip_install.assert_not_called()

    def test_missing_iopath_installs_complete_requirements(self) -> None:
        real_import = importlib.import_module
        installed = False

        def import_with_initial_iopath_gap(module_name: str):
            if module_name == "iopath" and not installed:
                raise ModuleNotFoundError("No module named 'iopath'")
            return real_import(module_name)

        def mark_installed(*_args, **_kwargs) -> None:
            nonlocal installed
            installed = True

        with (
            mock.patch.object(
                train.importlib,
                "import_module",
                side_effect=import_with_initial_iopath_gap,
            ),
            mock.patch.object(
                train.subprocess,
                "check_call",
                side_effect=mark_installed,
            ) as pip_install,
        ):
            train.bootstrap_dependencies()

        pip_install.assert_called_once()
        command = pip_install.call_args.args[0]
        self.assertEqual(command[:4], [train.sys.executable, "-m", "pip", "install"])
        self.assertIn("-r", command)
        self.assertIn(str(train.REPOSITORY_ROOT / "requirements.txt"), command)


if __name__ == "__main__":
    unittest.main()
