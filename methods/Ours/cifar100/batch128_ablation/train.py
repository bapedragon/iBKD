#!/usr/bin/env python3
"""Run the CIFAR-100 Ours v1 batch-size-128 ablation."""

from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from methods.Ours.cifar100.train import PROTOCOL_DEFAULTS
from methods.Ours.core import cli_main


# Keep every researcher-sync-v1 setting unchanged except the train batch size.
# The protocol name is metadata used to prevent this ablation from being mixed
# with the selected batch-64 result.
ABLATION_OVERRIDES = {
    "--protocol-name": "cifar100_deit_ti_ours_researcher_sync_v1_batch128",
    "--batch-size": "128",
}
ABLATION_DEFAULTS = tuple(
    (option, ABLATION_OVERRIDES.get(option, value))
    for option, value in PROTOCOL_DEFAULTS
)


def has_option(option: str) -> bool:
    return any(
        argument == option or argument.startswith(f"{option}=")
        for argument in sys.argv[1:]
    )


def supplied_value(option: str) -> str | None:
    arguments = sys.argv[1:]
    for index, argument in enumerate(arguments):
        if argument.startswith(f"{option}="):
            return argument.split("=", 1)[1]
        if argument == option:
            if index + 1 >= len(arguments):
                raise SystemExit(f"{option} requires a value.")
            return arguments[index + 1]
    return None


if __name__ == "__main__":
    if has_option("--dataset"):
        raise SystemExit("This wrapper fixes --dataset cifar100; remove --dataset.")

    for option, expected in ABLATION_DEFAULTS:
        actual = supplied_value(option)
        if actual is not None and actual != expected:
            raise SystemExit(
                f"This ablation locks {option}={expected}; received {actual}."
            )

    sys.argv[1:1] = ["--dataset", "cifar100"]
    for option, value in reversed(ABLATION_DEFAULTS):
        if not has_option(option):
            sys.argv[1:1] = [option, value]
    cli_main()
