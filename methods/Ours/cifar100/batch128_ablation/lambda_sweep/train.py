#!/usr/bin/env python3
"""Run one Ours v1 CIFAR-100 batch-128 lambda control."""

from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from methods.Ours.cifar100.batch128_ablation.train import ABLATION_DEFAULTS


ALLOWED_LAMBDAS = {
    "0": (0.0, "0"),
    "0.0": (0.0, "0"),
    "0.00": (0.0, "0"),
    "0.25": (0.25, "0p25"),
    ".25": (0.25, "0p25"),
}


def has_option(arguments: list[str], option: str) -> bool:
    return any(
        argument == option or argument.startswith(f"{option}=")
        for argument in arguments
    )


def supplied_value(arguments: list[str], option: str) -> str | None:
    for index, argument in enumerate(arguments):
        if argument.startswith(f"{option}="):
            return argument.split("=", 1)[1]
        if argument == option:
            if index + 1 >= len(arguments):
                raise ValueError(f"{option} requires a value.")
            return arguments[index + 1]
    return None


def extract_lambda_value(arguments: list[str]) -> tuple[float, str, list[str]]:
    values: list[str] = []
    cleaned: list[str] = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--lambda-value":
            if index + 1 >= len(arguments):
                raise ValueError("--lambda-value requires a value.")
            values.append(arguments[index + 1])
            index += 2
            continue
        if argument.startswith("--lambda-value="):
            values.append(argument.split("=", 1)[1])
            index += 1
            continue
        cleaned.append(argument)
        index += 1

    if len(values) != 1:
        raise ValueError("Supply --lambda-value exactly once.")
    if values[0] not in ALLOWED_LAMBDAS:
        raise ValueError("This batch-128 sweep permits only lambda 0 or 0.25.")
    value, name = ALLOWED_LAMBDAS[values[0]]
    return value, name, cleaned


def inject_locked_defaults(
    arguments: list[str], lambda_value: float, lambda_name: str
) -> list[str]:
    if has_option(arguments, "--dataset"):
        raise ValueError("This wrapper fixes --dataset cifar100; remove --dataset.")
    if has_option(arguments, "--fusion-ratio"):
        raise ValueError(
            "Do not pass --fusion-ratio directly; use --lambda-value."
        )
    if has_option(arguments, "--protocol-name"):
        raise ValueError("This wrapper fixes --protocol-name.")

    dynamic_defaults = (
        (
            "--protocol-name",
            "cifar100_deit_ti_ours_researcher_sync_v1_"
            f"batch128_lambda_{lambda_name}",
        ),
        ("--fusion-ratio", str(lambda_value)),
        ("--dataset", "cifar100"),
        *ABLATION_DEFAULTS,
    )

    # The dynamic protocol name must replace the baseline ablation name.
    unique_defaults: list[tuple[str, str]] = []
    seen: set[str] = set()
    for option, value in dynamic_defaults:
        if option in seen:
            continue
        seen.add(option)
        unique_defaults.append((option, value))

    for option, expected in unique_defaults:
        actual = supplied_value(arguments, option)
        if actual is not None and actual != expected:
            raise ValueError(
                f"This control locks {option}={expected}; received {actual}."
            )

    injected = list(arguments)
    for option, value in reversed(unique_defaults):
        if not has_option(injected, option):
            injected[0:0] = [option, value]
    return injected


def main() -> None:
    try:
        lambda_value, lambda_name, remaining = extract_lambda_value(sys.argv[1:])
        sys.argv[1:] = inject_locked_defaults(
            remaining,
            lambda_value,
            lambda_name,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    align_weight = 1.0 - lambda_value
    print(
        "[BATCH128_LAMBDA_CONTROL] "
        "reference_batch128_lambda0p5_top1=82.60% "
        f"only_change=lambda lambda={lambda_value:g}",
        flush=True,
    )
    print(
        "[BATCH128_LAMBDA_LOSS] "
        f"feature_loss={lambda_value:g}*L_fuse+"
        f"{align_weight:g}*L_align adaptive_beta=unchanged",
        flush=True,
    )

    from methods.Ours.core import cli_main

    cli_main()


if __name__ == "__main__":
    main()
