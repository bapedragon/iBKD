#!/usr/bin/env python3
"""Run the paired CUB-200 both-scratch suite with 300-epoch students."""

from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from methods import run_cub200_both_scratch_100ep_all as shared  # noqa: E402


def main() -> None:
    shared.LOCKED_STUDENT_EPOCHS = 300
    shared.main()


if __name__ == "__main__":
    main()
