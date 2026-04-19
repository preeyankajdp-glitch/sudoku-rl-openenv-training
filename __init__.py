# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sudoku Rl Environment."""

from pathlib import Path
import sys


def _ensure_package_root_on_path() -> None:
    package_root = str(Path(__file__).resolve().parent)
    if package_root not in sys.path:
        sys.path.insert(0, package_root)


try:
    from .models import SudokuRlAction, SudokuRlObservation, SudokuRlState
except ImportError:
    _ensure_package_root_on_path()
    from models import SudokuRlAction, SudokuRlObservation, SudokuRlState

__all__ = [
    "SudokuRlAction",
    "SudokuRlObservation",
    "SudokuRlState",
    "SudokuRlEnv",
]


def __getattr__(name: str):
    if name == "SudokuRlEnv":
        try:
            from .client import SudokuRlEnv
        except ImportError:
            _ensure_package_root_on_path()
            from client import SudokuRlEnv

        return SudokuRlEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
