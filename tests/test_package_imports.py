from __future__ import annotations

import importlib.util
from pathlib import Path


def test_root_init_imports_when_collected_as_plain_module() -> None:
    init_path = Path(__file__).resolve().parents[1] / "__init__.py"
    spec = importlib.util.spec_from_file_location("root_init_for_pytest", init_path)

    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.SudokuRlAction.__name__ == "SudokuRlAction"
    assert module.SudokuRlObservation.__name__ == "SudokuRlObservation"
    assert module.SudokuRlState.__name__ == "SudokuRlState"
