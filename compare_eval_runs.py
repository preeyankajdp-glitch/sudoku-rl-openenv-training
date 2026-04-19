"""Compare before/after Sudoku model evaluation JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two Sudoku evaluation runs.")
    parser.add_argument("baseline_json", help="Metrics JSON from the base model.")
    parser.add_argument("trained_json", help="Metrics JSON from the trained model.")
    return parser.parse_args()


def load_metrics(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fmt_delta(after: float, before: float) -> str:
    delta = after - before
    sign = "+" if delta >= 0 else ""
    return f"{after:.4f} ({sign}{delta:.4f})"


def main() -> None:
    args = parse_args()
    baseline = load_metrics(args.baseline_json)
    trained = load_metrics(args.trained_json)

    keys = [
        "success_rate",
        "avg_normalized_score",
        "avg_raw_score",
        "avg_steps",
        "avg_mistakes",
        "valid_move_rate",
        "parse_failure_rate",
    ]
    print(f"baseline: {baseline.get('model_name')}")
    print(f"trained:  {trained.get('model_name')}")
    print()
    for key in keys:
        before = float(baseline.get(key, 0.0))
        after = float(trained.get(key, 0.0))
        print(f"{key}: {before:.4f} -> {fmt_delta(after, before)}")


if __name__ == "__main__":
    main()
