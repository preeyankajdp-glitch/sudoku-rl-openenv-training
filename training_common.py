"""Shared prompt, oracle, and parsing helpers for Sudoku model training."""

from __future__ import annotations

import json
import re
import textwrap
from dataclasses import dataclass
from typing import Any, Iterable

from sudoku_rl.models import SudokuRlAction, SudokuRlObservation
from sudoku_rl.server.sudoku_rl_environment import SudokuRlEnvironment


SYSTEM_PROMPT = textwrap.dedent(
    """
    You are solving a Sudoku puzzle one move at a time.

    Return exactly one JSON object with this schema:
    {"row": <1-9>, "column": <1-9>, "value": <1-9>}

    Rules:
    - Choose one editable empty cell.
    - The editable empty cells are listed in the user prompt; choose only one of those row and column pairs.
    - If the previous action was invalid, do not repeat the same row, column, and value.
    - Choose a value from 1 to 9.
    - Do not include markdown, explanations, code fences, or extra text.
    """
).strip()


@dataclass(frozen=True)
class TrainingExample:
    """One prompt/response pair generated from the environment oracle."""

    system_prompt: str
    user_prompt: str
    response: str


def render_action_json(action: SudokuRlAction) -> str:
    """Render the canonical assistant response for a Sudoku action."""

    return json.dumps(
        {"row": action.row, "column": action.column, "value": action.value},
        separators=(",", ": "),
    )


def visible_candidates(board: list[list[int]], row_index: int, column_index: int) -> list[int]:
    """Return Sudoku candidates that do not violate currently visible values."""

    if board[row_index][column_index] != 0:
        return []

    used_values = set(board[row_index])
    used_values.update(board[current_row][column_index] for current_row in range(9))

    box_row = (row_index // 3) * 3
    box_column = (column_index // 3) * 3
    for current_row in range(box_row, box_row + 3):
        for current_column in range(box_column, box_column + 3):
            used_values.add(board[current_row][current_column])

    return [value for value in range(1, 10) if value not in used_values]


def editable_cells_block(observation: SudokuRlObservation) -> str:
    """Render editable empty cells using 1-based coordinates."""

    lines: list[str] = []
    candidate_board = [row[:] for row in observation.board]
    for row_index, column_index in observation.invalid_cells:
        if 0 <= row_index < 9 and 0 <= column_index < 9:
            candidate_board[row_index][column_index] = 0

    for row_index in range(9):
        for column_index in range(9):
            if observation.initial_puzzle[row_index][column_index] != 0:
                continue
            if observation.board[row_index][column_index] != 0 and [row_index, column_index] not in observation.invalid_cells:
                continue
            candidates = visible_candidates(candidate_board, row_index, column_index)
            candidates_text = ", ".join(str(value) for value in candidates) if candidates else "none"
            lines.append(
                f"- row={row_index + 1}, column={column_index + 1}, visible_candidates=[{candidates_text}]"
            )

    return "\n".join(lines) if lines else "None"


def build_user_prompt(
    *,
    step: int,
    observation: SudokuRlObservation,
    history: list[str] | None = None,
) -> str:
    """Build the model-facing prompt from an OpenEnv observation."""

    history_block = "\n".join((history or [])[-8:]) if history else "None"
    invalid_block = observation.invalid_cells if observation.invalid_cells else "[]"
    editable_block = editable_cells_block(observation)
    return textwrap.dedent(
        f"""
        Step: {step}
        Status: {observation.status}
        Score: {observation.score}
        Score delta from last move: {observation.score_delta}
        Moves attempted: {observation.moves}
        Mistakes: {observation.mistakes}
        Empty cells remaining: {observation.empty_cells_remaining}
        Incorrect cells: {observation.incorrect_cells}
        Invalid cells as 0-based [row, column] pairs: {invalid_block}
        Last environment message: {observation.message}

        Current board, where . means empty:
        {observation.board_text}

        Editable empty cells and visible candidates:
        {editable_block}

        Recent actions:
        {history_block}

        Return the next move as JSON only. The row and column must be one of the listed editable empty cells.
        If a recent action was invalid, choose a different value or a different listed cell.
        """
    ).strip()


def parse_action(text: str) -> SudokuRlAction | None:
    """Parse a model response into a Sudoku action, if possible."""

    cleaned = (text or "").strip()
    if not cleaned:
        return None

    candidates: list[dict[str, Any]] = []
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            candidates.append(parsed)
    except json.JSONDecodeError:
        pass

    for match in re.finditer(r"\{[^{}]+\}", cleaned, flags=re.DOTALL):
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            candidates.append(parsed)

    row_match = re.search(r'"?row"?\s*[:=]\s*([1-9])\b', cleaned, flags=re.IGNORECASE)
    column_match = re.search(r'"?(?:column|col)"?\s*[:=]\s*([1-9])\b', cleaned, flags=re.IGNORECASE)
    value_match = re.search(r'"?(?:value|number)"?\s*[:=]\s*([1-9])\b', cleaned, flags=re.IGNORECASE)
    if row_match and column_match and value_match:
        candidates.append(
            {
                "row": int(row_match.group(1)),
                "column": int(column_match.group(1)),
                "value": int(value_match.group(1)),
            }
        )

    for candidate in candidates:
        try:
            row = int(candidate["row"])
            column = int(candidate.get("column", candidate.get("col")))
            value = int(candidate.get("value", candidate.get("number")))
        except (KeyError, TypeError, ValueError):
            continue
        if 1 <= row <= 9 and 1 <= column <= 9 and 1 <= value <= 9:
            return SudokuRlAction(row=row, column=column, value=value)

    return None


def oracle_action(observation: SudokuRlObservation, solution_board: list[list[int]]) -> SudokuRlAction:
    """Choose the next correct action from the hidden solution for imitation data."""

    invalid_targets = [
        (int(row_index), int(column_index))
        for row_index, column_index in observation.invalid_cells
        if 0 <= row_index < 9 and 0 <= column_index < 9
    ]
    empty_targets = [
        (row_index, column_index)
        for row_index in range(9)
        for column_index in range(9)
        if observation.initial_puzzle[row_index][column_index] == 0
        and observation.board[row_index][column_index] != solution_board[row_index][column_index]
    ]

    target_cells = invalid_targets + [
        cell for cell in empty_targets if cell not in set(invalid_targets)
    ]
    if not target_cells:
        raise ValueError("No editable unsolved cells remain.")

    row_index, column_index = target_cells[0]
    return SudokuRlAction(
        row=row_index + 1,
        column=column_index + 1,
        value=solution_board[row_index][column_index],
    )


def fallback_action(observation: SudokuRlObservation) -> SudokuRlAction:
    """Return a deterministic legal-shaped action when model output is unparseable."""

    candidate_board = [row[:] for row in observation.board]
    for row_index, column_index in observation.invalid_cells:
        if 0 <= row_index < 9 and 0 <= column_index < 9:
            candidate_board[row_index][column_index] = 0

    for row_index in range(9):
        for column_index in range(9):
            if observation.initial_puzzle[row_index][column_index] != 0:
                continue
            if observation.board[row_index][column_index] != 0 and [row_index, column_index] not in observation.invalid_cells:
                continue
            candidates = visible_candidates(candidate_board, row_index, column_index)
            value = candidates[0] if candidates else 1
            return SudokuRlAction(row=row_index + 1, column=column_index + 1, value=value)
    return SudokuRlAction(row=1, column=1, value=1)


def iter_oracle_examples(
    *,
    episodes: int,
    empty_boxes: int,
    seed: int,
    max_steps: int | None = None,
) -> Iterable[TrainingExample]:
    """Yield supervised examples from oracle-solved Sudoku episodes."""

    for episode_index in range(episodes):
        env = SudokuRlEnvironment()
        observation = env.reset(seed=seed + episode_index, empty_boxes=empty_boxes)
        history: list[str] = []
        step_limit = max_steps or max(observation.empty_boxes, 1)

        for step in range(1, step_limit + 1):
            if observation.done:
                break
            action = oracle_action(observation, env.solution_board)
            yield TrainingExample(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=build_user_prompt(
                    step=step,
                    observation=observation,
                    history=history,
                ),
                response=render_action_json(action),
            )

            observation = env.step(action)
            history.append(
                f"step={step} row={action.row} column={action.column} "
                f"value={action.value} reward={observation.reward:+.2f} "
                f"status={observation.status}"
            )
