from __future__ import annotations

from sudoku_rl.training_common import editable_cells_block, iter_oracle_examples, parse_action
from sudoku_rl.server.sudoku_rl_environment import SudokuRlEnvironment


def test_parse_action_reads_json_object() -> None:
    action = parse_action('{"row": 2, "column": 3, "value": 4}')

    assert action is not None
    assert action.row == 2
    assert action.column == 3
    assert action.value == 4


def test_parse_action_reads_embedded_text() -> None:
    action = parse_action('next move: {"row": 9, "col": 1, "number": 7}')

    assert action is not None
    assert action.row == 9
    assert action.column == 1
    assert action.value == 7


def test_parse_action_rejects_out_of_range_values() -> None:
    assert parse_action('{"row": 10, "column": 9, "value": 3}') is None


def test_oracle_examples_use_json_responses() -> None:
    examples = list(iter_oracle_examples(episodes=1, empty_boxes=2, seed=123))

    assert len(examples) == 2
    assert "Current board" in examples[0].user_prompt
    assert "Editable empty cells" in examples[0].user_prompt
    assert examples[0].response.startswith('{"row": ')
    assert '"column": ' in examples[0].response
    assert '"value": ' in examples[0].response


def test_editable_cells_block_lists_only_empty_editable_cells() -> None:
    env = SudokuRlEnvironment()
    observation = env.reset(seed=1000, empty_boxes=5)

    block = editable_cells_block(observation)

    assert "row=9, column=9" not in block
    for row_index in range(9):
        for column_index in range(9):
            if observation.board[row_index][column_index] == 0:
                assert f"row={row_index + 1}, column={column_index + 1}" in block
