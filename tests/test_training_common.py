from __future__ import annotations

from sudoku_rl.training_common import iter_oracle_examples, parse_action


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


def test_oracle_examples_use_json_responses() -> None:
    examples = list(iter_oracle_examples(episodes=1, empty_boxes=2, seed=123))

    assert len(examples) == 2
    assert "Current board" in examples[0].user_prompt
    assert examples[0].response.startswith('{"row": ')
    assert '"column": ' in examples[0].response
    assert '"value": ' in examples[0].response
