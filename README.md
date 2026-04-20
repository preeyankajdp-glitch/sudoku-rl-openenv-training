---
title: Sudoku RL
emoji: 🧩
colorFrom: amber
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - sudoku
  - reinforcement-learning
---

# Sudoku RL

An OpenEnv Sudoku environment backed by a real episode state machine instead of a frontend-only app.

## What It Does

- `reset(empty_boxes=...)` starts a new Sudoku puzzle with the requested number of blank cells.
- `step(...)` updates exactly one cell using either a flat `index` (`0..80`) or `row` and `column` (`1..9`).
- Every observation returns the full board, score, invalid cells, mistake reason, and a compact `status_summary`.
- Wrong moves stay on the board and are highlighted, matching the behavior in `sudoku_app.py`.
- Reward logic matches the original app: each move changes score by `round(100 / empty_boxes)`, positive for valid moves and negative for mistakes.

## Action Schema

`SudokuRlAction` accepts:

- `index`: 0-based flat cell index.
- `row`: 1-based row.
- `column`: 1-based column.
- `value`: number to place.
- `number`: alias for `value`.

Use either `index` or `row` + `column`.

## Observation Schema

`SudokuRlObservation` includes:

- `status`: `ready`, `in_progress`, `invalid_move`, or `solved`
- `message`: human-readable result of the last reset or move
- `status_summary`: compact state summary for agents
- `board`: current board after the update
- `invalid_cells` and `invalid_indices`
- `mistake_reason`
- `score`, `score_delta`, `score_step`
- `moves`, `mistakes`
- `board_text`

## Quick Start

```python
from sudoku_rl import SudokuRlAction, SudokuRlEnv

with SudokuRlEnv(base_url="http://localhost:8000") as env:
    reset_result = env.reset(empty_boxes=35)
    print(reset_result.observation.board_text)

    step_result = env.step(SudokuRlAction(index=0, value=5))
    print(step_result.observation.status)
    print(step_result.observation.message)
```

## Local Development

```bash
cd sudoku_rl
uv sync
uv run --extra dev pytest
uv run openenv validate .
uv run --project . server
```

The custom UI is available at `/web` and mirrors the original Sudoku Gradio layout while talking to the OpenEnv backend.

## Training And Before/After Evaluation

This repo keeps Sudoku as the OpenEnv environment and uses plain PyTorch plus
Hugging Face Transformers for model evaluation and supervised fine-tuning. The
training script creates oracle trajectories from the local environment, then
teaches a causal language model to emit the next move as JSON.

Install the training extras in your Lightning AI Studio or notebook terminal:

```bash
cd sudoku_rl
pip install -e ".[train]"
```

Run a baseline evaluation before training:

```bash
sudoku-eval \
  --model-name Qwen/Qwen3-0.6B \
  --episodes 20 \
  --empty-boxes 35 \
  --output-json outputs/evals/baseline.json
```

By default, parse failures are penalized as invalid model actions. This keeps
baseline metrics honest: helper fallback code does not get credit for solving a
puzzle when the model failed to emit parseable JSON.

Fine-tune on oracle-generated Sudoku traces:

```bash
sudoku-train \
  --model-name Qwen/Qwen3-0.6B \
  --episodes 128 \
  --empty-boxes 35 \
  --epochs 1 \
  --output-dir outputs/checkpoints/sudoku-sft
```

Evaluate the trained checkpoint on the same benchmark shape:

```bash
sudoku-eval \
  --model-name outputs/checkpoints/sudoku-sft \
  --episodes 20 \
  --empty-boxes 35 \
  --output-json outputs/evals/trained.json
```

Compare the before and after metrics:

```bash
sudoku-compare outputs/evals/baseline.json outputs/evals/trained.json
```

Launch a side-by-side playback UI:

```bash
sudoku-playback-ui --host 0.0.0.0 --port 7860
```

Open the exposed port in Lightning AI, enter the baseline and trained model
paths, and click `Generate Playback` to watch both agents place numbers on the
same puzzle.

The main metrics are:

- `success_rate`: fraction of episodes solved.
- `avg_normalized_score`: score normalized to roughly 0..1.
- `valid_move_rate`: fraction of generated actions accepted by the environment.
- `parse_failure_rate`: fraction of model generations that could not be parsed as action JSON.

This is supervised fine-tuning from an oracle, not PPO/GRPO yet. It is the
fastest first step for checking whether the model learns the action format and
Sudoku move policy before adding a full RL update loop.

## Lightning AI / Notebook Usage

If your notebook kernel starts inside the `sudoku_rl` folder, Python does not
normally see the package's parent directory, so `from sudoku_rl import ...`
can fail with `ModuleNotFoundError`.

This repo includes a local compatibility shim so that import works when you
open the project directory directly, but you still need the project
dependencies installed in the notebook environment:

```bash
cd sudoku_rl
pip install -e .
```

If you prefer `uv`:

```bash
cd sudoku_rl
uv sync
```

For training in Lightning AI, prefer:

```bash
cd sudoku_rl
pip install -e ".[train]"
```
