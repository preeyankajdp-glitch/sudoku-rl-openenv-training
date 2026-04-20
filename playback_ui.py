"""Gradio playback UI for comparing baseline and trained Sudoku agents."""

from __future__ import annotations

import argparse
import html
import json
import random
import uuid
from dataclasses import dataclass
from typing import Any

from sudoku_rl.evaluate_model import generate_text, invalid_parse_action
from sudoku_rl.server.sudoku_rl_environment import SudokuRlEnvironment
from sudoku_rl.train_transformers import choose_device, render_prompt
from sudoku_rl.training_common import (
    SYSTEM_PROMPT,
    TrainingExample,
    build_user_prompt,
    fallback_action,
    parse_action,
    render_action_json,
)


@dataclass
class AgentArtifacts:
    model: Any
    tokenizer: Any
    device: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a Sudoku baseline-vs-trained playback UI.")
    parser.add_argument("--host", default="0.0.0.0", help="Host for the Gradio server.")
    parser.add_argument("--port", type=int, default=7860, help="Port for the Gradio server.")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share URL.")
    return parser.parse_args()


def load_agent(model_name: str, device: str) -> AgentArtifacts:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.eos_token is None:
        tokenizer.eos_token = tokenizer.pad_token or ""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    model.to(device)
    model.eval()
    torch.manual_seed(0)
    return AgentArtifacts(model=model, tokenizer=tokenizer, device=device)


def unload_agent(agent: AgentArtifacts) -> None:
    import gc
    import torch

    del agent.model
    del agent.tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def action_key(action: Any) -> tuple[int | None, int | None, int | None]:
    return (action.row, action.column, action.value)


def frame_from_observation(
    *,
    step: int,
    board: list[list[int]],
    status: str,
    score: int,
    action: str,
    raw_output: str,
    source: str,
    reward: float,
    move_valid: bool | None,
    message: str,
) -> dict[str, Any]:
    return {
        "step": step,
        "board": board,
        "status": status,
        "score": score,
        "action": action,
        "raw_output": raw_output,
        "source": source,
        "reward": reward,
        "move_valid": move_valid,
        "message": message,
    }


def rollout_agent(
    *,
    model_name: str,
    label: str,
    seed: int,
    empty_boxes: int,
    max_steps: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: str,
    repeat_guard: bool,
    fallback_policy: str,
) -> dict[str, Any]:
    env = SudokuRlEnvironment()
    observation = env.reset(seed=seed, empty_boxes=empty_boxes)
    agent = load_agent(model_name, device=device)
    history: list[str] = []
    frames: list[dict[str, Any]] = [
        frame_from_observation(
            step=0,
            board=observation.board,
            status=observation.status,
            score=observation.score,
            action="",
            raw_output="",
            source="reset",
            reward=0.0,
            move_valid=None,
            message=observation.message,
        )
    ]
    invalid_actions: set[tuple[int | None, int | None, int | None]] = set()
    parse_failures = 0
    valid_moves = 0
    repaired_moves = 0

    try:
        for step in range(1, max_steps + 1):
            if observation.done:
                break

            prompt = TrainingExample(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=build_user_prompt(step=step, observation=observation, history=history),
                response="",
            )
            raw_text = generate_text(
                model=agent.model,
                tokenizer=agent.tokenizer,
                device=agent.device,
                prompt_text=render_prompt(agent.tokenizer, prompt),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            action = parse_action(raw_text)
            source = "model"
            if action is None:
                parse_failures += 1
                if fallback_policy == "candidate":
                    action = fallback_action(observation)
                    source = "parse_fallback_candidate"
                else:
                    action = invalid_parse_action(observation)
                    source = "parse_fallback_invalid"

            if repeat_guard and source == "model" and action_key(action) in invalid_actions:
                action = fallback_action(observation)
                source = "repeat_guard_candidate"
                repaired_moves += 1

            observation = env.step(action)
            if observation.move_valid:
                valid_moves += 1
            else:
                invalid_actions.add(action_key(action))

            action_text = render_action_json(action)
            history.append(
                f"step={step} source={source} action={action_text} "
                f"reward={observation.reward:+.2f} status={observation.status}"
            )
            frames.append(
                frame_from_observation(
                    step=step,
                    board=observation.board,
                    status=observation.status,
                    score=observation.score,
                    action=action_text,
                    raw_output=raw_text,
                    source=source,
                    reward=float(observation.reward or 0.0),
                    move_valid=observation.move_valid,
                    message=observation.message,
                )
            )

            if observation.done:
                break
    finally:
        unload_agent(agent)

    total_steps = max(observation.moves, 1)
    return {
        "label": label,
        "model_name": model_name,
        "success": observation.status == "solved",
        "score": observation.score,
        "steps": observation.moves,
        "mistakes": observation.mistakes,
        "valid_move_rate": valid_moves / total_steps,
        "parse_failures": parse_failures,
        "repaired_moves": repaired_moves,
        "frames": frames,
    }


def board_html(board: list[list[int]]) -> str:
    cells = []
    for row_index, row in enumerate(board):
        for column_index, value in enumerate(row):
            classes = ["cell"]
            if row_index in {2, 5}:
                classes.append("box-bottom")
            if column_index in {2, 5}:
                classes.append("box-right")
            text = "" if value == 0 else str(value)
            cells.append(f"<div class='{' '.join(classes)}'>{html.escape(text)}</div>")
    return "<div class='board'>" + "".join(cells) + "</div>"


def render_agent_panel(agent: dict[str, Any], frame_index: int) -> str:
    frames = agent["frames"]
    frame = frames[min(frame_index, len(frames) - 1)]
    valid_class = "valid" if frame["move_valid"] else "invalid"
    if frame["move_valid"] is None:
        valid_class = "neutral"
    return f"""
    <section class="agent-card">
      <header>
        <h3>{html.escape(agent["label"])}</h3>
        <p>{html.escape(agent["model_name"])}</p>
      </header>
      {board_html(frame["board"])}
      <div class="metrics">
        <span>step {frame["step"]}</span>
        <span>score {frame["score"]}</span>
        <span class="{valid_class}">{html.escape(str(frame["status"]))}</span>
      </div>
      <div class="move">
        <strong>action</strong><code>{html.escape(frame["action"] or "reset")}</code>
        <strong>source</strong><code>{html.escape(frame["source"])}</code>
        <strong>reward</strong><code>{frame["reward"]:+.2f}</code>
      </div>
      <details>
        <summary>Raw output and message</summary>
        <pre>{html.escape(frame["raw_output"] or "<none>")}</pre>
        <p>{html.escape(frame["message"])}</p>
      </details>
    </section>
    """


def render_playback(baseline: dict[str, Any], trained: dict[str, Any]) -> str:
    max_frames = max(len(baseline["frames"]), len(trained["frames"]))
    payload = json.dumps({"baseline": baseline, "trained": trained})
    player_id = f"player-{uuid.uuid4().hex}"
    initial_panels = render_agent_panel(baseline, 0) + render_agent_panel(trained, 0)
    summary = f"""
    <div class="summary">
      <div><strong>Baseline:</strong> success={baseline["success"]}, score={baseline["score"]}, mistakes={baseline["mistakes"]}, valid_rate={baseline["valid_move_rate"]:.2f}</div>
      <div><strong>Trained:</strong> success={trained["success"]}, score={trained["score"]}, mistakes={trained["mistakes"]}, valid_rate={trained["valid_move_rate"]:.2f}</div>
    </div>
    """
    return f"""
    <style>
      #{player_id} {{ font-family: Inter, system-ui, sans-serif; color: #f4f4f5; }}
      #{player_id} .summary {{ display: grid; gap: 6px; margin: 8px 0 14px; color: #d4d4d8; }}
      #{player_id} .controls {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 10px 0 16px; }}
      #{player_id} button {{ background: #8b5cf6; color: white; border: 0; border-radius: 6px; padding: 8px 12px; cursor: pointer; }}
      #{player_id} input[type=range] {{ min-width: 280px; flex: 1; }}
      #{player_id} .panels {{ display: grid; grid-template-columns: repeat(2, minmax(320px, 1fr)); gap: 16px; }}
      #{player_id} .agent-card {{ border: 1px solid #3f3f46; border-radius: 8px; padding: 14px; background: #18181b; }}
      #{player_id} h3 {{ margin: 0; font-size: 20px; }}
      #{player_id} header p {{ margin: 3px 0 12px; color: #a1a1aa; font-size: 12px; word-break: break-all; }}
      #{player_id} .board {{ display: grid; grid-template-columns: repeat(9, 34px); grid-auto-rows: 34px; width: max-content; border: 2px solid #e4e4e7; background: #27272a; }}
      #{player_id} .cell {{ display: grid; place-items: center; border-right: 1px solid #71717a; border-bottom: 1px solid #71717a; font-size: 18px; font-weight: 700; color: #f8fafc; }}
      #{player_id} .box-right {{ border-right: 2px solid #e4e4e7; }}
      #{player_id} .box-bottom {{ border-bottom: 2px solid #e4e4e7; }}
      #{player_id} .metrics {{ display: flex; gap: 8px; margin: 12px 0; flex-wrap: wrap; }}
      #{player_id} .metrics span {{ background: #27272a; border-radius: 6px; padding: 5px 8px; }}
      #{player_id} .valid {{ color: #86efac; }}
      #{player_id} .invalid {{ color: #fca5a5; }}
      #{player_id} .neutral {{ color: #bfdbfe; }}
      #{player_id} .move {{ display: grid; grid-template-columns: auto 1fr; gap: 6px 10px; align-items: center; }}
      #{player_id} code {{ background: #09090b; border-radius: 4px; padding: 4px 6px; color: #e5e7eb; white-space: pre-wrap; }}
      #{player_id} details {{ margin-top: 12px; color: #d4d4d8; }}
      #{player_id} pre {{ white-space: pre-wrap; background: #09090b; border-radius: 6px; padding: 8px; }}
      @media (max-width: 760px) {{ #{player_id} .panels {{ grid-template-columns: 1fr; }} }}
    </style>
    <div id="{player_id}">
      {summary}
      <div class="controls">
        <button data-action="prev">Prev</button>
        <button data-action="play">Play</button>
        <button data-action="pause">Pause</button>
        <button data-action="next">Next</button>
        <input type="range" min="0" max="{max_frames - 1}" value="0" />
        <span class="counter">0 / {max_frames - 1}</span>
      </div>
      <div class="panels">{initial_panels}</div>
    </div>
    <script type="application/json" id="{player_id}-data">{html.escape(payload)}</script>
    <script>
      (() => {{
        const root = document.getElementById("{player_id}");
        const dataEl = document.getElementById("{player_id}-data");
        if (!root || !dataEl) return;
        const data = JSON.parse(dataEl.textContent);
        const panels = root.querySelector(".panels");
        const slider = root.querySelector("input[type=range]");
        const counter = root.querySelector(".counter");
        let idx = 0;
        let timer = null;
        const renderBoard = (board) => {{
          let cells = "";
          for (let r = 0; r < 9; r++) {{
            for (let c = 0; c < 9; c++) {{
              const cls = ["cell"];
              if (r === 2 || r === 5) cls.push("box-bottom");
              if (c === 2 || c === 5) cls.push("box-right");
              cells += `<div class="${{cls.join(" ")}}">${{board[r][c] || ""}}</div>`;
            }}
          }}
          return `<div class="board">${{cells}}</div>`;
        }};
        const esc = (value) => String(value ?? "").replace(/[&<>"']/g, ch => ({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}}[ch]));
        const panel = (agent, frameIndex) => {{
          const frame = agent.frames[Math.min(frameIndex, agent.frames.length - 1)];
          const validClass = frame.move_valid === null ? "neutral" : frame.move_valid ? "valid" : "invalid";
          return `<section class="agent-card">
            <header><h3>${{esc(agent.label)}}</h3><p>${{esc(agent.model_name)}}</p></header>
            ${{renderBoard(frame.board)}}
            <div class="metrics"><span>step ${{frame.step}}</span><span>score ${{frame.score}}</span><span class="${{validClass}}">${{esc(frame.status)}}</span></div>
            <div class="move"><strong>action</strong><code>${{esc(frame.action || "reset")}}</code><strong>source</strong><code>${{esc(frame.source)}}</code><strong>reward</strong><code>${{Number(frame.reward).toFixed(2)}}</code></div>
            <details><summary>Raw output and message</summary><pre>${{esc(frame.raw_output || "<none>")}}</pre><p>${{esc(frame.message)}}</p></details>
          </section>`;
        }};
        const render = () => {{
          panels.innerHTML = panel(data.baseline, idx) + panel(data.trained, idx);
          slider.value = idx;
          counter.textContent = `${{idx}} / {max_frames - 1}`;
        }};
        const stop = () => {{ if (timer) clearInterval(timer); timer = null; }};
        root.querySelector('[data-action="prev"]').onclick = () => {{ idx = Math.max(0, idx - 1); render(); }};
        root.querySelector('[data-action="next"]').onclick = () => {{ idx = Math.min({max_frames - 1}, idx + 1); render(); }};
        root.querySelector('[data-action="pause"]').onclick = stop;
        root.querySelector('[data-action="play"]').onclick = () => {{
          stop();
          timer = setInterval(() => {{
            idx = Math.min({max_frames - 1}, idx + 1);
            render();
            if (idx >= {max_frames - 1}) stop();
          }}, 900);
        }};
        slider.oninput = () => {{ idx = Number(slider.value); render(); }};
      }})();
    </script>
    """


def generate_playback(
    baseline_model: str,
    trained_model: str,
    empty_boxes: int,
    seed: int,
    max_steps: int,
    device_choice: str,
    repeat_guard: bool,
) -> str:
    import torch

    random.seed(seed)
    device = choose_device(device_choice)
    steps = max_steps if max_steps > 0 else max(empty_boxes + 10, 40)
    baseline = rollout_agent(
        model_name=baseline_model,
        label="Baseline",
        seed=seed,
        empty_boxes=empty_boxes,
        max_steps=steps,
        max_new_tokens=96,
        temperature=0.0,
        top_p=0.95,
        device=device,
        repeat_guard=False,
        fallback_policy="invalid",
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    trained = rollout_agent(
        model_name=trained_model,
        label="Trained",
        seed=seed,
        empty_boxes=empty_boxes,
        max_steps=steps,
        max_new_tokens=96,
        temperature=0.0,
        top_p=0.95,
        device=device,
        repeat_guard=repeat_guard,
        fallback_policy="invalid",
    )
    return render_playback(baseline, trained)


def build_app() -> Any:
    import gradio as gr

    with gr.Blocks(title="Sudoku Agent Playback") as demo:
        gr.Markdown("# Sudoku Agent Playback")
        with gr.Row():
            baseline_model = gr.Textbox(value="Qwen/Qwen3-0.6B", label="Baseline model")
            trained_model = gr.Textbox(
                value="outputs/checkpoints/sudoku-20-to-35-clean",
                label="Trained model",
            )
        with gr.Row():
            empty_boxes = gr.Number(value=35, label="Empty cells", precision=0)
            seed = gr.Number(value=1001, label="Seed", precision=0)
            max_steps = gr.Number(value=45, label="Max steps", precision=0)
            device_choice = gr.Dropdown(["auto", "cuda", "cpu", "mps"], value="cuda", label="Device")
            repeat_guard = gr.Checkbox(value=False, label="Repeat guard for trained")
        run = gr.Button("Generate Playback", variant="primary")
        output = gr.HTML()

        run.click(
            fn=generate_playback,
            inputs=[
                baseline_model,
                trained_model,
                empty_boxes,
                seed,
                max_steps,
                device_choice,
                repeat_guard,
            ],
            outputs=output,
        )
    return demo


def main() -> None:
    args = parse_args()
    app = build_app()
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
