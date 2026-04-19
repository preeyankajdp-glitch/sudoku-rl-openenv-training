"""Evaluate a language model policy on the local Sudoku OpenEnv environment."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean
from typing import Any

from sudoku_rl.server.sudoku_rl_environment import SudokuRlEnvironment
from sudoku_rl.training_common import (
    build_user_prompt,
    fallback_action,
    parse_action,
    render_action_json,
)
from sudoku_rl.train_transformers import choose_device, render_prompt
from sudoku_rl.training_common import TrainingExample, SYSTEM_PROMPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Transformers policy on Sudoku RL.")
    parser.add_argument("--model-name", default="Qwen/Qwen3-0.6B", help="Model or checkpoint path.")
    parser.add_argument("--episodes", type=int, default=20, help="Evaluation episodes.")
    parser.add_argument("--empty-boxes", type=int, default=35, help="Number of blanks per puzzle.")
    parser.add_argument("--seed", type=int, default=1000, help="Evaluation seed.")
    parser.add_argument("--max-steps", type=int, default=0, help="Step limit. Default is empty_boxes * 3.")
    parser.add_argument("--max-new-tokens", type=int, default=96, help="Generation token budget.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Nucleus sampling value.")
    parser.add_argument("--output-json", default="", help="Optional path to write metrics JSON.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Inference device.")
    parser.add_argument("--verbose", action="store_true", help="Print board, raw model output, parsed action, and reward each step.")
    return parser.parse_args()


def generate_text(
    *,
    model: Any,
    tokenizer: Any,
    device: str,
    prompt_text: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    import torch

    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    generation_kwargs: dict[str, Any] = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if temperature > 0:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    with torch.no_grad():
        output_ids = model.generate(**generation_kwargs)

    generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def run_episode(
    *,
    model: Any,
    tokenizer: Any,
    device: str,
    seed: int,
    empty_boxes: int,
    max_steps: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    verbose: bool,
) -> dict[str, Any]:
    env = SudokuRlEnvironment()
    observation = env.reset(seed=seed, empty_boxes=empty_boxes)
    history: list[str] = []
    parse_failures = 0
    valid_moves = 0
    model_moves = 0
    fallback_moves = 0
    final_status = observation.status
    raw_score = observation.score

    for step in range(1, max_steps + 1):
        if observation.done:
            break

        example = TrainingExample(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=build_user_prompt(step=step, observation=observation, history=history),
            response="",
        )
        prompt_text = render_prompt(tokenizer, example)
        raw_text = generate_text(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt_text=prompt_text,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        action = parse_action(raw_text)
        source = "model"
        if action is None:
            parse_failures += 1
            action = fallback_action(observation)
            source = "parse_fallback"

        previous_board = observation.board_text

        observation = env.step(action)
        if observation.move_valid:
            valid_moves += 1
        if source == "model":
            model_moves += 1
        else:
            fallback_moves += 1

        history.append(
            f"step={step} source={source} action={render_action_json(action)} "
            f"reward={observation.reward:+.2f} status={observation.status}"
        )

        if verbose:
            print(f"\n[STEP DETAIL] seed={seed} step={step}", flush=True)
            print("[OBSERVATION BOARD]", flush=True)
            print(previous_board, flush=True)
            print("[MODEL RAW OUTPUT]", flush=True)
            print(raw_text or "<empty>", flush=True)
            print("[ACTION USED]", flush=True)
            print(f"source={source} action={render_action_json(action)}", flush=True)
            print("[ENV RESULT]", flush=True)
            print(
                f"reward={observation.reward:+.2f} status={observation.status} "
                f"move_valid={observation.move_valid} score={observation.score} "
                f"message={observation.message}",
                flush=True,
            )
        final_status = observation.status
        raw_score = observation.score

        if observation.done:
            break

    max_possible_score = max(empty_boxes * max(observation.score_step, 1), 1)
    normalized_score = max(min(raw_score / max_possible_score, 1.0), 0.0)
    return {
        "seed": seed,
        "success": final_status == "solved",
        "final_status": final_status,
        "raw_score": raw_score,
        "normalized_score": normalized_score,
        "steps": observation.moves,
        "mistakes": observation.mistakes,
        "valid_moves": valid_moves,
        "model_moves": model_moves,
        "fallback_moves": fallback_moves,
        "parse_failures": parse_failures,
    }


def summarize(model_name: str, empty_boxes: int, episodes: list[dict[str, Any]]) -> dict[str, Any]:
    total_steps = sum(episode["steps"] for episode in episodes)
    return {
        "model_name": model_name,
        "empty_boxes": empty_boxes,
        "episodes": len(episodes),
        "success_rate": mean(episode["success"] for episode in episodes),
        "avg_normalized_score": mean(episode["normalized_score"] for episode in episodes),
        "avg_raw_score": mean(episode["raw_score"] for episode in episodes),
        "avg_steps": mean(episode["steps"] for episode in episodes),
        "avg_mistakes": mean(episode["mistakes"] for episode in episodes),
        "valid_move_rate": (
            sum(episode["valid_moves"] for episode in episodes) / total_steps
            if total_steps
            else 0.0
        ),
        "parse_failure_rate": (
            sum(episode["parse_failures"] for episode in episodes) / total_steps
            if total_steps
            else 0.0
        ),
        "episode_results": episodes,
    }


def main() -> None:
    args = parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.eos_token is None:
        tokenizer.eos_token = tokenizer.pad_token or ""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)
    model.to(device)
    model.eval()

    max_steps = args.max_steps if args.max_steps > 0 else max(args.empty_boxes * 3, 40)
    episodes = []
    for episode_index in range(args.episodes):
        result = run_episode(
            model=model,
            tokenizer=tokenizer,
            device=device,
            seed=args.seed + episode_index,
            empty_boxes=args.empty_boxes,
            max_steps=max_steps,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            verbose=args.verbose,
        )
        episodes.append(result)
        print(
            f"[EVAL] episode={episode_index + 1}/{args.episodes} "
            f"success={result['success']} score={result['raw_score']} "
            f"steps={result['steps']} parse_failures={result['parse_failures']}",
            flush=True,
        )

    metrics = summarize(args.model_name, args.empty_boxes, episodes)
    print(json.dumps({key: value for key, value in metrics.items() if key != "episode_results"}, indent=2))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"[SAVED] {output_path}", flush=True)


if __name__ == "__main__":
    main()
