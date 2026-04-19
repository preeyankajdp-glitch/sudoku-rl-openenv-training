"""Supervised fine-tuning entry point for the Sudoku OpenEnv task.

This script intentionally uses plain PyTorch and Hugging Face Transformers.
It creates oracle trajectories from the local OpenEnv environment and trains a
causal language model to emit the next Sudoku action as JSON.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

from sudoku_rl.training_common import TrainingExample, iter_oracle_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a Transformers model on Sudoku action traces.")
    parser.add_argument("--model-name", default="Qwen/Qwen3-0.6B", help="Base model or checkpoint path.")
    parser.add_argument("--output-dir", default="outputs/checkpoints/sudoku-sft", help="Where to save the model.")
    parser.add_argument("--episodes", type=int, default=64, help="Oracle episodes used for training data.")
    parser.add_argument("--empty-boxes", type=int, default=35, help="Number of blanks per Sudoku puzzle.")
    parser.add_argument("--seed", type=int, default=42, help="Dataset and training seed.")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=1, help="Micro-batch size.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8, help="Optimizer accumulation steps.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="AdamW learning rate.")
    parser.add_argument("--max-length", type=int, default=2048, help="Maximum tokenized sequence length.")
    parser.add_argument("--max-examples", type=int, default=0, help="Optional cap for quick smoke runs.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Training device.")
    return parser.parse_args()


def choose_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def render_prompt(tokenizer: Any, example: TrainingExample) -> str:
    messages = [
        {"role": "system", "content": example.system_prompt},
        {"role": "user", "content": example.user_prompt},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return (
        f"System:\n{example.system_prompt}\n\n"
        f"User:\n{example.user_prompt}\n\n"
        "Assistant:\n"
    )


class PromptResponseDataset:
    def __init__(self, examples: list[TrainingExample], tokenizer: Any, max_length: int) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        example = self.examples[index]
        prompt_text = render_prompt(self.tokenizer, example)
        eos_text = self.tokenizer.eos_token or ""
        full_text = prompt_text + example.response + eos_text

        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False).input_ids
        tokenized = self.tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask
        labels = input_ids[:]
        prompt_length = min(len(prompt_ids), len(labels))
        labels[:prompt_length] = [-100] * prompt_length
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class CausalLmCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        max_length = max(len(feature["input_ids"]) for feature in features)
        batch: dict[str, list[list[int]]] = {"input_ids": [], "attention_mask": [], "labels": []}
        pad_id = self.tokenizer.pad_token_id

        for feature in features:
            padding = max_length - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [pad_id] * padding)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * padding)
            batch["labels"].append(feature["labels"] + [-100] * padding)

        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def main() -> None:
    args = parse_args()

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)

    examples = list(
        iter_oracle_examples(
            episodes=args.episodes,
            empty_boxes=args.empty_boxes,
            seed=args.seed,
        )
    )
    if args.max_examples > 0:
        examples = examples[: args.max_examples]
    if not examples:
        raise RuntimeError("No training examples were generated.")

    print(f"[DATA] examples={len(examples)} empty_boxes={args.empty_boxes}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.eos_token is None:
        tokenizer.eos_token = tokenizer.pad_token or ""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)
    model.to(device)
    model.train()

    dataset = PromptResponseDataset(examples, tokenizer=tokenizer, max_length=args.max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=CausalLmCollator(tokenizer),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        for batch_index, batch in enumerate(dataloader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()
            running_loss += float(outputs.loss.detach().cpu())

            should_step = batch_index % args.gradient_accumulation_steps == 0
            is_last = batch_index == len(dataloader)
            if should_step or is_last:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            if batch_index % 10 == 0 or is_last:
                avg_loss = running_loss / batch_index
                print(
                    f"[TRAIN] epoch={epoch} batch={batch_index}/{len(dataloader)} "
                    f"optimizer_step={global_step} loss={avg_loss:.4f}",
                    flush=True,
                )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[SAVED] {output_dir}", flush=True)


if __name__ == "__main__":
    main()
