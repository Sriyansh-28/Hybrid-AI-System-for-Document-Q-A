"""Supervised LoRA fine-tuning loop.

Uses TRL's SFTTrainer over the formatted instruction prompts. Only the LoRA
adapter weights are trained, which is where the compute saving versus a full
fine-tune comes from.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .prompt import build_example


def _load_split(path: str) -> Any:
    from datasets import load_dataset

    return load_dataset("json", data_files=path, split="train")


def run_training(cfg: PipelineConfig) -> str:
    """Fine-tune and save the adapter. Returns the output directory."""
    from transformers import TrainingArguments
    from trl import SFTTrainer

    from .model import attach_lora, load_base_model, load_tokenizer

    tokenizer = load_tokenizer(cfg.model)
    model = attach_lora(load_base_model(cfg.model), cfg.model)

    train_path = str(Path(cfg.data.processed_dir) / "train.jsonl")
    dataset = _load_split(train_path)

    def formatting_func(batch: dict) -> list[str]:
        records = [
            {
                "instruction": batch["instruction"][i],
                "input": batch.get("input", [""] * len(batch["instruction"]))[i],
                "output": batch["output"][i],
            }
            for i in range(len(batch["instruction"]))
        ]
        return [build_example(r) for r in records]

    args = TrainingArguments(
        output_dir=cfg.train.output_dir,
        num_train_epochs=cfg.train.epochs,
        per_device_train_batch_size=cfg.train.per_device_batch_size,
        gradient_accumulation_steps=cfg.train.grad_accum_steps,
        learning_rate=cfg.train.learning_rate,
        warmup_ratio=cfg.train.warmup_ratio,
        logging_steps=cfg.train.logging_steps,
        save_steps=cfg.train.save_steps,
        bf16=cfg.train.bf16,
        seed=cfg.train.seed,
        report_to=["none"],
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        formatting_func=formatting_func,
        max_seq_length=cfg.model.max_seq_length,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(cfg.train.output_dir)
    tokenizer.save_pretrained(cfg.train.output_dir)
    return cfg.train.output_dir
