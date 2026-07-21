"""Typed config objects loaded from YAML.

Everything that changes between experiments lives in a YAML file, not in the
code. That way a teammate can point the same scripts at a new dataset or a new
set of hyperparameters and re-run without editing Python.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    raw_path: str = "data/raw/sample_raw.jsonl"
    processed_dir: str = "data/processed"
    eval_fraction: float = 0.05
    max_examples: int | None = None
    min_output_chars: int = 1
    seed: int = 42


@dataclass
class ModelConfig:
    base_model: str = "meta-llama/Llama-2-7b-hf"
    max_seq_length: int = 1024
    load_in_4bit: bool = True
    # LoRA hyperparameters.
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )


@dataclass
class TrainConfig:
    output_dir: str = "outputs/adapter"
    epochs: float = 3.0
    per_device_batch_size: int = 4
    grad_accum_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_steps: int = 200
    bf16: bool = True
    seed: int = 42


@dataclass
class EvalConfig:
    adapter_path: str = "outputs/adapter"
    eval_path: str = "data/processed/eval.jsonl"
    report_path: str = "outputs/eval_report.json"
    max_new_tokens: int = 256
    # Semantic-match threshold for scoring correctness against the reference.
    similarity_threshold: float = 0.75


@dataclass
class PipelineConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def _build(cls: Any, values: dict[str, Any]) -> Any:
    known = {f.name for f in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**values)


def load_config(path: str | Path) -> PipelineConfig:
    """Load a PipelineConfig from a YAML file, falling back to defaults."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return PipelineConfig(
        data=_build(DataConfig, raw.get("data", {})),
        model=_build(ModelConfig, raw.get("model", {})),
        train=_build(TrainConfig, raw.get("train", {})),
        eval=_build(EvalConfig, raw.get("eval", {})),
    )
