"""Merge the LoRA adapter into the base weights for serving.

Adapter-swapping at serve time is great when you host many domains off one base
model. But for a single-domain deployment it's simpler and faster to bake the
adapter into a standalone set of weights, which is what this does.
"""

from __future__ import annotations

from pathlib import Path

from .config import ModelConfig


def merge_and_export(cfg: ModelConfig, adapter_path: str, output_path: str) -> str:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    from .model import load_tokenizer

    base = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    merged = model.merge_and_unload()

    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out)
    load_tokenizer(cfg).save_pretrained(out)
    return str(out)
