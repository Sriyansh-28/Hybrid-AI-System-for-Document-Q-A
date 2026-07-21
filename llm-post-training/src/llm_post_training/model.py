"""Model + tokenizer loading and LoRA wiring.

Heavy imports (torch, transformers, peft) live inside the functions so that the
data and prompt utilities can be imported and tested on a machine without a GPU
stack installed.
"""

from __future__ import annotations

from typing import Any

from .config import ModelConfig


def load_tokenizer(cfg: ModelConfig) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=True)
    # Decoder-only models often ship without a pad token; reuse EOS so that
    # batching and loss masking behave.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model(cfg: ModelConfig) -> Any:
    import torch
    from transformers import AutoModelForCausalLM

    quant_config = None
    if cfg.load_in_4bit:
        from transformers import BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = False
    return model


def attach_lora(model: Any, cfg: ModelConfig) -> Any:
    """Wrap a base model with a fresh LoRA adapter (QLoRA-ready)."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if cfg.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model
