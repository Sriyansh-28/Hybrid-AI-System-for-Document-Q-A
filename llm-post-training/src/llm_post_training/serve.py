"""Minimal FastAPI serving layer for the fine-tuned model.

A thin wrapper that loads the merged weights once at startup and exposes a
/generate endpoint. For higher throughput you'd swap this for vLLM or TGI, but
the request/response contract stays the same.
"""

from __future__ import annotations

import os
from typing import Any

from .prompt import build_prompt, extract_response

_model: Any = None
_tokenizer: Any = None


def _ensure_loaded() -> None:
    global _model, _tokenizer
    if _model is not None:
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = os.environ.get("MODEL_PATH", "outputs/merged")
    _tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    _model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    _model.eval()


def generate_answer(instruction: str, context: str | None = None, max_new_tokens: int = 256) -> str:
    import torch

    _ensure_loaded()
    prompt = build_prompt(instruction, context)
    inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)
    with torch.no_grad():
        out = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=_tokenizer.pad_token_id,
        )
    return extract_response(_tokenizer.decode(out[0], skip_special_tokens=True))


def build_app():
    """Construct the FastAPI app. Imported lazily so tests don't need fastapi."""
    from fastapi import FastAPI
    from pydantic import BaseModel

    app = FastAPI(title="Domain-Adapted LLM", version="0.1.0")

    class GenerateRequest(BaseModel):
        instruction: str
        context: str | None = None
        max_new_tokens: int = 256

    class GenerateResponse(BaseModel):
        answer: str

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/generate", response_model=GenerateResponse)
    def generate(req: GenerateRequest) -> GenerateResponse:
        answer = generate_answer(req.instruction, req.context, req.max_new_tokens)
        return GenerateResponse(answer=answer)

    return app
