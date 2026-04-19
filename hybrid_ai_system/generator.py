"""Context-aware answer generation using a seq2seq transformer (e.g. Flan-T5)."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from .config import GeneratorConfig
from .document_processor import Chunk

logger = logging.getLogger(__name__)


class AnswerGenerator:
    """
    Generates a natural-language answer to a question given retrieved context.

    The default model is ``google/flan-t5-base``, an instruction-tuned
    encoder-decoder that produces concise, factual answers without requiring
    a GPU.  Any seq2seq model available on Hugging Face can be substituted
    by changing ``GeneratorConfig.model_name``.

    Prompt format
    -------------
    The context chunks are concatenated in ranked order and wrapped in a
    standard "Answer the question based on the context" template so that
    instruction-tuned models understand the task out of the box.
    """

    PROMPT_TEMPLATE = (
        "Answer the following question based only on the provided context. "
        "If the context does not contain enough information, say 'I don't know'.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )

    def __init__(self, config: Optional[GeneratorConfig] = None) -> None:
        self.config = config or GeneratorConfig()
        self._tokenizer = None
        self._model = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        question: str,
        retrieved_chunks: List[Tuple[Chunk, float]],
        max_context_chars: int = 2000,
    ) -> str:
        """
        Generate an answer for *question* given *retrieved_chunks*.

        Parameters
        ----------
        question:
            The user's natural language question.
        retrieved_chunks:
            Ordered list of (Chunk, score) from the retriever.
        max_context_chars:
            Truncate combined context to this many characters to stay within
            the model's sequence length limit.

        Returns
        -------
        The generated answer string.
        """
        context = self._build_context(retrieved_chunks, max_context_chars)
        prompt = self.PROMPT_TEMPLATE.format(context=context, question=question)
        return self._run_model(prompt)

    def build_prompt(
        self,
        question: str,
        retrieved_chunks: List[Tuple[Chunk, float]],
        max_context_chars: int = 2000,
    ) -> str:
        """Return the prompt string (useful for debugging or external LLMs)."""
        context = self._build_context(retrieved_chunks, max_context_chars)
        return self.PROMPT_TEMPLATE.format(context=context, question=question)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_context(
        self,
        chunks: List[Tuple[Chunk, float]],
        max_chars: int,
    ) -> str:
        parts = []
        total = 0
        for chunk, _ in chunks:
            text = chunk.text.strip()
            if total + len(text) > max_chars:
                remaining = max_chars - total
                if remaining > 50:
                    parts.append(text[:remaining])
                break
            parts.append(text)
            total += len(text)
        return "\n\n---\n\n".join(parts)

    def _run_model(self, prompt: str) -> str:
        tokenizer, model = self._get_model()
        cfg = self.config

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        )
        # Move to device
        inputs = {k: v.to(cfg.device) for k, v in inputs.items()}

        output_ids = model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            num_beams=cfg.num_beams,
            temperature=cfg.temperature if cfg.temperature > 0 else 1.0,
            do_sample=cfg.temperature > 0,
            early_stopping=True,
        )
        return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    def _get_model(self):
        if self._model is None:
            try:
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: PLC0415
            except ImportError as exc:
                raise ImportError(
                    "Install transformers: pip install transformers"
                ) from exc

            import torch  # noqa: PLC0415

            model_name = self.config.model_name
            device = self.config.device

            logger.info("Loading generator model '%s' on device '%s' …", model_name, device)
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self._model.to(device)
            self._model.eval()
            logger.info("Generator model loaded")

        return self._tokenizer, self._model
