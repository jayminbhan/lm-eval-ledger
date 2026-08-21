# tasks/theoremqa.py
"""TheoremQA benchmark task - Theorem-based math/science questions."""
from __future__ import annotations

import math

from .base import TaskConfig, extract_boxed_strict, numeric_match


def build_prompt(example: dict, fewshot_block: str) -> str:
    """Build prompt for TheoremQA problems."""
    question = example.get("Question", "").strip()

    if fewshot_block:
        return f"{fewshot_block}\n\nProblem:\n{question}\n\nSolution:"
    return f"Problem:\n{question}\n\nSolution:"


def extract_gold(example: dict) -> str:
    """Extract gold answer from TheoremQA example."""
    return str(example.get("Answer", "")).strip()


def _hf_post_process(examples: list[dict]) -> list[dict]:
    """Filter out image-dependent questions from HF-loaded data."""
    text_only = []
    for ex in examples:
        picture = ex.get("Picture")
        if picture is None or (isinstance(picture, float) and math.isnan(picture)):
            text_only.append(ex)
    return text_only


def get_task() -> TaskConfig:
    """Get TheoremQA task (text-only, images filtered out)."""
    return TaskConfig(
        name="theoremqa",
        build_prompt=build_prompt,
        extract_gold=extract_gold,
        extract_pred=extract_boxed_strict,
        match_fn=numeric_match,
        stop_strings=["Problem:"],
        default_fewshot_k=0,
        fewshot_answer_field="Answer",
        description="TheoremQA - theorem-based math/science (text-only)",
        hf_repo="TIGER-Lab/TheoremQA",
        hf_split="test",
        hf_post_process=_hf_post_process,
    )
