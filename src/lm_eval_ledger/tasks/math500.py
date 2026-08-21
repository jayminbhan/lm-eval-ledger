# tasks/math500.py
"""MATH-500 benchmark task - subset of MATH dataset."""
from __future__ import annotations

from .base import (
    TaskConfig,
    extract_boxed_strict,
    normalized_match,
)


def build_prompt(example: dict, fewshot_block: str) -> str:
    """Build prompt for MATH-500 problems."""
    problem = example.get("problem", "").strip()
    if fewshot_block:
        return f"{fewshot_block}\n\nProblem: {problem}\nSolution:"
    return f"Problem: {problem}\nSolution:"


def extract_gold(example: dict) -> str:
    """Extract gold answer from MATH-500 example (may contain LaTeX)."""
    return str(example.get("answer", "")).strip()


def get_task() -> TaskConfig:
    """Get MATH-500 task configuration."""
    return TaskConfig(
        name="math500",
        build_prompt=build_prompt,
        extract_gold=extract_gold,
        extract_pred=extract_boxed_strict,
        match_fn=normalized_match,
        stop_strings=["Problem:"],
        default_fewshot_k=0,
        fewshot_answer_field="solution",
        description="MATH-500 - challenging competition math problems",
        hf_repo="HuggingFaceH4/MATH-500",
        hf_split="test",
    )
