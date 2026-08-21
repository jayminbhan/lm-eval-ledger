# tasks/aime.py
"""AIME benchmark tasks - American Invitational Mathematics Examination."""
from __future__ import annotations

from .base import (
    TaskConfig,
    extract_boxed_strict,
    numeric_match,
)


def build_prompt(example: dict, fewshot_block: str) -> str:
    """Build prompt for AIME problems."""
    # Handle different field name casing (Problem vs problem)
    problem = example.get("problem", example.get("Problem", "")).strip()
    if fewshot_block:
        return f"{fewshot_block}\n\nProblem:\n{problem}\n\nSolution:"
    return f"Problem:\n{problem}\n\nSolution:"


def extract_gold(example: dict) -> str:
    """Extract gold answer from AIME example (integers 0-999)."""
    # Handle different field name casing (Answer vs answer)
    answer = example.get("answer", example.get("Answer", ""))
    return str(answer).strip()


def get_task_2024() -> TaskConfig:
    """Get AIME 2024 task configuration."""
    return TaskConfig(
        name="aime_2024",
        build_prompt=build_prompt,
        extract_gold=extract_gold,
        extract_pred=extract_boxed_strict,
        match_fn=numeric_match,
        stop_strings=["Problem:"],
        default_fewshot_k=0,
        fewshot_answer_field="Answer",
        description="AIME 2024 - competition math problems",
        hf_repo="HuggingFaceH4/aime_2024",
        hf_split="train",
    )


def get_task_2025() -> TaskConfig:
    """Get AIME 2025 task configuration."""
    return TaskConfig(
        name="aime_2025",
        build_prompt=build_prompt,
        extract_gold=extract_gold,
        extract_pred=extract_boxed_strict,
        match_fn=numeric_match,
        stop_strings=["Problem:"],
        default_fewshot_k=0,
        fewshot_answer_field="answer",
        description="AIME 2025 - competition math problems",
        hf_repo="MathArena/aime_2025",
        hf_split="train",
    )
