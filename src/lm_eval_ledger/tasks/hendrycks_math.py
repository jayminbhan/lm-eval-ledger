# tasks/hendrycks_math.py
"""Hendrycks MATH benchmark task - Competition math problems."""
from __future__ import annotations

from .base import (
    TaskConfig,
    extract_boxed,
    extract_boxed_strict,
    normalized_match,
)

MATH_SUBJECTS = [
    "algebra", "counting_and_probability", "geometry",
    "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
]


def build_prompt(example: dict, fewshot_block: str) -> str:
    """Build prompt for Hendrycks MATH problems."""
    problem = example.get("problem", "").strip()

    if fewshot_block:
        return f"{fewshot_block}\n\nProblem:\n{problem}\n\nSolution:"
    return f"Problem:\n{problem}\n\nSolution:"


def extract_gold(example: dict) -> str:
    """Extract gold answer from Hendrycks MATH example.

    The solution field contains the full step-by-step solution with \\boxed{answer}.
    """
    solution = example.get("solution", "")
    return extract_boxed(solution)


def get_task() -> TaskConfig:
    """Get Hendrycks MATH task (all 7 subjects aggregated)."""
    return TaskConfig(
        name="hendrycks_math",
        build_prompt=build_prompt,
        extract_gold=extract_gold,
        extract_pred=extract_boxed_strict,
        match_fn=normalized_match,
        stop_strings=["Problem:"],
        default_fewshot_k=0,
        fewshot_solution_field="solution",
        description="Hendrycks MATH - competition math (7 subjects aggregated)",
        hf_repo="EleutherAI/hendrycks_math",
        hf_split="test",
        hf_configs=MATH_SUBJECTS,
        hf_config_field="subject",
        hf_fewshot_split="train",
        hf_fewshot_config="algebra",
    )
