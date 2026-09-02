# tasks/olympiad_bench.py
"""OlympiadBench benchmark tasks - Olympiad-level math and physics problems."""
from __future__ import annotations

from .base import (
    TaskConfig,
    extract_boxed_strict,
    normalized_match,
)


def build_prompt(example: dict, fewshot_block: str) -> str:
    """Build prompt for OlympiadBench problems."""
    question = example.get("question", "").strip()

    # Add context if available
    context = example.get("context", "")
    if context:
        question = f"{context}\n\n{question}"

    if fewshot_block:
        return f"{fewshot_block}\n\nProblem:\n{question}\n\nSolution:"
    return f"Problem:\n{question}\n\nSolution:"


def extract_gold(example: dict) -> str:
    """Extract gold answer from OlympiadBench example (final_answer is a list)."""
    final_answer = example.get("final_answer", [])
    if isinstance(final_answer, list) and final_answer:
        return str(final_answer[0]).strip()
    return str(final_answer).strip()


def match_answer(gold: str, pred: str) -> bool:
    """Match answers with flexibility for OlympiadBench."""
    # Symbolic equivalence covers normalized string equality and more
    from .base import symbolic_match
    if symbolic_match(gold, pred):
        return True

    # Try numeric comparison
    try:
        g = float(gold.replace(",", ""))
        p = float(pred.replace(",", ""))
        if abs(g - p) < 1e-6:
            return True
        # Allow relative tolerance for large numbers
        if g != 0 and abs((g - p) / g) < 1e-4:
            return True
    except (ValueError, TypeError):
        pass

    return False


def get_task_math_en() -> TaskConfig:
    """Get OlympiadBench Math (English) task - OE_TO_maths_en_COMP only."""
    return TaskConfig(
        name="olympiad_bench_math_en",
        build_prompt=build_prompt,
        extract_gold=extract_gold,
        extract_pred=extract_boxed_strict,
        match_fn=match_answer,
        stop_strings=["Problem:"],
        default_fewshot_k=0,
        fewshot_solution_field="solution",
        fewshot_final_answer_field="final_answer",
        description="OlympiadBench Math (English) - OE_TO_maths_en_COMP",
        hf_repo="Hothan/OlympiadBench",
        hf_config="OE_TO_maths_en_COMP",
        hf_split="train",
    )


def get_task_physics_en() -> TaskConfig:
    """Get OlympiadBench Physics (English) task - OE_TO_physics_en_COMP only."""
    return TaskConfig(
        name="olympiad_bench_physics_en",
        build_prompt=build_prompt,
        extract_gold=extract_gold,
        extract_pred=extract_boxed_strict,
        match_fn=match_answer,
        stop_strings=["Problem:"],
        default_fewshot_k=0,
        fewshot_solution_field="solution",
        fewshot_final_answer_field="final_answer",
        description="OlympiadBench Physics (English) - OE_TO_physics_en_COMP",
        hf_repo="Hothan/OlympiadBench",
        hf_config="OE_TO_physics_en_COMP",
        hf_split="train",
    )
