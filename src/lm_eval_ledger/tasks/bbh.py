# tasks/bbh.py
"""BIG-Bench Hard (BBH) benchmark task - 27 challenging reasoning subtasks."""
from __future__ import annotations

from .base import TaskConfig, extract_boxed_strict, normalized_match

BBH_SUBTASKS = [
    "boolean_expressions", "causal_judgement", "date_understanding",
    "disambiguation_qa", "dyck_languages", "formal_fallacies",
    "geometric_shapes", "hyperbaton",
    "logical_deduction_five_objects", "logical_deduction_seven_objects",
    "logical_deduction_three_objects", "movie_recommendation",
    "multistep_arithmetic_two", "navigate", "object_counting",
    "penguins_in_a_table", "reasoning_about_colored_objects", "ruin_names",
    "salient_translation_error_detection", "snarks", "sports_understanding",
    "temporal_sequences", "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects", "web_of_lies", "word_sorting",
]


def build_prompt(example: dict, fewshot_block: str) -> str:
    """Build prompt for BBH problems.

    The 'input' field already contains the full question text,
    including any MCQ options for relevant subtasks.
    """
    question = example.get("input", "").strip()

    if fewshot_block:
        return f"{fewshot_block}\n\nQuestion:\n{question}\n\nAnswer:"
    return f"Question:\n{question}\n\nAnswer:"


def extract_gold(example: dict) -> str:
    """Extract gold answer from BBH example."""
    target = example.get("target", "").strip()
    # Strip parentheses from MCQ answers: "(B)" -> "B"
    if len(target) == 3 and target.startswith("(") and target.endswith(")"):
        return target[1]
    return target


def match_answer(gold: str, pred: str) -> bool:
    """Match answers with flexibility for BBH's mixed formats.

    Handles MCQ letters, True/False, numbers, and text answers.
    """
    # Strip parens for MCQ: "(B)" vs "B"
    g = gold.strip().strip("()")
    p = pred.strip().strip("()")

    # Case-insensitive exact match
    if g.lower() == p.lower():
        return True

    # Fall back to normalized match (handles LaTeX artifacts)
    return normalized_match(g, p)


def get_task() -> TaskConfig:
    """Get BBH task (all 27 subtasks aggregated)."""
    return TaskConfig(
        name="bbh",
        build_prompt=build_prompt,
        extract_gold=extract_gold,
        extract_pred=extract_boxed_strict,
        match_fn=match_answer,
        stop_strings=["Question:"],
        default_fewshot_k=0,
        fewshot_answer_field="target",
        description="BIG-Bench Hard - 27 challenging reasoning subtasks",
        hf_repo="lukaemon/bbh",
        hf_split="test",
        hf_configs=BBH_SUBTASKS,
        hf_config_field="subtask",
    )
