# tasks/gsm8k.py
"""GSM8K benchmark tasks - Grade School Math 8K (main and socratic versions)."""
from __future__ import annotations

from .base import (
    TaskConfig,
    extract_after_marker,
    extract_boxed_strict,
    extract_number,
    numeric_match,
)


def build_prompt(example: dict, fewshot_block: str) -> str:
    """Build prompt for GSM8K."""
    q = example["question"].strip()
    if fewshot_block:
        return f"{fewshot_block}\n\nQuestion: {q}\nAnswer:"
    return f"Question: {q}\nAnswer:"


def extract_gold(example: dict) -> str:
    """Extract gold answer from GSM8K example (number after ####)."""
    answer = example.get("answer", "")
    tail = extract_after_marker(answer, "####")
    return extract_number(tail)


def extract_pred(model_output: str) -> str:
    """Extract predicted answer from the first \\boxed{...} in model output.

    Strips commas and dollar signs for consistent numeric matching.
    Returns empty string if the format isn't followed.
    """
    raw = extract_boxed_strict(model_output)
    return raw.replace(",", "").replace("$", "").strip()


def get_task_main() -> TaskConfig:
    """Get GSM8K main task configuration."""
    return TaskConfig(
        name="gsm8k_main",
        build_prompt=build_prompt,
        extract_gold=extract_gold,
        extract_pred=extract_pred,
        match_fn=numeric_match,
        stop_strings=["Question:", "\n\nQuestion"],
        default_fewshot_k=8,
        fewshot_answer_field="answer",
        description="GSM8K Main - arithmetic word problems",
        hf_repo="openai/gsm8k",
        hf_config="main",
        hf_split="test",
        hf_fewshot_split="train",
    )


def get_task_socratic() -> TaskConfig:
    """Get GSM8K socratic task configuration."""
    return TaskConfig(
        name="gsm8k_socratic",
        build_prompt=build_prompt,
        extract_gold=extract_gold,
        extract_pred=extract_pred,
        match_fn=numeric_match,
        stop_strings=["Question:", "\n\nQuestion"],
        default_fewshot_k=8,
        fewshot_answer_field="answer",
        description="GSM8K Socratic - arithmetic word problems (socratic version)",
        hf_repo="openai/gsm8k",
        hf_config="socratic",
        hf_split="test",
        hf_fewshot_split="train",
        hf_fewshot_config="main",  # Use main's few-shot examples
    )
