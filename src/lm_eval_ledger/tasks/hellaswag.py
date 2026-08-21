# tasks/hellaswag.py
"""HellaSwag benchmark task - Commonsense sentence completion MCQ."""
from __future__ import annotations

from .base import TaskConfig, exact_match, extract_boxed_letter

CHOICE_LABELS = ["A", "B", "C", "D"]


def build_prompt(example: dict, fewshot_block: str) -> str:
    """Build prompt for HellaSwag with multiple choice format."""
    ctx = example.get("ctx", "").strip()
    activity = example.get("activity_label", "").strip()
    endings = example.get("endings", [])

    choice_text = "\n".join(
        f"{CHOICE_LABELS[i]}. {ending}"
        for i, ending in enumerate(endings)
        if i < len(CHOICE_LABELS)
    )

    prompt = f"Activity: {activity}\nContext: {ctx}\n\nHow does this end?\n{choice_text}\nAnswer:"

    if fewshot_block:
        return f"{fewshot_block}\n\n{prompt}"
    return prompt


def extract_gold(example: dict) -> str:
    """Extract gold answer from HellaSwag example (label "0"-"3" -> A-D)."""
    label = example.get("label", "0")
    try:
        idx = int(label)
    except (ValueError, TypeError):
        return str(label)
    if 0 <= idx < len(CHOICE_LABELS):
        return CHOICE_LABELS[idx]
    return str(label)


def extract_pred(model_output: str) -> str:
    """Extract predicted choice letter from \\boxed{...} in model output."""
    return extract_boxed_letter(model_output, CHOICE_LABELS)


def get_choice_texts(example: dict) -> list[str]:
    """Get list of choice texts for completion log-likelihood scoring."""
    return example.get("endings", [])


def get_task_generate() -> TaskConfig:
    """Get HellaSwag task (generative)."""
    return TaskConfig(
        name="hellaswag_generate",
        build_prompt=build_prompt,
        extract_gold=extract_gold,
        extract_pred=extract_pred,
        match_fn=exact_match,
        stop_strings=["Activity:"],
        default_fewshot_k=0,
        fewshot_answer_field="label",
        description="HellaSwag - commonsense sentence completion (generative)",
        hf_repo="Rowan/hellaswag",
        hf_split="validation",
        hf_fewshot_split="train",
    )


def get_task_logprob_token() -> TaskConfig:
    """Get HellaSwag task (first-token log-probability)."""
    return TaskConfig(
        name="hellaswag_logprob_token",
        build_prompt=build_prompt,
        extract_gold=extract_gold,
        extract_pred=extract_pred,
        match_fn=exact_match,
        stop_strings=[],
        default_fewshot_k=0,
        fewshot_answer_field="label",
        description="HellaSwag - commonsense sentence completion (first-token log-probability)",
        eval_mode="logprob_token",
        choice_labels=CHOICE_LABELS,
        hf_repo="Rowan/hellaswag",
        hf_split="validation",
        hf_fewshot_split="train",
    )


def get_task_logprob_seq() -> TaskConfig:
    """Get HellaSwag task (completion log-likelihood)."""
    return TaskConfig(
        name="hellaswag_logprob_seq",
        build_prompt=build_prompt,
        extract_gold=extract_gold,
        extract_pred=extract_pred,
        match_fn=exact_match,
        stop_strings=[],
        default_fewshot_k=0,
        fewshot_answer_field="label",
        description="HellaSwag - commonsense sentence completion (completion log-likelihood)",
        eval_mode="logprob_seq",
        choice_labels=CHOICE_LABELS,
        get_choice_texts=get_choice_texts,
        hf_repo="Rowan/hellaswag",
        hf_split="validation",
        hf_fewshot_split="train",
    )
