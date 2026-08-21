# tasks/mmlu_pro.py
"""MMLU-Pro benchmark task - Harder MCQ with up to 10 options."""
from __future__ import annotations

from .base import TaskConfig, exact_match, extract_boxed_letter

CHOICE_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


def build_prompt(example: dict, fewshot_block: str) -> str:
    """Build prompt for MMLU-Pro with multiple choice format."""
    question = example.get("question", "").strip()
    options = example.get("options", [])

    choice_text = "\n".join(
        f"{CHOICE_LABELS[i]}. {option}"
        for i, option in enumerate(options)
        if i < len(CHOICE_LABELS)
    )

    prompt = f"Question: {question}\n{choice_text}\nAnswer:"

    if fewshot_block:
        return f"{fewshot_block}\n\n{prompt}"
    return prompt


def extract_gold(example: dict) -> str:
    """Extract gold answer from MMLU-Pro example (letter A-J)."""
    return str(example.get("answer", "")).strip().upper()


def extract_pred(model_output: str) -> str:
    """Extract predicted choice letter from \\boxed{...} in model output."""
    return extract_boxed_letter(model_output, CHOICE_LABELS)


def get_choice_texts(example: dict) -> list[str]:
    """Get list of choice texts for completion log-likelihood scoring."""
    return example.get("options", [])


def get_task_generate() -> TaskConfig:
    """Get MMLU-Pro task (generative)."""
    return TaskConfig(
        name="mmlu_pro_generate",
        build_prompt=build_prompt,
        extract_gold=extract_gold,
        extract_pred=extract_pred,
        match_fn=exact_match,
        stop_strings=["Question:"],
        default_fewshot_k=0,
        fewshot_answer_field="answer",
        description="MMLU-Pro - harder MCQ with up to 10 options (generative)",
        hf_repo="TIGER-Lab/MMLU-Pro",
        hf_split="test",
        hf_fewshot_split="validation",
    )


def get_task_logprob_token() -> TaskConfig:
    """Get MMLU-Pro task (first-token log-probability)."""
    return TaskConfig(
        name="mmlu_pro_logprob_token",
        build_prompt=build_prompt,
        extract_gold=extract_gold,
        extract_pred=extract_pred,
        match_fn=exact_match,
        stop_strings=[],
        default_fewshot_k=0,
        fewshot_answer_field="answer",
        description="MMLU-Pro - harder MCQ with up to 10 options (first-token log-probability)",
        eval_mode="logprob_token",
        choice_labels=CHOICE_LABELS,
        hf_repo="TIGER-Lab/MMLU-Pro",
        hf_split="test",
        hf_fewshot_split="validation",
    )


def get_task_logprob_seq() -> TaskConfig:
    """Get MMLU-Pro task (completion log-likelihood)."""
    return TaskConfig(
        name="mmlu_pro_logprob_seq",
        build_prompt=build_prompt,
        extract_gold=extract_gold,
        extract_pred=extract_pred,
        match_fn=exact_match,
        stop_strings=[],
        default_fewshot_k=0,
        fewshot_answer_field="answer",
        description="MMLU-Pro - harder MCQ with up to 10 options (completion log-likelihood)",
        eval_mode="logprob_seq",
        choice_labels=CHOICE_LABELS,
        get_choice_texts=get_choice_texts,
        hf_repo="TIGER-Lab/MMLU-Pro",
        hf_split="test",
        hf_fewshot_split="validation",
    )
