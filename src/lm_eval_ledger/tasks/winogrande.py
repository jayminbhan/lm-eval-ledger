# tasks/winogrande.py
"""WinoGrande benchmark task - Commonsense coreference resolution MCQ."""
from __future__ import annotations

from .base import TaskConfig, exact_match, extract_boxed_letter

CHOICE_LABELS = ["A", "B"]


def build_prompt(example: dict, fewshot_block: str) -> str:
    """Build prompt for WinoGrande with binary choice format."""
    sentence = example.get("sentence", "").strip()
    option1 = example.get("option1", "").strip()
    option2 = example.get("option2", "").strip()

    prompt = (
        f"Fill in the blank:\n{sentence}\n"
        f"A. {option1}\n"
        f"B. {option2}\n"
        f"Answer:"
    )

    if fewshot_block:
        return f"{fewshot_block}\n\n{prompt}"
    return prompt


def extract_gold(example: dict) -> str:
    """Extract gold answer from WinoGrande example ("1"/"2" -> A/B)."""
    answer = str(example.get("answer", "1")).strip()
    if answer == "1":
        return "A"
    if answer == "2":
        return "B"
    return answer


def extract_pred(model_output: str) -> str:
    """Extract predicted choice letter from \\boxed{...} in model output."""
    return extract_boxed_letter(model_output, CHOICE_LABELS)


def get_choice_texts(example: dict) -> list[str]:
    """Get list of choice texts for completion log-likelihood scoring."""
    return [
        example.get("option1", "").strip(),
        example.get("option2", "").strip(),
    ]


def get_task_generate() -> TaskConfig:
    """Get WinoGrande task (generative)."""
    return TaskConfig(
        name="winogrande",
        build_prompt=build_prompt,
        extract_gold=extract_gold,
        extract_pred=extract_pred,
        match_fn=exact_match,
        stop_strings=["Fill in the blank:"],
        default_fewshot_k=0,
        fewshot_answer_field="answer",
        description="WinoGrande - commonsense coreference resolution (generative)",
        hf_repo="allenai/winogrande",
        hf_config="winogrande_xl",
        hf_split="validation",
        hf_fewshot_split="train",
    )


def get_task_logprob_token() -> TaskConfig:
    """Get WinoGrande task (first-token log-probability)."""
    return TaskConfig(
        name="winogrande_logprob_token",
        build_prompt=build_prompt,
        extract_gold=extract_gold,
        extract_pred=extract_pred,
        match_fn=exact_match,
        stop_strings=[],
        default_fewshot_k=0,
        fewshot_answer_field="answer",
        description="WinoGrande - commonsense coreference resolution (first-token log-probability)",
        eval_mode="logprob_token",
        choice_labels=CHOICE_LABELS,
        hf_repo="allenai/winogrande",
        hf_config="winogrande_xl",
        hf_split="validation",
        hf_fewshot_split="train",
    )


def get_task_logprob_seq() -> TaskConfig:
    """Get WinoGrande task (completion log-likelihood)."""
    return TaskConfig(
        name="winogrande_logprob_seq",
        build_prompt=build_prompt,
        extract_gold=extract_gold,
        extract_pred=extract_pred,
        match_fn=exact_match,
        stop_strings=[],
        default_fewshot_k=0,
        fewshot_answer_field="answer",
        description="WinoGrande - commonsense coreference resolution (completion log-likelihood)",
        eval_mode="logprob_seq",
        choice_labels=CHOICE_LABELS,
        get_choice_texts=get_choice_texts,
        hf_repo="allenai/winogrande",
        hf_config="winogrande_xl",
        hf_split="validation",
        hf_fewshot_split="train",
    )
