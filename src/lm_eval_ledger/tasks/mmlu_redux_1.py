# tasks/mmlu_redux_1.py
"""MMLU Redux 1.0 (edinburgh-dawg) benchmark task - Error-annotated MMLU subset."""
from __future__ import annotations

from .base import apply_mmlu_redux_annotations, TaskConfig, exact_match, extract_boxed_letter

CHOICE_LABELS = ["A", "B", "C", "D"]


def build_prompt(example: dict, fewshot_block: str) -> str:
    """Build prompt for MMLU Redux 1.0 with multiple choice format."""
    question = example.get("question", "").strip()
    choices = example.get("choices", [])

    # choices is a list of strings
    choice_text = "\n".join(
        f"{CHOICE_LABELS[i]}. {choice}"
        for i, choice in enumerate(choices)
    )

    prompt = f"Question: {question}\n{choice_text}\nAnswer:"

    if fewshot_block:
        return f"{fewshot_block}\n\n{prompt}"
    return prompt


def extract_gold(example: dict) -> str:
    """Extract gold answer from MMLU Redux 1.0 example (index -> letter)."""
    answer_idx = example.get("answer", 0)
    if isinstance(answer_idx, (int, float)) and 0 <= int(answer_idx) < 4:
        return CHOICE_LABELS[int(answer_idx)]
    return str(answer_idx)


def extract_pred(model_output: str) -> str:
    """Extract predicted choice letter from \\boxed{...} in model output."""
    return extract_boxed_letter(model_output, CHOICE_LABELS)


def get_choice_texts(example: dict) -> list[str]:
    """Get list of choice texts for completion log-likelihood scoring."""
    return example.get("choices", [])


MMLU_REDUX_1_SUBJECTS = [
    "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
    "college_chemistry", "college_computer_science", "college_mathematics",
    "college_medicine", "college_physics", "conceptual_physics",
    "econometrics", "electrical_engineering", "formal_logic", "global_facts",
    "high_school_chemistry", "high_school_geography",
    "high_school_macroeconomics", "high_school_mathematics",
    "high_school_physics", "high_school_statistics", "high_school_us_history",
    "human_aging", "logical_fallacies", "machine_learning", "miscellaneous",
    "philosophy", "professional_accounting", "professional_law",
    "public_relations", "virology",
]


def get_task_generate() -> TaskConfig:
    """Get MMLU Redux 1.0 task (generative)."""
    return TaskConfig(
        name="mmlu_redux_1",
        build_prompt=build_prompt,
        extract_gold=extract_gold,
        extract_pred=extract_pred,
        match_fn=exact_match,
        stop_strings=["Question:"],
        default_fewshot_k=0,
        description="MMLU Redux 1.0 - error-annotated MMLU subset (generative)",
        hf_repo="edinburgh-dawg/mmlu-redux",
        hf_split="test",
        hf_configs=MMLU_REDUX_1_SUBJECTS,
        hf_post_process=apply_mmlu_redux_annotations,
        hf_config_field="subject",
    )


def get_task_logprob_token() -> TaskConfig:
    """Get MMLU Redux 1.0 task (first-token log-probability)."""
    return TaskConfig(
        name="mmlu_redux_1_logprob_token",
        build_prompt=build_prompt,
        extract_gold=extract_gold,
        extract_pred=extract_pred,
        match_fn=exact_match,
        stop_strings=[],
        default_fewshot_k=0,
        description="MMLU Redux 1.0 - error-annotated MMLU subset (first-token log-probability)",
        eval_mode="logprob_token",
        choice_labels=CHOICE_LABELS,
        hf_repo="edinburgh-dawg/mmlu-redux",
        hf_split="test",
        hf_configs=MMLU_REDUX_1_SUBJECTS,
        hf_post_process=apply_mmlu_redux_annotations,
        hf_config_field="subject",
    )


def get_task_logprob_seq() -> TaskConfig:
    """Get MMLU Redux 1.0 task (completion log-likelihood)."""
    return TaskConfig(
        name="mmlu_redux_1_logprob_seq",
        build_prompt=build_prompt,
        extract_gold=extract_gold,
        extract_pred=extract_pred,
        match_fn=exact_match,
        stop_strings=[],
        default_fewshot_k=0,
        description="MMLU Redux 1.0 - error-annotated MMLU subset (completion log-likelihood)",
        eval_mode="logprob_seq",
        choice_labels=CHOICE_LABELS,
        get_choice_texts=get_choice_texts,
        hf_repo="edinburgh-dawg/mmlu-redux",
        hf_split="test",
        hf_configs=MMLU_REDUX_1_SUBJECTS,
        hf_post_process=apply_mmlu_redux_annotations,
        hf_config_field="subject",
    )
