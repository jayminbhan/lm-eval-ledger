# tasks/mmlu_redux_2.py
"""MMLU Redux 2.0 benchmark task - Multiple choice question answering."""
from __future__ import annotations

from .base import TaskConfig, exact_match, extract_boxed_letter

CHOICE_LABELS = ["A", "B", "C", "D"]

# List of all available subjects
MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history", "human_aging",
    "human_sexuality", "international_law", "jurisprudence",
    "logical_fallacies", "machine_learning", "management", "marketing",
    "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
    "nutrition", "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology", "us_foreign_policy",
    "virology", "world_religions",
]


def build_prompt(example: dict, fewshot_block: str) -> str:
    """Build prompt for MMLU with multiple choice format."""
    question = example.get("question", "").strip()
    choices = example.get("choices", [])

    choice_text = "\n".join(
        f"{CHOICE_LABELS[i]}. {choice}"
        for i, choice in enumerate(choices)
    )

    prompt = f"Question: {question}\n{choice_text}\nAnswer:"

    if fewshot_block:
        return f"{fewshot_block}\n\n{prompt}"
    return prompt


def extract_gold(example: dict) -> str:
    """Extract gold answer from MMLU example (index 0-3 -> letter)."""
    answer_idx = example.get("answer", 0)
    if isinstance(answer_idx, int) and 0 <= answer_idx < 4:
        return CHOICE_LABELS[answer_idx]
    return str(answer_idx)


def extract_pred(model_output: str) -> str:
    """Extract predicted choice letter from \\boxed{...} in model output."""
    return extract_boxed_letter(model_output, CHOICE_LABELS)


def get_choice_texts(example: dict) -> list[str]:
    """Get list of choice texts for completion log-likelihood scoring."""
    return example.get("choices", [])


def get_task() -> TaskConfig:
    """Get MMLU Redux 2.0 task (generative)."""
    return TaskConfig(
        name="mmlu_redux_2",
        build_prompt=build_prompt,
        extract_gold=extract_gold,
        extract_pred=extract_pred,
        match_fn=exact_match,
        stop_strings=["Question:"],
        default_fewshot_k=0,
        fewshot_answer_field="answer",
        description="MMLU Redux 2.0 - multiple choice across many subjects",
        hf_repo="edinburgh-dawg/mmlu-redux-2.0",
        hf_split="test",
        hf_configs=MMLU_SUBJECTS,
        hf_config_field="subject",
    )


def get_task_logprob_token() -> TaskConfig:
    """Get MMLU Redux 2.0 task (first-token log-probability)."""
    return TaskConfig(
        name="mmlu_redux_2_logprob_token",
        build_prompt=build_prompt,
        extract_gold=extract_gold,
        extract_pred=extract_pred,
        match_fn=exact_match,
        stop_strings=[],
        default_fewshot_k=0,
        fewshot_answer_field="answer",
        description="MMLU Redux 2.0 - multiple choice (first-token log-probability)",
        eval_mode="logprob_token",
        choice_labels=CHOICE_LABELS,
        hf_repo="edinburgh-dawg/mmlu-redux-2.0",
        hf_split="test",
        hf_configs=MMLU_SUBJECTS,
        hf_config_field="subject",
    )


def get_task_logprob_seq() -> TaskConfig:
    """Get MMLU Redux 2.0 task (completion log-likelihood)."""
    return TaskConfig(
        name="mmlu_redux_2_logprob_seq",
        build_prompt=build_prompt,
        extract_gold=extract_gold,
        extract_pred=extract_pred,
        match_fn=exact_match,
        stop_strings=[],
        default_fewshot_k=0,
        fewshot_answer_field="answer",
        description="MMLU Redux 2.0 - multiple choice (completion log-likelihood)",
        eval_mode="logprob_seq",
        choice_labels=CHOICE_LABELS,
        get_choice_texts=get_choice_texts,
        hf_repo="edinburgh-dawg/mmlu-redux-2.0",
        hf_split="test",
        hf_configs=MMLU_SUBJECTS,
        hf_config_field="subject",
    )
