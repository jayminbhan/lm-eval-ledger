# tasks/arc.py
"""ARC (AI2 Reasoning Challenge) benchmark tasks - Science MCQ."""
from __future__ import annotations

from .base import TaskConfig, exact_match, extract_boxed_letter

CHOICE_LABELS = ["A", "B", "C", "D", "E"]


def build_prompt(example: dict, fewshot_block: str) -> str:
    """Build prompt for ARC with multiple choice format."""
    question = example.get("question", "").strip()
    choices = example.get("choices", {})

    # choices is {'text': [...], 'label': [...]}
    texts = choices.get("text", [])
    labels = choices.get("label", [])
    choice_text = "\n".join(f"{label}. {text}" for label, text in zip(labels, texts))

    prompt = f"Question: {question}\n{choice_text}\nAnswer:"

    if fewshot_block:
        return f"{fewshot_block}\n\n{prompt}"
    return prompt


def extract_gold(example: dict) -> str:
    """Extract gold answer from ARC example."""
    return str(example.get("answerKey", "")).strip().upper()


def extract_pred(model_output: str) -> str:
    """Extract predicted choice letter from \\boxed{...} in model output."""
    return extract_boxed_letter(model_output, CHOICE_LABELS)


def get_choice_texts(example: dict) -> list[str]:
    """Get list of choice texts for completion log-likelihood scoring."""
    choices = example.get("choices", {})
    return choices.get("text", [])


def _make_task(name: str, hf_config: str, description: str, eval_mode: str) -> TaskConfig:
    """Build an ARC TaskConfig for the given subset and eval mode."""
    return TaskConfig(
        name=name,
        build_prompt=build_prompt,
        extract_gold=extract_gold,
        extract_pred=extract_pred,
        match_fn=exact_match,
        stop_strings=["Question:"] if eval_mode == "generate" else [],
        default_fewshot_k=0,
        fewshot_answer_field="answerKey",
        description=description,
        eval_mode=eval_mode,
        choice_labels=CHOICE_LABELS if eval_mode != "generate" else [],
        get_choice_texts=get_choice_texts if eval_mode == "logprob_seq" else None,
        hf_repo="allenai/ai2_arc",
        hf_config=hf_config,
        hf_split="test",
        hf_fewshot_split="train",
    )


# ── ARC-Easy ─────────────────────────────────────────────

def get_task_easy_generate() -> TaskConfig:
    return _make_task(
        "arc_easy", "ARC-Easy",
        "ARC-Easy - science MCQ (generative)", "generate")


def get_task_easy_logprob_token() -> TaskConfig:
    return _make_task(
        "arc_easy_logprob_token", "ARC-Easy",
        "ARC-Easy - science MCQ (first-token log-probability)", "logprob_token")


def get_task_easy_logprob_seq() -> TaskConfig:
    return _make_task(
        "arc_easy_logprob_seq", "ARC-Easy",
        "ARC-Easy - science MCQ (completion log-likelihood)", "logprob_seq")


# ── ARC-Challenge ────────────────────────────────────────

def get_task_challenge_generate() -> TaskConfig:
    return _make_task(
        "arc_challenge", "ARC-Challenge",
        "ARC-Challenge - science MCQ (generative)", "generate")


def get_task_challenge_logprob_token() -> TaskConfig:
    return _make_task(
        "arc_challenge_logprob_token", "ARC-Challenge",
        "ARC-Challenge - science MCQ (first-token log-probability)", "logprob_token")


def get_task_challenge_logprob_seq() -> TaskConfig:
    return _make_task(
        "arc_challenge_logprob_seq", "ARC-Challenge",
        "ARC-Challenge - science MCQ (completion log-likelihood)", "logprob_seq")
