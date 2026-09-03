# tasks/gpqa.py
"""GPQA (Graduate-Level Google-Proof Q&A) benchmark tasks.

Paper: Rein et al., "GPQA: A Graduate-Level Google-Proof Q&A Benchmark",
arXiv:2311.12022 (2023).
Dataset: Idavidrein/gpqa (official; gated - accept terms on the HF page)
Splits: diamond (198), main (448), extended (546)
Format: rows with Question, Correct Answer, Incorrect Answer 1-3.
Choices are shuffled deterministically (seeded by row index) so the
correct answer isn't always in the same position.
"""
from __future__ import annotations

import random

from .base import TaskConfig, exact_match, extract_boxed_letter

CHOICE_LABELS = ["A", "B", "C", "D"]


def _hf_post_process(examples: list[dict]) -> list[dict]:
    """Post-process GPQA from HF: shuffle choices deterministically by row index."""
    processed = []
    for idx, raw in enumerate(examples):
        correct = raw["Correct Answer"].strip()
        incorrects = [
            raw["Incorrect Answer 1"].strip(),
            raw["Incorrect Answer 2"].strip(),
            raw["Incorrect Answer 3"].strip(),
        ]
        choices = [correct] + incorrects
        rng = random.Random(idx)
        rng.shuffle(choices)
        answer_label = CHOICE_LABELS[choices.index(correct)]
        processed.append({
            "question": raw["Question"].strip(),
            "choices": choices,
            "answer": answer_label,
            "domain": raw.get("High-level domain", ""),
            "subdomain": raw.get("Subdomain", ""),
            "idx": idx,
        })
    return processed


def build_prompt(example: dict, fewshot_block: str) -> str:
    """Build prompt for GPQA with multiple choice format."""
    question = example["question"]
    choices = example["choices"]
    choice_text = "\n".join(
        f"{label}. {text}" for label, text in zip(CHOICE_LABELS, choices)
    )

    prompt = f"Question: {question}\n{choice_text}\nAnswer:"

    if fewshot_block:
        return f"{fewshot_block}\n\n{prompt}"
    return prompt


def extract_gold(example: dict) -> str:
    return example["answer"]


def extract_pred(model_output: str) -> str:
    """Extract predicted choice letter from \\boxed{...} in model output."""
    return extract_boxed_letter(model_output, CHOICE_LABELS)


def get_choice_texts(example: dict) -> list[str]:
    """Get list of choice texts for completion log-likelihood scoring."""
    return example.get("choices", [])


def _make_task(name: str, hf_config: str, description: str, eval_mode: str) -> TaskConfig:
    """Build a GPQA TaskConfig for the given subset and eval mode."""
    return TaskConfig(
        name=name,
        build_prompt=build_prompt,
        extract_gold=extract_gold,
        extract_pred=extract_pred,
        match_fn=exact_match,
        stop_strings=["Question:"] if eval_mode == "generate" else [],
        default_fewshot_k=0,
        description=description,
        eval_mode=eval_mode,
        choice_labels=CHOICE_LABELS if eval_mode != "generate" else [],
        get_choice_texts=get_choice_texts if eval_mode == "logprob_seq" else None,
        hf_repo="Idavidrein/gpqa",
        hf_revision="633f5ee89ab8ad4522a9f850766b73f62147ffdd",  # pinned 2026-09
        hf_config=hf_config,
        hf_split="train",
        hf_post_process=_hf_post_process,
    )


# --- Diamond (198) ---

def get_task_diamond_generate() -> TaskConfig:
    return _make_task(
        "gpqa_diamond", "gpqa_diamond",
        "GPQA Diamond - graduate-level science MCQ (generative)", "generate")


def get_task_diamond_logprob_token() -> TaskConfig:
    return _make_task(
        "gpqa_diamond_logprob_token", "gpqa_diamond",
        "GPQA Diamond - graduate-level science MCQ (first-token log-probability)", "logprob_token")


def get_task_diamond_logprob_seq() -> TaskConfig:
    return _make_task(
        "gpqa_diamond_logprob_seq", "gpqa_diamond",
        "GPQA Diamond - graduate-level science MCQ (completion log-likelihood)", "logprob_seq")


# --- Main (448) ---

def get_task_main_generate() -> TaskConfig:
    return _make_task(
        "gpqa_main", "gpqa_main",
        "GPQA Main - graduate-level science MCQ (generative)", "generate")


def get_task_main_logprob_token() -> TaskConfig:
    return _make_task(
        "gpqa_main_logprob_token", "gpqa_main",
        "GPQA Main - graduate-level science MCQ (first-token log-probability)", "logprob_token")


def get_task_main_logprob_seq() -> TaskConfig:
    return _make_task(
        "gpqa_main_logprob_seq", "gpqa_main",
        "GPQA Main - graduate-level science MCQ (completion log-likelihood)", "logprob_seq")


# --- Extended (546) ---

def get_task_extended_generate() -> TaskConfig:
    return _make_task(
        "gpqa_extended", "gpqa_extended",
        "GPQA Extended - graduate-level science MCQ (generative)", "generate")


def get_task_extended_logprob_token() -> TaskConfig:
    return _make_task(
        "gpqa_extended_logprob_token", "gpqa_extended",
        "GPQA Extended - graduate-level science MCQ (first-token log-probability)", "logprob_token")


def get_task_extended_logprob_seq() -> TaskConfig:
    return _make_task(
        "gpqa_extended_logprob_seq", "gpqa_extended",
        "GPQA Extended - graduate-level science MCQ (completion log-likelihood)", "logprob_seq")
