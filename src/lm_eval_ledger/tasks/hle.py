# tasks/hle.py
"""HLE (Humanity's Last Exam) benchmark task - text-only subset.

Paper: Phan et al., "Humanity's Last Exam", arXiv:2501.14249 (2025).
Dataset: cais/hle (official; GATED - request access on the HF dataset page
and log in with `huggingface-cli login` before running).

2,500 expert-written questions across 100+ subjects, designed to be the
"final" closed-ended academic benchmark. Questions are either exact-match
or multiple-choice (choices embedded in the question text). ~10% of
questions require images; those are filtered out here (text-only eval).

Scoring note: the official leaderboard verifies exact-match answers with an
LLM judge (o3-mini). This task uses normalized string matching instead,
which is stricter - treat scores as a lower bound.
"""
from __future__ import annotations

from .base import TaskConfig, extract_boxed_strict, normalized_match


def _hf_post_process(examples: list[dict]) -> list[dict]:
    """Keep text-only questions (drop any with an attached image)."""
    text_only = [ex for ex in examples if not ex.get("image")]
    print(f"  [INFO] HLE: {len(text_only)}/{len(examples)} text-only questions kept")
    return text_only


def build_prompt(example: dict, fewshot_block: str) -> str:
    """Build prompt for HLE (choices, if any, are already in the question)."""
    q = example["question"].strip()
    if fewshot_block:
        return f"{fewshot_block}\n\nQuestion: {q}\nAnswer:"
    return f"Question: {q}\nAnswer:"


def extract_gold(example: dict) -> str:
    return str(example.get("answer", "")).strip()


def extract_pred(model_output: str) -> str:
    return extract_boxed_strict(model_output)


def match_fn(gold: str, pred: str) -> bool:
    """Normalized match; for single-letter MCQ golds also accept 'B. text' forms."""
    if not pred:
        return False
    if normalized_match(gold, pred):
        return True
    # Multiple-choice questions have a single-letter gold (A-Z); accept a
    # prediction whose first character is that letter (e.g. "B" vs "B. 7.2 eV")
    g = gold.strip().upper()
    if len(g) == 1 and g.isalpha():
        return pred.strip().upper()[:1] == g
    return False


def get_task() -> TaskConfig:
    """Get HLE (text-only) task configuration."""
    return TaskConfig(
        name="hle",
        build_prompt=build_prompt,
        extract_gold=extract_gold,
        extract_pred=extract_pred,
        match_fn=match_fn,
        stop_strings=["Question:"],
        default_fewshot_k=0,
        description="Humanity's Last Exam - text-only expert questions "
                    "(gated dataset; string-match approximation of LLM-judge scoring)",
        hf_repo="cais/hle",
        hf_split="test",
        hf_post_process=_hf_post_process,
    )
