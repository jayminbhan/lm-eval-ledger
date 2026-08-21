# tasks/base.py
"""Base classes and utilities for benchmark tasks."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
import json
import re


@dataclass
class TaskConfig:
    """Configuration for a benchmark task."""
    name: str
    build_prompt: Callable[[dict, str], str]
    extract_gold: Callable[[dict], str]
    extract_pred: Callable[[str], str]
    match_fn: Callable[[str, str], bool]
    stop_strings: list[str] = field(default_factory=list)

    # Curated few-shot examples file (relative to data/ directory)
    fewshot_path: str = ""

    # Default few-shot count
    default_fewshot_k: int = 0

    # Field names for few-shot examples (task-specific)
    fewshot_answer_field: str = "answer"  # For simple answer-only format
    # For solution+answer format (reasoning with #### delimiter)
    fewshot_solution_field: str = ""  # Field containing reasoning/solution
    fewshot_final_answer_field: str = ""  # Field containing final answer (separate from solution)

    # Description for help text
    description: str = ""

    # Evaluation mode:
    #   "generate"      - text generation + regex extraction
    #   "logprob_token"  - first-token log-probability (pick letter with highest P)
    #   "logprob_seq"    - completion log-likelihood (score full answer text, normalized by token count)
    eval_mode: str = "generate"
    # Choice labels for logprob MCQ evaluation (e.g., ["A","B","C","D"])
    choice_labels: list[str] = field(default_factory=list)
    # Function to get choice texts for logprob_seq (e.g., ["The Moon's pull", "Wind", ...])
    get_choice_texts: Callable[[dict], list[str]] | None = None

    # HuggingFace datasets auto-download (None = local-only)
    hf_repo: str | None = None              # e.g., "openai/gsm8k"
    hf_config: str | None = None            # e.g., "main", "ARC-Challenge"
    hf_split: str = "test"                  # split for test data
    hf_fewshot_split: str | None = None     # split for fewshot data (e.g., "train")
    hf_fewshot_config: str | None = None    # if fewshot from a different config
    hf_configs: list[str] | None = None     # multi-config datasets (iterate all configs)
    hf_config_field: str | None = None      # field name to tag examples (e.g., "subtask")
    hf_post_process: Callable[[list[dict]], list[dict]] | None = None  # post-load transform


# ============================================================
# Common utilities
# ============================================================

def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL file into list of dicts."""
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_from_hf(task: TaskConfig, split: str | None = None) -> list[dict]:
    """Load dataset from HuggingFace using the datasets library.

    Args:
        task: TaskConfig with hf_repo set.
        split: Override split (defaults to task.hf_split).

    Returns:
        list[dict] of examples.

    Raises:
        ValueError if hf_repo is not configured.
    """
    from datasets import load_dataset

    if not task.hf_repo:
        raise ValueError(f"Task '{task.name}' has no hf_repo configured")

    split = split or task.hf_split

    if task.hf_configs:
        # Multi-config: iterate all configs and tag each example
        all_examples = []
        n = len(task.hf_configs)
        for i, config_name in enumerate(task.hf_configs, 1):
            print(f"  [{i}/{n}] Loading {task.hf_repo}/{config_name} split={split}")
            ds = load_dataset(task.hf_repo, config_name, split=split)
            rows = [dict(row) for row in ds]
            if task.hf_config_field:
                for row in rows:
                    row[task.hf_config_field] = config_name
            all_examples.extend(rows)
        examples = all_examples
    else:
        # Single config
        ds = load_dataset(task.hf_repo, task.hf_config, split=split)
        examples = [dict(row) for row in ds]

    if task.hf_post_process:
        examples = task.hf_post_process(examples)

    return examples


# ============================================================
# Common answer extraction functions
# ============================================================

_NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")
_BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def extract_number(text: str) -> str:
    """Extract first number from text, stripping commas from result."""
    match = _NUMBER_RE.search(text)
    if match:
        return match.group(0).replace(",", "")
    return text.strip()


def extract_last_number(text: str) -> str:
    """Extract last number from text, stripping commas from result."""
    matches = list(_NUMBER_RE.finditer(text))
    if matches:
        return matches[-1].group(0).replace(",", "")
    return text.strip()


def extract_boxed(text: str) -> str:
    """Extract content from \\boxed{...}, falling back to the full text."""
    match = _BOXED_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def extract_boxed_strict(text: str) -> str:
    """Extract content from the FIRST \\boxed{...}, or "" if none found."""
    match = _BOXED_RE.search(text)
    if match:
        return match.group(1).strip()
    return ""


def extract_boxed_letter(text: str, labels: list[str]) -> str:
    """Extract an MCQ choice letter from \\boxed{...}.

    Returns "" if there is no \\boxed{} or its first character isn't a valid label.
    """
    answer = extract_boxed_strict(text).upper()
    if answer and answer[0] in labels:
        return answer[0]
    return ""


def extract_after_marker(text: str, marker: str = "####") -> str:
    """Extract text after a marker (e.g., ####)."""
    if marker in text:
        return text.split(marker)[-1].strip()
    return text.strip()


# ============================================================
# Common matching functions
# ============================================================

def exact_match(gold: str, pred: str) -> bool:
    """Exact string match after stripping whitespace."""
    return gold.strip() == pred.strip()


def numeric_match(gold: str, pred: str) -> bool:
    """Match numbers, handling floats and integers."""
    try:
        g = float(gold.replace(",", ""))
        p = float(pred.replace(",", ""))
        # Check if both are integers
        if g == int(g) and p == int(p):
            return int(g) == int(p)
        return abs(g - p) < 1e-6
    except (ValueError, TypeError):
        return gold.strip() == pred.strip()


def normalize_answer(text: str) -> str:
    """Normalize answer text for comparison."""
    text = text.strip().lower()
    # Remove LaTeX sizing/formatting commands (before removing backslashes)
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\bigl", "").replace("\\bigr", "")
    text = text.replace("\\Bigl", "").replace("\\Bigr", "")
    text = text.replace("\\biggl", "").replace("\\biggr", "")
    text = text.replace("\\Biggl", "").replace("\\Biggr", "")
    # Remove common LaTeX artifacts
    text = text.replace("\\", "").replace("$", "")
    text = text.replace("{", "").replace("}", "")
    text = text.replace(" ", "")
    return text


def normalized_match(gold: str, pred: str) -> bool:
    """Match after normalizing both strings."""
    return normalize_answer(gold) == normalize_answer(pred)
