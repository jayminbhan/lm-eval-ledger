# tasks/mrcr.py
"""MRCR (Multi-Round Co-reference Resolution) long-context benchmark.

Origin: introduced in Vodrahalli et al., "Michelangelo: Long Context
Evaluations Beyond Haystacks via Latent Structure Queries", arXiv:2409.12640
(Google DeepMind, 2024). OpenAI released this expanded open-source version
alongside GPT-4.1.
Dataset: openai/mrcr (official).

Each example is a long synthetic user/assistant conversation containing
2, 4, or 8 identical requests ("needles", e.g. "write a poem about
tapirs"); the model must reproduce the i-th needle's answer exactly,
prefixed with a per-example random string. Prompts range from ~4k to ~1M
tokens, so set max_model_len as high as your hardware allows - examples
that don't fit the context window are skipped with a warning.

Scoring (official): SequenceMatcher ratio between response and gold answer,
0 if the response doesn't start with the random prefix. This is a partial-
credit score in [0, 1]; the task's "accuracy" is the mean ratio, matching
the official metric. Run with apply_chat_template: true - the conversation
is fed as real multi-turn chat messages.
"""
from __future__ import annotations

import json
import re
from difflib import SequenceMatcher

from .base import TaskConfig

# Trailing special tokens (e.g. "<|im_end|>") kept by skip_special_tokens=False
# would otherwise slightly depress the match ratio.
_TRAILING_SPECIAL_RE = re.compile(r"(?:<\|[^|]*\|>\s*)+$")


def _make_post_process(n_needles: int):
    def _post(examples: list[dict]) -> list[dict]:
        kept = []
        for idx, ex in enumerate(examples):
            if ex.get("n_needles") == n_needles:
                ex["id"] = idx
                kept.append(ex)
        # Shortest first: examples over the context budget are skipped at run
        # time, so this ordering lets max_examples subsets (smoke tests) pick
        # examples that actually fit. Full runs evaluate every fitting example
        # regardless of order.
        kept.sort(key=lambda ex: ex.get("n_chars", 0))
        print(f"  [INFO] MRCR: kept {len(kept)} examples with {n_needles} needles "
              f"(sorted shortest-first)")
        return kept
    return _post


def build_messages(example: dict) -> list[dict]:
    """The example's prompt field is the full conversation as JSON messages."""
    return json.loads(example["prompt"])


def build_prompt(example: dict, fewshot_block: str) -> str:
    """Plain-text fallback for base models (MRCR is meant for chat models).

    Ignores fewshot_block: the model must reply with the prefixed answer
    only, so no format instruction may be injected.
    """
    messages = build_messages(example)
    parts = [f"{m['role']}: {m['content']}" for m in messages]
    return "\n\n".join(parts) + "\n\nassistant:"


def extract_gold(example: dict) -> str:
    return json.dumps({
        "answer": example["answer"],
        "prepend": example["random_string_to_prepend"],
    })


def extract_gold_display(example: dict) -> str:
    """The expected answer text itself (grading metadata goes to gold_data)."""
    return example["answer"]


def extract_pred(model_output: str) -> str:
    """The raw response is graded; only strip leading whitespace and
    trailing special tokens (artifacts of raw-completion decoding)."""
    return _TRAILING_SPECIAL_RE.sub("", model_output.lstrip()).rstrip()


def match_fn(gold: str, pred: str) -> float:
    """Official MRCR grader: 0 unless the response starts with the random
    prefix; otherwise SequenceMatcher ratio against the gold answer."""
    data = json.loads(gold)
    answer, prepend = data["answer"], data["prepend"]
    if not pred.startswith(prepend):
        return 0.0
    pred = pred.removeprefix(prepend)
    answer = answer.removeprefix(prepend)
    return float(SequenceMatcher(None, pred, answer).ratio())


def _make_task(n_needles: int) -> TaskConfig:
    return TaskConfig(
        name=f"mrcr_{n_needles}needle",
        build_prompt=build_prompt,
        build_messages=build_messages,
        extract_gold=extract_gold,
        extract_gold_display=extract_gold_display,
        extract_pred=extract_pred,
        match_fn=match_fn,
        default_fewshot_k=0,
        description=f"MRCR {n_needles}-needle - long-context co-reference "
                    "(partial credit: mean SequenceMatcher ratio; "
                    "use apply_chat_template and a large max_model_len)",
        hf_repo="openai/mrcr",
        hf_split="train",
        hf_post_process=_make_post_process(n_needles),
    )


def get_task_2needle() -> TaskConfig:
    return _make_task(2)


def get_task_4needle() -> TaskConfig:
    return _make_task(4)


def get_task_8needle() -> TaskConfig:
    return _make_task(8)
