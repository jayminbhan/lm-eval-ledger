# tasks/__init__.py
"""
Benchmark task registry.

Each task module should define a `get_task()` function that returns a TaskConfig.
"""
from __future__ import annotations
from typing import Callable

from .base import TaskConfig, load_jsonl, load_from_hf

# Import all task modules
from . import gsm8k
from . import aime
from . import math500
from . import mmlu_redux_2
from . import mmlu_redux_1
from . import mmlu_pro
from . import olympiad_bench
from . import arc
from . import hellaswag
from . import winogrande
from . import hendrycks_math
from . import theoremqa
from . import bbh
from . import gpqa


# ============================================================
# Task Registry
# ============================================================

TASK_REGISTRY: dict[str, Callable[..., TaskConfig]] = {
    # GSM8K
    "gsm8k_main": gsm8k.get_task_main,
    "gsm8k_socratic": gsm8k.get_task_socratic,

    # AIME
    "aime_2024": aime.get_task_2024,
    "aime_2025": aime.get_task_2025,

    # MATH-500
    "math500": math500.get_task,

    # Hendrycks MATH (7 subjects aggregated)
    "hendrycks_math": hendrycks_math.get_task,

    # MMLU Redux 2.0
    "mmlu_redux_2_generate": mmlu_redux_2.get_task,
    "mmlu_redux_2_logprob_token": mmlu_redux_2.get_task_logprob_token,
    "mmlu_redux_2_logprob_seq": mmlu_redux_2.get_task_logprob_seq,

    # MMLU Redux 1.0 (edinburgh-dawg)
    "mmlu_redux_1_generate": mmlu_redux_1.get_task_generate,
    "mmlu_redux_1_logprob_token": mmlu_redux_1.get_task_logprob_token,
    "mmlu_redux_1_logprob_seq": mmlu_redux_1.get_task_logprob_seq,

    # MMLU-Pro
    "mmlu_pro_generate": mmlu_pro.get_task_generate,
    "mmlu_pro_logprob_token": mmlu_pro.get_task_logprob_token,
    "mmlu_pro_logprob_seq": mmlu_pro.get_task_logprob_seq,

    # ARC (AI2 Reasoning Challenge)
    "arc_easy_generate": arc.get_task_easy_generate,
    "arc_easy_logprob_token": arc.get_task_easy_logprob_token,
    "arc_easy_logprob_seq": arc.get_task_easy_logprob_seq,
    "arc_challenge_generate": arc.get_task_challenge_generate,
    "arc_challenge_logprob_token": arc.get_task_challenge_logprob_token,
    "arc_challenge_logprob_seq": arc.get_task_challenge_logprob_seq,

    # HellaSwag
    "hellaswag_generate": hellaswag.get_task_generate,
    "hellaswag_logprob_token": hellaswag.get_task_logprob_token,
    "hellaswag_logprob_seq": hellaswag.get_task_logprob_seq,

    # WinoGrande
    "winogrande_generate": winogrande.get_task_generate,
    "winogrande_logprob_token": winogrande.get_task_logprob_token,
    "winogrande_logprob_seq": winogrande.get_task_logprob_seq,

    # OlympiadBench variants
    "olympiad_bench_math_en": olympiad_bench.get_task_math_en,
    "olympiad_bench_physics_en": olympiad_bench.get_task_physics_en,

    # TheoremQA
    "theoremqa": theoremqa.get_task,

    # BIG-Bench Hard (27 subtasks aggregated)
    "bbh": bbh.get_task,

    # GPQA (Graduate-Level Google-Proof Q&A)
    "gpqa_diamond_generate": gpqa.get_task_diamond_generate,
    "gpqa_diamond_logprob_token": gpqa.get_task_diamond_logprob_token,
    "gpqa_diamond_logprob_seq": gpqa.get_task_diamond_logprob_seq,
    "gpqa_main_generate": gpqa.get_task_main_generate,
    "gpqa_main_logprob_token": gpqa.get_task_main_logprob_token,
    "gpqa_main_logprob_seq": gpqa.get_task_main_logprob_seq,
    "gpqa_extended_generate": gpqa.get_task_extended_generate,
    "gpqa_extended_logprob_token": gpqa.get_task_extended_logprob_token,
    "gpqa_extended_logprob_seq": gpqa.get_task_extended_logprob_seq,
}


def get_available_tasks() -> list[str]:
    """Return list of available task names."""
    return list(TASK_REGISTRY.keys())


def get_task(task_name: str) -> TaskConfig:
    """
    Get a task configuration by name.

    Args:
        task_name: Name of the task (e.g., "gsm8k_main", "aime_2024")

    Returns:
        TaskConfig instance
    """
    if task_name not in TASK_REGISTRY:
        available = ", ".join(sorted(TASK_REGISTRY.keys()))
        raise ValueError(f"Unknown task '{task_name}'. Available tasks: {available}")

    return TASK_REGISTRY[task_name]()


__all__ = [
    "TaskConfig",
    "TASK_REGISTRY",
    "get_available_tasks",
    "get_task",
    "load_jsonl",
    "load_from_hf",
]
