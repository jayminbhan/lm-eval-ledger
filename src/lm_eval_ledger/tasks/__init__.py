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
from . import hle
from . import livecodebench
from . import mrcr


# ============================================================
# Task Registry
# ============================================================

TASK_REGISTRY: dict[str, Callable[..., TaskConfig]] = {
    # GSM8K
    "gsm8k": gsm8k.get_task,

    # AIME
    "aime_2024": aime.get_task_2024,
    "aime_2025": aime.get_task_2025,

    # MATH-500
    "math500": math500.get_task,

    # Hendrycks MATH (7 subjects aggregated)
    "math": hendrycks_math.get_task,

    # MMLU Redux 2.0
    "mmlu_redux_2": mmlu_redux_2.get_task,
    "mmlu_redux_2_logprob_token": mmlu_redux_2.get_task_logprob_token,
    "mmlu_redux_2_logprob_seq": mmlu_redux_2.get_task_logprob_seq,

    # MMLU Redux 1.0 (edinburgh-dawg)
    "mmlu_redux_1": mmlu_redux_1.get_task_generate,
    "mmlu_redux_1_logprob_token": mmlu_redux_1.get_task_logprob_token,
    "mmlu_redux_1_logprob_seq": mmlu_redux_1.get_task_logprob_seq,

    # MMLU-Pro
    "mmlu_pro": mmlu_pro.get_task_generate,
    "mmlu_pro_logprob_token": mmlu_pro.get_task_logprob_token,
    "mmlu_pro_logprob_seq": mmlu_pro.get_task_logprob_seq,

    # ARC (AI2 Reasoning Challenge)
    "arc_easy": arc.get_task_easy_generate,
    "arc_easy_logprob_token": arc.get_task_easy_logprob_token,
    "arc_easy_logprob_seq": arc.get_task_easy_logprob_seq,
    "arc_challenge": arc.get_task_challenge_generate,
    "arc_challenge_logprob_token": arc.get_task_challenge_logprob_token,
    "arc_challenge_logprob_seq": arc.get_task_challenge_logprob_seq,

    # HellaSwag
    "hellaswag": hellaswag.get_task_generate,
    "hellaswag_logprob_token": hellaswag.get_task_logprob_token,
    "hellaswag_logprob_seq": hellaswag.get_task_logprob_seq,

    # WinoGrande
    "winogrande": winogrande.get_task_generate,
    "winogrande_logprob_token": winogrande.get_task_logprob_token,
    "winogrande_logprob_seq": winogrande.get_task_logprob_seq,

    # OlympiadBench variants
    "olympiad_bench_math_en": olympiad_bench.get_task_math_en,
    "olympiad_bench_physics_en": olympiad_bench.get_task_physics_en,

    # TheoremQA
    # "theoremqa": theoremqa.get_task,

    # BIG-Bench Hard (27 subtasks aggregated)
    "bbh": bbh.get_task,

    # GPQA (Graduate-Level Google-Proof Q&A)
    "gpqa_diamond": gpqa.get_task_diamond_generate,
    "gpqa_diamond_logprob_token": gpqa.get_task_diamond_logprob_token,
    "gpqa_diamond_logprob_seq": gpqa.get_task_diamond_logprob_seq,
    "gpqa_main": gpqa.get_task_main_generate,
    "gpqa_main_logprob_token": gpqa.get_task_main_logprob_token,
    "gpqa_main_logprob_seq": gpqa.get_task_main_logprob_seq,
    "gpqa_extended": gpqa.get_task_extended_generate,
    "gpqa_extended_logprob_token": gpqa.get_task_extended_logprob_token,
    "gpqa_extended_logprob_seq": gpqa.get_task_extended_logprob_seq,

    # HLE (Humanity's Last Exam, text-only; gated dataset)
    # "hle": hle.get_task,

    # LiveCodeBench (code generation; executes generated code locally)
    "livecodebench": livecodebench.get_task,
    # per-release delta slices (the upstream-defined units): vN loads
    # only the problems ADDED in release N; the newest delta is the most
    # contamination-safe slice available
    **{f"livecodebench_v{n}": (lambda n=n: livecodebench.get_task_delta(n))
       for n in range(1, 7)},

    # MRCR (long-context multi-round co-reference; partial-credit scoring)
    # "mrcr_2needle": mrcr.get_task_2needle,
    # "mrcr_4needle": mrcr.get_task_4needle,
    # "mrcr_8needle": mrcr.get_task_8needle,
}


def register_task(name: str, factory: Callable[..., TaskConfig],
                  overwrite: bool = False) -> None:
    """Register a custom task so it can be referenced by name in configs.

    Args:
        name: Task name (e.g., "my_task").
        factory: Zero-arg callable returning a TaskConfig.
        overwrite: Allow replacing an existing registration.
    """
    if name in TASK_REGISTRY and not overwrite:
        raise ValueError(
            f"Task '{name}' is already registered (pass overwrite=True to replace)"
        )
    TASK_REGISTRY[name] = factory


def get_available_tasks() -> list[str]:
    """Return list of available task names."""
    return list(TASK_REGISTRY.keys())


def get_task(task_name: str) -> TaskConfig:
    """
    Get a task configuration by name.

    Args:
        task_name: Name of the task (e.g., "gsm8k", "aime_2024")

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
    "register_task",
    "get_available_tasks",
    "get_task",
    "load_jsonl",
    "load_from_hf",
]
