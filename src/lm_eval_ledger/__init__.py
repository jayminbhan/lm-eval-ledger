"""lm-eval-ledger: vLLM benchmark harness that records everything to SQLite.

Library usage:

    from lm_eval_ledger import RunConfig, run

    cfg = RunConfig(models=["Qwen/Qwen2.5-0.5B-Instruct"],
                    tasks=[("gsm8k", 0)], max_examples=10)
    db_path = run(cfg)

Custom tasks:

    from lm_eval_ledger import TaskConfig, register_task
    register_task("my_task", my_task_factory)
"""
from __future__ import annotations

try:  # single source of truth: pyproject's version, via installed metadata
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("lm-eval-ledger")
except Exception:  # running from a bare source tree
    __version__ = "0.0.0+src"

from .config import RunConfig, load_yaml_config, resolve_config
from .db import LedgerDatabase
from .tasks import (
    TASK_REGISTRY,
    TaskConfig,
    get_available_tasks,
    get_task,
    register_task,
)


def run(cfg: RunConfig, **kwargs):
    """Run all configured benchmarks; returns the results database path.

    Imports the runner lazily so that `import lm_eval_ledger` stays cheap
    (the runner pulls in vLLM).
    """
    from .runner import run as _run
    return _run(cfg, **kwargs)


def verify_run(db_path, verifier_model, **kwargs):
    """LLM-verify a finished run's database (see lm_eval_ledger.verifier).

    Lazy import: the verifier pulls in vLLM.
    """
    from .verifier import verify_run as _verify
    return _verify(db_path, verifier_model, **kwargs)


__all__ = [
    "__version__",
    "RunConfig",
    "run",
    "verify_run",
    "LedgerDatabase",
    "load_yaml_config",
    "resolve_config",
    "TaskConfig",
    "TASK_REGISTRY",
    "register_task",
    "get_available_tasks",
    "get_task",
]
