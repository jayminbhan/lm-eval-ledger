# backends/__init__.py
"""Backend registry and factory.

Backends are imported lazily so the base install stays light; a missing
engine produces an actionable install hint instead of an ImportError.
"""
from __future__ import annotations

from .base import Backend, GenResult

#: backend name -> (module, class, pip extra)
_BACKENDS = {
    "vllm": (".vllm_backend", "VllmBackend", "vllm"),
    "hf": (".hf_backend", "HfBackend", "hf"),
    "server": (".server_backend", "ServerBackend", None),
    "sglang": (".sglang_backend", "SglangBackend", "sglang"),
}

BACKEND_NAMES = tuple(_BACKENDS)


def get_backend(name: str) -> Backend:
    """Instantiate a backend by name, with an install hint on failure."""
    if name not in _BACKENDS:
        raise ValueError(
            f"Unknown backend {name!r}. Available: {', '.join(BACKEND_NAMES)}")
    module_name, class_name, extra = _BACKENDS[name]
    import importlib
    import sys
    try:
        module = importlib.import_module(module_name, package=__name__)
    except ImportError as e:
        if name in ("vllm", "sglang") and sys.platform != "linux":
            raise RuntimeError(
                f"Backend {name!r} is Linux-only ({name} publishes no "
                f"{sys.platform} wheels). On this platform use "
                f"backend: hf, or backend: server against a native "
                f"llama.cpp / ollama / LM Studio endpoint.") from e
        hint = (f"pip install lm-eval-ledger[{extra}]" if extra
                else "pip install httpx")
        raise RuntimeError(
            f"Backend {name!r} is not installed ({e}). Install it with: {hint}"
        ) from e
    return getattr(module, class_name)()


__all__ = ["Backend", "GenResult", "get_backend", "BACKEND_NAMES"]
