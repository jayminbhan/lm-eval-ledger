# config.py
"""Run configuration for the benchmark harness.

Precedence (lowest to highest):
    RunConfig defaults  <  YAML file (-c/--config, default ./bench.yaml)  <  CLI flags

The YAML file holds only data (which models/tasks to run and with what
settings) - all sweep/run logic stays in run_bench.py. Unknown YAML keys are
rejected so a typo can never silently fall back to a default.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

DEFAULT_CONFIG_FILE = "bench.yaml"


@dataclass
class RunConfig:
    """All settings for a benchmark run."""

    # Models: HuggingFace ids or local paths, run sequentially.
    # "verl:/path/to/run_dir" expands to every global_step_* checkpoint's
    # huggingface/ dir inside that veRL run directory.
    models: list[str] = field(default_factory=list)

    # Tasks: list of (task_name, fewshot_k) pairs; fewshot_k None = task default.
    # Empty list = run ALL registered tasks with their defaults.
    tasks: list[tuple[str, int | None]] = field(default_factory=list)

    # Evaluation size / batching
    max_examples: int | None = None   # None = all examples
    batch_size: int | None = None     # None = single llm.generate call

    # Prompting
    apply_chat_template: bool = False  # True for instruct/chat models
    # Extra kwargs for the model's chat template, passed to every backend's
    # templating (tokenizer kwargs in-process; chat_template_kwargs over the
    # server API). Model-family specific, e.g.:
    #   {enable_thinking: false}   # Qwen3-family hybrid thinking off
    #   {thinking: true}           # Granite
    chat_template_kwargs: dict | None = None

    # Sampling
    temperature: float = 0.0
    top_p: float = 0.95
    max_tokens: int = 1024
    pass_k: int = 1                   # >1 = pass@k (needs temperature > 0)
    seed: int = 42

    # Inference backend: "vllm" (reference; pip install lm-eval-ledger[vllm]),
    # "hf" (transformers+accelerate; [hf]), "sglang" ([sglang]), or
    # "server" (any OpenAI-compatible endpoint: llama.cpp, ollama, ...).
    backend: str = "vllm"

    # Server backend settings (backend: server only)
    server_url: str = "http://localhost:8080/v1"
    api_key: str | None = None
    server_concurrency: int = 8       # in-flight requests (match server slots)
    request_timeout: float = 600.0    # seconds per request (long generations)
    # Extra JSON merged into every request body (server-specific knobs,
    # e.g. {chat_template_kwargs: {enable_thinking: false}} or top_k)
    server_extra_body: dict | None = None

    # vLLM / hardware
    gpu_memory_utilization: float = 0.95
    max_model_len: int | None = 4096  # None = model default
    enforce_eager: bool = True
    gpu_ids: list[int] | None = None  # None = default GPU, [3] = pin, [0,1] = parallel workers

    # Quantization: null, a single method for all models ("bitsandbytes",
    # "awq", "gptq", "fp8"), or a {model_tag: method} mapping.
    quantization: str | dict[str, str | None] | None = None

    # Output/input directories; relative paths resolve against the
    # current working directory.
    results_dir: str = "results"   # ledger DB + resolved config YAMLs
    logs_dir: str = "logs"         # tee'd stdout/stderr logs
    data_dir: str = "data"         # optional local few-shot files

    # The ledger database file all runs append to.
    # None = <results_dir>/ledger.sqlite3
    db_path: str | None = None

    # LLM answer verification (post-run pass over the results DB).
    # Set verifier_model (e.g. "opencompass/CompassVerifier-7B") to enable.
    # Mode "fallback" re-judges only string-match failures (sample is correct
    # if either pipeline accepts it); "all" lets the verifier verdict decide.
    verifier_model: str | None = None
    verifier_mode: str = "fallback"
    verifier_max_model_len: int = 16384

    def quantization_for(self, model_tag: str) -> str | None:
        """Resolve the quantization method for one model."""
        if isinstance(self.quantization, dict):
            return self.quantization.get(model_tag)
        return self.quantization

    def to_dict(self) -> dict:
        """Plain-data dict representation (tasks as {name, fewshot} mappings)."""
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        data["tasks"] = [{"name": name, "fewshot": k} for name, k in self.tasks]
        return data

    def to_yaml(self) -> str:
        """Serialize the resolved config as YAML (round-trips through load)."""
        return yaml.safe_dump(self.to_dict(), sort_keys=False, default_flow_style=False)


_VALID_KEYS = {f.name for f in fields(RunConfig)}


# ============================================================
# Model list expansion
# ============================================================

def expand_verl_checkpoints(run_dir: str) -> list[str]:
    """Expand a veRL run directory into HF model paths, one per checkpoint.

    Finds all global_step_* checkpoints and returns paths to their huggingface/
    dirs. Handles both GRPO (actor/huggingface/) and SFT (huggingface/) layouts.
    """
    run_path = Path(run_dir)
    if not run_path.exists():
        print(f"[WARN] veRL run dir not found: {run_dir}")
        return []
    step_dirs = sorted(
        [d for d in run_path.iterdir() if d.is_dir() and d.name.startswith("global_step_")],
        key=lambda d: int(d.name.split("_")[-1]),
    )
    models = []
    for step_dir in step_dirs:
        # GRPO layout: global_step_XXX/actor/huggingface/
        hf_path = step_dir / "actor" / "huggingface"
        if not hf_path.exists():
            # SFT layout: global_step_XXX/huggingface/
            hf_path = step_dir / "huggingface"
        if hf_path.exists() and (hf_path / "config.json").exists():
            models.append(str(hf_path))
    if not models:
        print(f"[WARN] No checkpoints found in veRL run dir: {run_dir}")
    return models


def expand_models(models: list) -> list[str]:
    """Expand model entries: 'verl:<run_dir>' becomes one entry per checkpoint."""
    expanded: list[str] = []
    for entry in models:
        entry = str(entry)
        if entry.startswith("verl:"):
            expanded.extend(expand_verl_checkpoints(entry[len("verl:"):]))
        else:
            expanded.append(entry)
    return expanded


# ============================================================
# Task list parsing
# ============================================================

def parse_task_entry(entry) -> list[tuple[str, int | None]]:
    """Parse one task entry into (name, fewshot_k) pairs.

    Accepted forms (YAML and CLI):
        "gsm8k_main"                         -> [(gsm8k_main, None)]  (task default k)
        "gsm8k_main:4"                       -> [(gsm8k_main, 4)]
        "gsm8k_main:0,4,8"                   -> ladder, 3 pairs
        {name: gsm8k_main}                   -> [(gsm8k_main, None)]
        {name: gsm8k_main, fewshot: 4}       -> [(gsm8k_main, 4)]
        {name: gsm8k_main, fewshot: [0, 4]}  -> ladder, 2 pairs
    """
    if isinstance(entry, str):
        name, sep, k_part = entry.partition(":")
        name = name.strip()
        if not name:
            raise ValueError(f"Invalid task entry: {entry!r}")
        if not sep:
            return [(name, None)]
        try:
            ks = [int(k) for k in k_part.split(",")]
        except ValueError:
            raise ValueError(
                f"Invalid fewshot spec in task entry {entry!r} (expected NAME[:K[,K...]])"
            ) from None
        return [(name, k) for k in ks]

    if isinstance(entry, dict):
        unknown = set(entry) - {"name", "fewshot"}
        if unknown:
            raise ValueError(f"Unknown keys {sorted(unknown)} in task entry {entry!r}")
        name = entry.get("name")
        if not name or not isinstance(name, str):
            raise ValueError(f"Task entry missing 'name': {entry!r}")
        fewshot = entry.get("fewshot")
        if fewshot is None:
            return [(name, None)]
        if isinstance(fewshot, list):
            return [(name, int(k)) for k in fewshot]
        return [(name, int(fewshot))]

    raise ValueError(f"Task entry must be a string or mapping, got: {entry!r}")


def parse_tasks(entries: list) -> list[tuple[str, int | None]]:
    """Parse and expand a list of task entries (fewshot ladders included)."""
    tasks: list[tuple[str, int | None]] = []
    for entry in entries:
        tasks.extend(parse_task_entry(entry))
    return tasks


# ============================================================
# YAML loading
# ============================================================

def load_yaml_config(path: Path) -> dict:
    """Load and validate a YAML config file into a plain dict.

    Rejects unknown keys so typos fail loudly instead of silently using defaults.
    """
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: config must be a YAML mapping, got {type(data).__name__}")

    unknown = set(data) - _VALID_KEYS
    if unknown:
        raise ValueError(
            f"{path}: unknown config keys {sorted(unknown)}. "
            f"Valid keys: {sorted(_VALID_KEYS)}"
        )
    return data


# ============================================================
# CLI
# ============================================================

def _int_or_none(value: str) -> int | None:
    """Parse an int CLI value where 0 or negative means None (= no limit)."""
    n = int(value)
    return n if n > 0 else None


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser: config file, per-field overrides, internal flags."""
    p = argparse.ArgumentParser(
        prog="run_bench",
        description="LLM benchmark runner. Settings come from a YAML config "
                    "file; any flag given here overrides the file.",
    )
    p.add_argument("-c", "--config", type=Path, default=None, metavar="YAML",
                   help=f"config file (default: ./{DEFAULT_CONFIG_FILE} if present)")
    p.add_argument("--model", action="append", dest="models", metavar="MODEL",
                   help="model HF id or local path; repeatable (replaces YAML models). "
                        "Use verl:<run_dir> to expand veRL checkpoints")
    p.add_argument("--task", action="append", dest="tasks", metavar="NAME[:K[,K...]]",
                   help="task with optional fewshot k or ladder, e.g. gsm8k_main:0,4,8; "
                        "repeatable (replaces YAML tasks)")
    p.add_argument("--max-examples", type=_int_or_none, default=None, metavar="N",
                   help="limit examples per task (0 = all)")
    p.add_argument("--batch-size", type=_int_or_none, default=None, metavar="N",
                   help="inference batch size (0 = single batch)")
    p.add_argument("--apply-chat-template", action=argparse.BooleanOptionalAction,
                   default=None, help="wrap prompts in the model's chat template")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--pass-k", type=int, default=None,
                   help="responses per sample; correct if ANY matches")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--gpu-memory-utilization", type=float, default=None)
    p.add_argument("--max-model-len", type=_int_or_none, default=None, metavar="N",
                   help="max context length (0 = model default)")
    p.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=None,
                   help="disable CUDA graphs to save memory")
    p.add_argument("--gpu-ids", type=str, default=None, metavar="I[,I...]",
                   help="GPUs to use, e.g. 3 (pin) or 0,1,2 (parallel workers)")
    p.add_argument("--quantization", type=str, default=None, metavar="METHOD",
                   help="quantization for all models (bitsandbytes/awq/gptq/fp8; "
                        "'none' to force model default; per-model maps are YAML-only)")
    p.add_argument("--results-dir", type=str, default=None, metavar="DIR",
                   help="directory for result DBs and resolved configs (default: ./results)")
    p.add_argument("--logs-dir", type=str, default=None, metavar="DIR",
                   help="directory for run logs (default: ./logs)")
    p.add_argument("--data-dir", type=str, default=None, metavar="DIR",
                   help="directory with optional local few-shot files (default: ./data)")
    p.add_argument("--db-path", type=str, default=None, metavar="FILE",
                   help="ledger database file (default: <results-dir>/ledger.sqlite3)")
    p.add_argument("--backend", type=str, default=None,
                   choices=["vllm", "hf", "server", "sglang"],
                   help="inference backend (default: vllm)")
    p.add_argument("--server-url", type=str, default=None, metavar="URL",
                   help="OpenAI-compatible endpoint for --backend server "
                        "(default: http://localhost:8080/v1)")
    p.add_argument("--api-key", type=str, default=None,
                   help="bearer token for --backend server, if the endpoint needs one")
    p.add_argument("--server-concurrency", type=int, default=None, metavar="N",
                   help="concurrent requests for --backend server (default: 8)")
    p.add_argument("--verifier-model", type=str, default=None, metavar="MODEL",
                   help="LLM verifier for a post-run verification pass "
                        "(e.g. opencompass/CompassVerifier-7B)")
    p.add_argument("--verifier-mode", type=str, default=None,
                   choices=["fallback", "all"],
                   help="fallback: re-judge only string-match failures; "
                        "all: verifier verdict decides")
    p.add_argument("--verifier-max-model-len", type=int, default=None, metavar="N",
                   help="context length for the verifier model (default: 16384)")
    # Internal flags for multi-GPU worker processes (not user-facing)
    p.add_argument("--shard", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument("--run-name", type=str, default=None, help=argparse.SUPPRESS)
    return p


def config_from_resolved_yaml(text: str) -> RunConfig:
    """Rebuild a RunConfig from a resolved config (the runs.config_yaml
    column, or `lm-eval-ledger config` output)."""
    data = yaml.safe_load(text) or {}
    unknown = set(data) - _VALID_KEYS
    if unknown:
        raise ValueError(f"Unknown config keys in resolved YAML: {sorted(unknown)}")
    data["models"] = expand_models(data.get("models") or [])
    data["tasks"] = parse_tasks(data.get("tasks") or [])
    return RunConfig(**data)


def resolve_config(args: argparse.Namespace) -> RunConfig:
    """Merge defaults, YAML file, and CLI flags into a validated RunConfig."""
    # ---------- YAML layer ----------
    config_path = args.config
    if config_path is None:
        default_path = Path(DEFAULT_CONFIG_FILE)
        if default_path.exists():
            config_path = default_path
    data: dict = {}
    if config_path is not None:
        if not Path(config_path).exists():
            raise ValueError(f"Config file not found: {config_path}")
        print(f"[INFO] Loading config from {config_path}")
        data = load_yaml_config(Path(config_path))

    # ---------- CLI override layer (only flags the user actually passed) ----------
    for key in ("max_examples", "batch_size", "apply_chat_template", "temperature",
                "top_p", "max_tokens", "pass_k", "seed", "gpu_memory_utilization",
                "max_model_len", "enforce_eager", "results_dir", "logs_dir", "data_dir",
                "db_path", "verifier_model", "verifier_mode", "verifier_max_model_len",
                "backend", "server_url", "api_key", "server_concurrency"):
        value = getattr(args, key)
        if value is not None:
            data[key] = value
    if args.models is not None:
        data["models"] = args.models
    if args.tasks is not None:
        data["tasks"] = args.tasks
    if args.gpu_ids is not None:
        data["gpu_ids"] = [int(g) for g in args.gpu_ids.split(",")]
    if args.quantization is not None:
        data["quantization"] = None if args.quantization.lower() == "none" else args.quantization

    # ---------- parse structured fields ----------
    data["models"] = expand_models(data.get("models") or [])
    data["tasks"] = parse_tasks(data.get("tasks") or [])

    cfg = RunConfig(**data)
    # The config file byte-for-byte as written, stored in the ledger next to
    # the resolved form. Plain attribute (not a dataclass field), so it stays
    # out of to_dict()/to_yaml() and the resolved-config round-trip.
    cfg.source_yaml_text = (
        Path(config_path).read_text() if config_path is not None else None
    )

    # ---------- validation ----------
    if not cfg.models:
        raise ValueError(
            "No models configured. Add a 'models:' list to the config file "
            f"(default: ./{DEFAULT_CONFIG_FILE}) or pass --model."
        )
    if cfg.pass_k < 1:
        raise ValueError(f"pass_k must be >= 1, got {cfg.pass_k}")
    if cfg.pass_k > 1 and cfg.temperature <= 0:
        print("[WARN] pass_k > 1 with temperature 0 will generate identical responses")
    if cfg.gpu_ids is not None and (
        not isinstance(cfg.gpu_ids, list) or not all(isinstance(g, int) for g in cfg.gpu_ids)
    ):
        raise ValueError(f"gpu_ids must be a list of ints or null, got {cfg.gpu_ids!r}")
    if cfg.verifier_mode not in ("fallback", "all"):
        raise ValueError(
            f"verifier_mode must be 'fallback' or 'all', got {cfg.verifier_mode!r}"
        )
    if cfg.backend not in ("vllm", "hf", "server", "sglang"):
        raise ValueError(
            f"backend must be one of vllm/hf/server/sglang, got {cfg.backend!r}"
        )
    if cfg.backend == "server" and cfg.gpu_ids and len(cfg.gpu_ids) > 1:
        raise ValueError(
            "backend 'server' does not use local GPUs; multi-GPU worker mode "
            "(gpu_ids with 2+ entries) is not applicable"
        )

    return cfg
