# preflight.py
"""Launch-time context-budget checks, in the spirit of the thinking menu:
probe what the harness can discover, and catch configurations that are
guaranteed (or likely) to waste the run before any GPU time is spent.

A generation slot must hold prompt + max_tokens, so the invariant is
    max_tokens + prompt headroom <= context window
where "context window" is max_model_len for in-process backends and the
server's per-slot n_ctx (total -c divided by -np) for backend: server.

Interactive launches get a prompt with a suggested fix; non-interactive
launches fail hard on certain-failure conditions (zero prompt budget)
and print warnings otherwise.
"""
from __future__ import annotations

import re
import sys

# Prompts below this many tokens exist in no registered task; a slot
# leaving less than this for prompts is treated as misconfigured.
MIN_PROMPT_HEADROOM = 1024
# Below this, most few-shot / LiveCodeBench prompts will be skipped.
COMFORTABLE_HEADROOM = 2048


def _tty() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _ask(question: str, default_yes: bool = True) -> bool:
    suffix = "[Y/n]" if default_yes else "[y/N]"
    raw = input(f"{question} {suffix}: ").strip().lower()
    if not raw:
        return default_yes
    return raw.startswith("y")


def fetch_server_slots(server_url: str) -> tuple[int, int] | None:
    """(total_slots, n_ctx_per_slot) from a llama.cpp-style /props, or
    None when the endpoint does not expose it (other server kinds)."""
    try:
        import httpx
        base = re.sub(r"/v1/?$", "", server_url.rstrip("/"))
        d = httpx.get(f"{base}/props", timeout=10).json()
        slots = d.get("total_slots")
        n_ctx = d.get("default_generation_settings", {}).get("n_ctx")
        if slots and n_ctx:
            return int(slots), int(n_ctx)
    except Exception:
        pass
    return None


def _check_window(cfg, label: str, window: int, source: str,
                  fix_hint: str) -> None:
    """Enforce max_tokens + headroom <= window, interactively or hard."""
    headroom = window - cfg.max_tokens
    if headroom >= COMFORTABLE_HEADROOM:
        return
    if headroom < MIN_PROMPT_HEADROOM:
        msg = (f"[PREFLIGHT] {label}: max_tokens {cfg.max_tokens} leaves "
               f"{max(headroom, 0)} prompt tokens in the {window}-token "
               f"window ({source}). Prompts will be skipped or rejected.")
        suggested = max(window - COMFORTABLE_HEADROOM, 256)
        if _tty():
            print(msg)
            if _ask(f"[PREFLIGHT] Cap max_tokens to {suggested} for this run?"):
                cfg.max_tokens = suggested
                print(f"[PREFLIGHT] max_tokens = {suggested} for this run. "
                      f"To make it permanent, set max_tokens: {suggested} "
                      f"in your config{fix_hint}.")
                return
            if _ask("[PREFLIGHT] Launch anyway (samples will fail)?",
                    default_yes=False):
                return
            raise SystemExit(2)
        raise ValueError(msg + fix_hint)
    print(f"[PREFLIGHT] {label}: only {headroom} prompt tokens fit beside "
          f"max_tokens {cfg.max_tokens} in the {window}-token window "
          f"({source}); long prompts (few-shot, LiveCodeBench) may be "
          f"skipped or rejected.")


def preflight_context(cfg) -> None:
    """Context-budget checks for every model entry's effective config."""
    from dataclasses import replace as _dc_replace

    from .config import model_spec

    seen_servers: set[str] = set()
    for entry in cfg.models:
        name, overrides = model_spec(entry)
        eff = _dc_replace(cfg, **overrides) if overrides else cfg

        if eff.backend == "server":
            if eff.server_url in seen_servers:
                continue
            seen_servers.add(eff.server_url)
            probed = fetch_server_slots(eff.server_url)
            if probed is None:
                continue  # not a llama.cpp-style server; nothing to check
            slots, n_ctx = probed
            _check_window(
                cfg, f"server {eff.server_url}", n_ctx,
                f"{slots} slots x {n_ctx} each",
                " or relaunch llama-server with a larger -c")
            if eff.server_concurrency > slots:
                print(f"[PREFLIGHT] server_concurrency "
                      f"{eff.server_concurrency} > the server's {slots} "
                      f"slots; extra requests just queue. ", end="")
                if _tty() and _ask(f"Set server_concurrency = {slots}?"):
                    cfg.server_concurrency = slots
                    print(f"[PREFLIGHT] server_concurrency = {slots}.")
                else:
                    print()
        else:
            if eff.max_model_len is None:
                continue  # model-default window; runner's skip guard covers it
            _check_window(
                cfg, f"{name}", eff.max_model_len,
                f"max_model_len {eff.max_model_len}",
                " or raise max_model_len")
