# servermodel.py
"""Interactive server model-name reconciliation.

With backend: server, the models: entry must be the id the endpoint
serves (llama-server derives it from the GGUF path, ollama from the
tag, ...), and the run's model_tag in the ledger comes from that name.
When a run is interactive (TTY) and a configured name is not in the
server's /models list, this offers the served ids as a menu and
rewrites the entry for this run, printing the YAML to make it
permanent. Non-interactive runs are never prompted; the server backend
warns and sends the configured name as-is.
"""
from __future__ import annotations

import sys


def fetch_served_models(cfg) -> list[str] | None:
    """Model ids the endpoint reports at /models, or None if unreachable."""
    try:
        import httpx
        headers = {"Authorization": f"Bearer {cfg.api_key}"} if cfg.api_key else {}
        r = httpx.get(f"{cfg.server_url.rstrip('/')}/models", headers=headers,
                      timeout=10)
        r.raise_for_status()
        return [m.get("id") for m in r.json().get("data", []) if m.get("id")]
    except Exception:
        return None


def maybe_prompt_server_model(cfg) -> None:
    """Reconcile server-backend model names with what the endpoint serves.

    Asks once PER MODEL entry whose effective backend is server and whose
    name is not in that server's /models list. The chosen id replaces
    the entry's name (overrides preserved). No-ops when non-interactive
    or when the server is unreachable (the backend reports that itself).
    """
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return
    from dataclasses import replace as _dc_replace

    from .config import model_spec

    served_cache: dict[str, list[str] | None] = {}
    renamed: list[tuple[str, str]] = []
    for idx, entry in enumerate(cfg.models):
        name, overrides = model_spec(entry)
        eff = _dc_replace(cfg, **overrides) if overrides else cfg
        if eff.backend != "server":
            continue
        if eff.server_url not in served_cache:
            served_cache[eff.server_url] = fetch_served_models(eff)
        served = served_cache[eff.server_url]
        if not served or name in served:
            continue

        print(f"\n[SERVER] {name!r} is not served at {eff.server_url}")
        print(f"[SERVER] the endpoint reports {len(served)} model(s) - "
              f"pick the one this entry means:")
        for i, sid in enumerate(served, 1):
            print(f"  {i}) {sid}")
        keep = len(served) + 1
        print(f"  {keep}) keep {name!r} (send it anyway)")
        default = "1" if len(served) == 1 else str(keep)
        while True:
            raw = input(f"Select 1-{keep} [{default}]: ").strip() or default
            if raw.isdigit() and 1 <= int(raw) <= keep:
                break
            print("Invalid choice.")
        choice = int(raw)
        if choice == keep:
            continue
        new = served[choice - 1]
        cfg.models[idx] = {"name": new, **overrides} if overrides else new
        renamed.append((name, new))

    if not renamed:
        return
    print("[SERVER] Applied. To make this permanent, use these names in "
          "your config's models: list:")
    for old, new in renamed:
        print(f"  - {new}    # was {old}")
