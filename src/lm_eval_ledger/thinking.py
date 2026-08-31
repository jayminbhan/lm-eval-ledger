# thinking.py
"""Interactive thinking-mode discovery.

When a run is interactive (TTY), uses a chat template, and the config
does not set chat_template_kwargs, this probes the model's chat template
for known thinking-mode knobs (enable_thinking, reasoning_effort, ...),
lets the user pick a mode, and prints the YAML snippet to make the
choice permanent. Non-interactive runs are never prompted.

Silence the prompt for a config permanently with:
    chat_template_kwargs: {}        # explicit template defaults
"""
from __future__ import annotations

import re
import sys

# knobs we know how to detect in chat templates
_BOOL_KNOBS = ("enable_thinking", "thinking")
_ENUM_KNOBS = ("reasoning_effort",)


def fetch_chat_template(cfg, model: str) -> str | None:
    """The model's chat template text, or None if unavailable."""
    if cfg.backend == "server":
        # llama.cpp exposes it at /props; other servers may not
        try:
            import httpx
            base = re.sub(r"/v1/?$", "", cfg.server_url.rstrip("/"))
            r = httpx.get(f"{base}/props", timeout=10)
            return r.json().get("chat_template") or None
        except Exception:
            return None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        return tok.chat_template
    except Exception:
        return None


def detect_thinking_knobs(template: str) -> dict:
    """Known thinking knobs referenced by the template.

    Returns {knob: {"type": "bool"} | {"type": "enum", "options": [...],
    "default": str | None}}.
    """
    knobs: dict = {}
    for k in _BOOL_KNOBS:
        if re.search(rf"\b{k}\b", template):
            # the template's own default, when statically visible:
            #   {k} if {k} is defined else True   |   {k} | default(true)
            m = re.search(
                rf"{k}\s+is\s+defined\s+else\s+(True|False|true|false)"
                rf"|{k}\s*\|\s*default\(\s*(True|False|true|false)", template)
            default = None
            if m:
                default = (m.group(1) or m.group(2)).lower() == "true"
            knobs[k] = {"type": "bool", "default": default}
    for k in _ENUM_KNOBS:
        if re.search(rf"\b{k}\b", template):
            options = sorted(set(
                re.findall(rf"{k}\w*\s*==\s*['\"](\w+)['\"]", template)))
            m = re.search(rf"{k}\s*\|\s*default\(\s*['\"](\w+)['\"]", template)
            default = m.group(1) if m else None
            if default and default not in options:
                options.append(default)
            knobs[k] = {"type": "enum", "options": options, "default": default}
    # "thinking" alone is too generic if enable_thinking already matched
    if "enable_thinking" in knobs:
        knobs.pop("thinking", None)
    return knobs


def _build_choices(knobs: dict) -> list[tuple[str, dict | None]]:
    """(label, kwargs) menu entries; kwargs None = send nothing."""
    bool_knob = next((k for k in _BOOL_KNOBS if k in knobs), None)
    bool_default = knobs[bool_knob].get("default") if bool_knob else None
    default_note = ("" if bool_default is None else
                    f" = thinking {'ON' if bool_default else 'OFF'}"
                    f" for this model")
    choices: list[tuple[str, dict | None]] = [
        (f"template default (send nothing{default_note})", None)]
    enum = knobs.get("reasoning_effort")
    if bool_knob:
        choices.append((f"thinking off  ({bool_knob}: false)",
                        {bool_knob: False}))
    if enum and enum["options"]:
        for opt in enum["options"]:
            kwargs = {"reasoning_effort": opt}
            if bool_knob:
                kwargs = {bool_knob: True, **kwargs}
            suffix = " [template default]" if opt == enum.get("default") else ""
            choices.append((f"thinking on, reasoning_effort: {opt}{suffix}",
                            kwargs))
    elif bool_knob:
        choices.append((f"thinking on  ({bool_knob}: true)", {bool_knob: True}))
    return choices


def _yaml_snippet(kwargs: dict) -> str:
    lines = ["chat_template_kwargs:"]
    for k, v in kwargs.items():
        lines.append(f"  {k}: {str(v).lower() if isinstance(v, bool) else v}")
    return "\n".join(lines)


def maybe_prompt_thinking_mode(cfg) -> None:
    """Interactively pick thinking modes when the config left them unset.

    Asks once PER MODEL whose entry has no chat_template_kwargs override
    (skipping models whose templates expose no knobs); each answer is
    stored as that model's per-model override, and a copy-paste YAML
    models: snippet is printed to make the choices permanent. No-ops
    when non-interactive, chat templating is off, or the global
    chat_template_kwargs is set ({} = explicit opt-out).
    """
    if not cfg.apply_chat_template or cfg.chat_template_kwargs is not None:
        return
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return
    from dataclasses import replace as _dc_replace

    from .config import model_spec

    chosen: list[tuple[str, dict | None]] = []
    for idx, entry in enumerate(cfg.models):
        name, overrides = model_spec(entry)
        if "chat_template_kwargs" in overrides:
            continue  # this model already chose in the config
        eff = (_dc_replace(cfg, **overrides) if overrides else cfg)
        template = fetch_chat_template(eff, name)
        if not template:
            continue
        knobs = detect_thinking_knobs(template)
        if not knobs:
            continue

        choices = _build_choices(knobs)
        print(f"\n[THINKING] {name}'s chat template supports thinking-mode "
              f"knobs: {', '.join(knobs)}")
        print("[THINKING] chat_template_kwargs is not set for this model - "
              "choose a mode for this run:")
        for i, (label, _) in enumerate(choices, 1):
            print(f"  {i}) {label}")
        while True:
            raw = input(f"Select 1-{len(choices)} [1]: ").strip() or "1"
            if raw.isdigit() and 1 <= int(raw) <= len(choices):
                break
            print("Invalid choice.")
        _, kwargs = choices[int(raw) - 1]
        chosen.append((name, kwargs))
        if kwargs is not None:
            cfg.models[idx] = {"name": name, **overrides,
                               "chat_template_kwargs": kwargs}

    if not chosen:
        return
    if all(kw is None for _, kw in chosen):
        print("[THINKING] Using template defaults. To silence this prompt, "
              "add to your config:\n  chat_template_kwargs: {}")
        cfg.chat_template_kwargs = {}
        return
    print("[THINKING] Applied. To make this permanent, use these models: "
          "entries in your config:")
    for name, kwargs in chosen:
        if kwargs is None:
            print(f"  - {name}")
        else:
            inner = ", ".join(
                f"{k}: {str(v).lower() if isinstance(v, bool) else v}"
                for k, v in kwargs.items())
            print(f"  - name: {name}\n    chat_template_kwargs: {{{inner}}}")
