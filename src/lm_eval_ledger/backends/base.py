# backends/base.py
"""Inference backend interface.

A backend owns everything engine-specific: loading/unloading a model,
generation, the two logprob evaluation primitives, chat templating, and
token counting. The runner is backend-agnostic and consults
`capabilities` to fail loudly on eval modes an engine cannot provide.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GenResult:
    """One generated response."""
    text: str
    finish_reason: str = ""   # "stop" | "length" | "error" | ...
    stop_reason: object = None  # engine-specific detail (e.g. matched stop string)
    n_tokens: int | None = None  # generated tokens, from the engine's own count


class Backend:
    """Interface; subclasses implement one inference engine."""

    name: str = "base"
    #: eval modes this backend supports: subset of
    #: {"generate", "logprob_token", "logprob_seq"}
    capabilities: frozenset = frozenset({"generate"})
    #: True if generation should receive chat `messages` instead of a
    #: template-rendered prompt string (server-side templating).
    prefers_messages: bool = False

    # ---- lifecycle ----

    def load(self, model: str, cfg, quantization: str | None = None) -> None:
        raise NotImplementedError

    def unload(self) -> None:
        pass

    # ---- generation ----

    def generate(self, prompts: list[str], *, temperature: float, top_p: float,
                 max_tokens: int, stop: list[str] | None, n: int, seed: int,
                 batch_size: int | None,
                 on_result=None) -> list[list[GenResult]]:
        """Return n GenResults per prompt, aligned with `prompts`.

        on_result(index, [GenResult, ...]) is an optional streaming hook:
        backends that generate incrementally call it as each prompt
        completes (possibly from worker threads). Backends may ignore it;
        the runner writes any un-streamed results after the call returns.
        """
        raise NotImplementedError

    def chat_generate(self, messages_list: list[list[dict]], *, temperature: float,
                      top_p: float, max_tokens: int, stop: list[str] | None,
                      n: int, seed: int, batch_size: int | None,
                      on_result=None) -> list[list[GenResult]]:
        """Chat-endpoint variant (only for prefers_messages backends)."""
        raise NotImplementedError

    # ---- logprob primitives ----

    def first_token_logprobs(self, prompts: list[str], labels: list[str],
                             *, seed: int = 0) -> list[dict[str, float]]:
        """Per prompt: {label: logprob of the label as the next token}."""
        raise NotImplementedError(
            f"backend '{self.name}' does not support logprob_token evaluation")

    def score_completions(self, base_prompts: list[str],
                          choice_texts: list[list[str]],
                          *, seed: int = 0) -> list[list[float]]:
        """Per prompt: length-normalized log-likelihood of each choice text."""
        raise NotImplementedError(
            f"backend '{self.name}' does not support logprob_seq evaluation")

    # ---- text utilities ----

    #: extra chat-template kwargs (set by load() from cfg.chat_template_kwargs)
    template_kwargs: dict = {}

    def apply_chat_template(self, messages: list[dict]) -> str | None:
        """Render messages with the model's chat template, or None if the
        backend cannot template client-side."""
        return None

    def count_tokens(self, text: str) -> int | None:
        """Token count for context budgeting, or None if unknown."""
        return None
