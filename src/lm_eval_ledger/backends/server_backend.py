# backends/server_backend.py
"""OpenAI-compatible HTTP server backend.

Talks to anything speaking the OpenAI API: llama.cpp's llama-server,
ollama, LM Studio, `vllm serve`, SGLang's server, hosted endpoints.
The model entry in the config is the model name the server expects.

Reliability: preflight health check at load, bounded concurrency,
retries with backoff on connection errors/5xx, and per-request failures
surface as GenResult(finish_reason="error") so a run records partial
failure instead of dying.

Capabilities: generate everywhere; logprob_token where the server
returns completions logprobs (llama-server does, many others do not);
logprob_seq is not expressible over this API.
"""
from __future__ import annotations

import concurrent.futures
import time

import httpx

from .base import Backend, GenResult

_RETRIES = 3


class ServerBackend(Backend):
    name = "server"
    capabilities = frozenset({"generate", "logprob_token"})
    prefers_messages = True  # chat templating happens server-side

    def __init__(self):
        self.client: httpx.Client | None = None
        self.base_url = ""
        self.model = ""
        self.concurrency = 8

    # ---- lifecycle ----

    def load(self, model: str, cfg, quantization: str | None = None) -> None:
        if quantization:
            print(f"[WARN] server backend ignores quantization={quantization!r} "
                  f"(quantization is the server's concern)")
        self.model = model
        self.base_url = cfg.server_url.rstrip("/")
        self.concurrency = cfg.server_concurrency
        headers = {}
        if cfg.api_key:
            headers["Authorization"] = f"Bearer {cfg.api_key}"
        self.client = httpx.Client(base_url=self.base_url, headers=headers,
                                   timeout=cfg.request_timeout)
        # Preflight: fail in second one, not sample one
        r = self.client.get("/models")
        r.raise_for_status()
        served = [m.get("id") for m in r.json().get("data", [])]
        print(f"[INFO] Server at {self.base_url} is up; serving: {served}")
        if served and model not in served:
            print(f"[WARN] model {model!r} not in the server's model list; "
                  f"requests will use it anyway")

    def unload(self) -> None:
        if self.client:
            self.client.close()
        self.client = None

    # ---- request plumbing ----

    def _post(self, path: str, payload: dict) -> dict:
        last = None
        for attempt in range(_RETRIES):
            try:
                r = self.client.post(path, json=payload)
                if r.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"server {r.status_code}", request=r.request, response=r)
                r.raise_for_status()
                return r.json()
            except (httpx.TransportError, httpx.HTTPStatusError) as e:
                last = e
                if attempt < _RETRIES - 1:
                    time.sleep(1.5 * (attempt + 1))
        raise last

    def _map(self, fn, items):
        """Bounded-concurrency map preserving order."""
        with concurrent.futures.ThreadPoolExecutor(self.concurrency) as pool:
            return list(pool.map(fn, items))

    # ---- generation ----

    def _completion_results(self, payload_for, items, n, parse):
        def one(item):
            group = []
            for k in range(n):
                try:
                    data = self._post(*payload_for(item, k))
                    group.append(parse(data))
                except Exception as e:
                    group.append(GenResult(text="", finish_reason="error",
                                           stop_reason=f"{type(e).__name__}: {e}"))
            return group
        return self._map(one, items)

    def generate(self, prompts, *, temperature, top_p, max_tokens, stop, n,
                 seed, batch_size):
        def payload_for(prompt, k):
            return "/completions", {
                "model": self.model, "prompt": prompt,
                "temperature": temperature, "top_p": top_p,
                "max_tokens": max_tokens, "stop": stop or None,
                "seed": seed + k,
            }

        def parse(data):
            choice = data["choices"][0]
            return GenResult(text=choice.get("text", ""),
                             finish_reason=choice.get("finish_reason") or "",
                             stop_reason=None)
        return self._completion_results(payload_for, prompts, n, parse)

    def chat_generate(self, messages_list, *, temperature, top_p, max_tokens,
                      stop, n, seed, batch_size):
        def payload_for(messages, k):
            return "/chat/completions", {
                "model": self.model, "messages": messages,
                "temperature": temperature, "top_p": top_p,
                "max_tokens": max_tokens, "stop": stop or None,
                "seed": seed + k,
            }

        def parse(data):
            choice = data["choices"][0]
            return GenResult(
                text=(choice.get("message") or {}).get("content") or "",
                finish_reason=choice.get("finish_reason") or "",
                stop_reason=None)
        return self._completion_results(payload_for, messages_list, n, parse)

    # ---- logprob primitives ----

    def first_token_logprobs(self, prompts, labels, *, seed=0):
        """Best-effort via completions top-logprobs; matches label token
        text (with or without a leading space)."""
        def one(prompt):
            data = self._post("/completions", {
                "model": self.model, "prompt": prompt,
                "max_tokens": 1, "temperature": 0, "logprobs": 20,
            })
            lp = (data["choices"][0].get("logprobs") or {})
            top = (lp.get("top_logprobs") or [{}])
            top0 = top[0] if top else {}
            if not top0:
                raise RuntimeError(
                    "server returned no logprobs; this server does not "
                    "support logprob_token evaluation - use the vllm or hf "
                    "backend for this task")
            out = {}
            for label in labels:
                out[label] = max(
                    (v for t, v in top0.items() if t.strip() == label),
                    default=float("-inf"))
            return out
        return self._map(one, prompts)
