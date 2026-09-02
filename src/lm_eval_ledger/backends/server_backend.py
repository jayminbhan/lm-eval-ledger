# backends/server_backend.py
"""OpenAI-compatible HTTP server backend.

Talks to anything speaking the OpenAI API: llama.cpp's llama-server,
ollama, LM Studio, `vllm serve`, SGLang's server, hosted endpoints.
The model entry in the config is the model name the server expects.

Reliability: preflight health check at load, bounded concurrency,
retries with backoff on connection errors/5xx, and per-request failures
surface as GenResult(finish_reason="error") so a run records partial
failure instead of dying.

Capabilities: generate everywhere; logprob_token on llama.cpp
(llama-server returns completions logprobs; most other endpoints do
not, and fail with a clear error on the first request);
logprob_seq is not expressible over this API.
"""
from __future__ import annotations

import concurrent.futures
import threading
import time

import httpx

from .base import Backend, GenResult

_RETRIES = 3


class ServerBackend(Backend):
    name = "server"
    # "vision": image content parts pass through to the chat endpoint;
    # whether they work is the served model's capability (the server
    # errors per-request otherwise).
    capabilities = frozenset({"generate", "logprob_token", "vision"})
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
        self.extra_body = dict(cfg.server_extra_body or {})
        self.template_kwargs = dict(cfg.chat_template_kwargs or {})
        if self.extra_body:
            print(f"[INFO] server extra body: {self.extra_body}")
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
                if r.status_code >= 400:
                    # carry the server's own error message for diagnosis
                    raise httpx.HTTPStatusError(
                        f"server {r.status_code}: {r.text[:200]}",
                        request=r.request, response=r)
                return r.json()
            except (httpx.TransportError, httpx.HTTPStatusError) as e:
                last = e
                if attempt < _RETRIES - 1:
                    time.sleep(1.5 * (attempt + 1))
        raise last

    def _map(self, fn, items, desc="requests"):
        """Bounded-concurrency map preserving order, with periodic progress
        (count, rate, ETA) so long batches are not silent."""
        total = len(items)
        state = {"done": 0, "last": time.time()}
        lock = threading.Lock()
        start = time.time()

        def wrapped(item):
            result = fn(item)
            with lock:
                state["done"] += 1
                now = time.time()
                if state["done"] == total or now - state["last"] >= 30:
                    state["last"] = now
                    rate = state["done"] / max(now - start, 1e-9)
                    eta = (total - state["done"]) / rate if rate > 0 else 0
                    print(f"[INFO] server {desc}: {state['done']}/{total} "
                          f"({rate * 60:.1f}/min, ETA "
                          f"{int(eta // 60)}m{int(eta % 60):02d}s)",
                          flush=True)
            return result

        with concurrent.futures.ThreadPoolExecutor(self.concurrency) as pool:
            return list(pool.map(wrapped, items))

    # ---- generation ----

    def _completion_results(self, payload_for, items, n, parse, on_result=None):
        def one(pair):
            idx, item = pair
            group = []
            for k in range(n):
                try:
                    data = self._post(*payload_for(item, k))
                    group.append(parse(data))
                except Exception as e:
                    group.append(GenResult(text="", finish_reason="error",
                                           stop_reason=f"{type(e).__name__}: {e}"))
            if on_result is not None:
                on_result(idx, group)
            return group
        return self._map(one, list(enumerate(items)), desc="generations")

    def generate(self, prompts, *, temperature, top_p, max_tokens, stop, n,
                 seed, batch_size, on_result=None):
        def payload_for(prompt, k):
            return "/completions", {
                "model": self.model, "prompt": prompt,
                "temperature": temperature, "top_p": top_p,
                "max_tokens": max_tokens, "stop": stop or None,
                "seed": seed + k, **self.extra_body,
            }

        def parse(data):
            choice = data["choices"][0]
            return GenResult(text=choice.get("text", ""),
                             finish_reason=choice.get("finish_reason") or "",
                             stop_reason=None)
        return self._completion_results(payload_for, prompts, n, parse,
                                        on_result=on_result)

    def chat_generate(self, messages_list, *, temperature, top_p, max_tokens,
                      stop, n, seed, batch_size, on_result=None):
        def payload_for(messages, k):
            return "/chat/completions", {
                "model": self.model, "messages": messages,
                "temperature": temperature, "top_p": top_p,
                "max_tokens": max_tokens, "stop": stop or None,
                "seed": seed + k, **self.extra_body,
                **({"chat_template_kwargs": self.template_kwargs}
                   if self.template_kwargs else {}),
            }

        def parse(data):
            choice = data["choices"][0]
            msg = choice.get("message") or {}
            content = msg.get("content") or ""
            # Reasoning models via llama-server (and others) return chain-of-
            # thought in a separate reasoning_content field; keep it so the
            # ledger records the full output and \boxed{} extraction can see
            # everything the model produced.
            reasoning = msg.get("reasoning_content") or ""
            text = (f"<think>\n{reasoning}\n</think>\n{content}"
                    if reasoning else content)
            return GenResult(
                text=text,
                finish_reason=choice.get("finish_reason") or "",
                stop_reason=None)
        return self._completion_results(payload_for, messages_list, n, parse,
                                        on_result=on_result)

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
        return self._map(one, prompts, desc="logprob requests")
