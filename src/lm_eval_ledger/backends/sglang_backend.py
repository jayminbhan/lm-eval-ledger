# backends/sglang_backend.py
"""SGLang backend: in-process batched inference via sglang.Engine.

An alternative high-throughput engine to vLLM. Supports generate and
logprob_token; logprob_seq is not implemented (use vllm or hf).
"""
from __future__ import annotations

from .base import Backend, GenResult


class SglangBackend(Backend):
    name = "sglang"
    capabilities = frozenset({"generate", "logprob_token"})

    def __init__(self):
        self.engine = None
        self.tokenizer = None

    # ---- lifecycle ----

    def load(self, model: str, cfg, quantization: str | None = None) -> None:
        import sglang as sgl
        from transformers import AutoTokenizer

        kwargs = {
            "model_path": model,
            "trust_remote_code": True,
            "mem_fraction_static": cfg.gpu_memory_utilization,
        }
        if cfg.max_model_len is not None:
            kwargs["context_length"] = cfg.max_model_len
        if quantization:
            kwargs["quantization"] = quantization
        self.engine = sgl.Engine(**kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model, trust_remote_code=True)

    def unload(self) -> None:
        import gc
        if self.engine is not None:
            self.engine.shutdown()
        self.engine = None
        self.tokenizer = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    # ---- generation ----

    @staticmethod
    def _finish(meta: dict) -> tuple[str, object]:
        fr = meta.get("finish_reason") or {}
        if isinstance(fr, dict):
            return fr.get("type", ""), fr.get("matched")
        return str(fr), None

    def generate(self, prompts, *, temperature, top_p, max_tokens, stop, n,
                 seed, batch_size):
        # n > 1 via prompt repetition: universally supported, groups cleanly
        flat = [p for p in prompts for _ in range(n)]
        params = {
            "temperature": temperature, "top_p": top_p,
            "max_new_tokens": max_tokens,
        }
        if stop:
            params["stop"] = stop
        outputs = self.engine.generate(flat, params)
        results = []
        for i in range(0, len(outputs), n):
            group = []
            for out in outputs[i:i + n]:
                finish, matched = self._finish(out.get("meta_info", {}))
                group.append(GenResult(text=out.get("text", ""),
                                       finish_reason=finish,
                                       stop_reason=matched))
            results.append(group)
        return results

    # ---- logprob primitives ----

    def first_token_logprobs(self, prompts, labels, *, seed=0):
        label_ids = {}
        for label in labels:
            ids = self.tokenizer.encode(f" {label}", add_special_tokens=False)
            label_ids[label] = ids[-1]
        print(f"[INFO] Choice token IDs: {label_ids}")

        params = {"temperature": 0, "max_new_tokens": 1}
        outputs = self.engine.generate(
            list(prompts), params, return_logprob=True, top_logprobs_num=100)

        results = []
        for out in outputs:
            meta = out.get("meta_info", {})
            # [(logprob, token_id, token_text), ...] for the first output token
            top = (meta.get("output_top_logprobs") or [[]])[0] or []
            by_id = {}
            for entry in top:
                try:
                    lp, tid = float(entry[0]), int(entry[1])
                except (TypeError, ValueError, IndexError):
                    continue
                by_id[tid] = lp
            if not by_id:
                raise RuntimeError(
                    "sglang returned no top logprobs; cannot run "
                    "logprob_token - use the vllm or hf backend")
            results.append({label: by_id.get(tid, float("-inf"))
                            for label, tid in label_ids.items()})
        return results

    # ---- text utilities ----

    def apply_chat_template(self, messages):
        if not hasattr(self.tokenizer, "apply_chat_template"):
            return None
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            return None

    def count_tokens(self, text):
        return len(self.tokenizer.encode(text))
