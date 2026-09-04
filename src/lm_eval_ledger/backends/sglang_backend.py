# backends/sglang_backend.py
"""SGLang backend: in-process batched inference via sglang.Engine.

An alternative high-throughput engine to vLLM. Supports generate and
logprob_token; logprob_seq is not implemented (use vllm or hf).
"""
from __future__ import annotations

import os

from .base import safe_on_result, Backend, GenResult


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
        if cfg.tensor_parallel_size > 1:
            kwargs["tp_size"] = cfg.tensor_parallel_size
        if quantization:
            kwargs["quantization"] = quantization
        # sglang has no per-request seed; the engine-level random_seed is
        # the only reproducibility handle it offers
        kwargs["random_seed"] = cfg.seed
        print("[INFO] sglang: no per-request seeds - engine random_seed set "
              f"to {cfg.seed}; pass_k>1 samples differ but a rerun is not "
              "guaranteed bit-identical")
        # sglang refuses to start multimodal models on torch 2.9.1 with
        # cuDNN < 9.15 (a Conv3d bug in the VISION encoder). torch pins that
        # cuDNN exactly, so the fix cannot ship in our [sglang] extra. Text
        # eval never runs the vision tower; for modality: all it is a
        # performance bug, not a correctness one - warn, do not refuse.
        if "SGLANG_DISABLE_CUDNN_CHECK" not in os.environ:
            os.environ["SGLANG_DISABLE_CUDNN_CHECK"] = "1"
            if cfg.modality == "all":
                print("[WARN] sglang: images on torch 2.9.1 + cuDNN < 9.15 can be "
                      "slow/memory-hungry (Conv3d bug); fix with "
                      "`pip install nvidia-cudnn-cu12>=9.15`")
        # Blackwell (sm_100+): sglang's default flashinfer attention has no
        # kernels for hybrid linear-attention (GDN) models such as Qwen3.5 /
        # Qwen3-Next and aborts in the scheduler child; triton does.
        if "attention_backend" not in kwargs and _needs_triton_attention(model):
            kwargs["attention_backend"] = "triton"
            print("[INFO] sglang: Blackwell GPU + hybrid linear-attention model "
                  "-> attention_backend=triton (flashinfer has no GDN kernels)")
        self.engine = sgl.Engine(**kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model, trust_remote_code=True)
        self.template_kwargs = dict(cfg.chat_template_kwargs or {})

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
                 seed, batch_size, on_result=None):
        """One engine call by default; with batch_size set, chunked calls
        stream each finished chunk through on_result."""
        params = {
            "temperature": temperature, "top_p": top_p,
            "max_new_tokens": max_tokens,
        }
        if stop:
            params["stop"] = stop
        chunk = batch_size if batch_size and batch_size > 0 else len(prompts)
        results: list[list[GenResult]] = []
        for start in range(0, len(prompts), chunk):
            batch = prompts[start:start + chunk]
            # n > 1 via prompt repetition: universally supported, groups cleanly
            flat = [p for p in batch for _ in range(n)]
            outputs = self.engine.generate(flat, params)
            for i in range(0, len(outputs), n):
                group = []
                for out in outputs[i:i + n]:
                    meta = out.get("meta_info", {})
                    finish, matched = self._finish(meta)
                    group.append(GenResult(text=out.get("text", ""),
                                           finish_reason=finish,
                                           stop_reason=matched,
                                           n_tokens=meta.get("completion_tokens")))
                safe_on_result(on_result, start + i // n, group)
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
                messages, tokenize=False, add_generation_prompt=True,
                **self.template_kwargs)
        except Exception:
            return None

    def count_tokens(self, text):
        return len(self.tokenizer.encode(text))


def _needs_triton_attention(model: str) -> bool:
    """True on a Blackwell-class GPU when the model mixes linear attention."""
    try:
        import torch
        if torch.cuda.get_device_capability()[0] < 10:
            return False
    except Exception:
        return False
    try:
        import json
        from pathlib import Path
        local = Path(model) / "config.json"
        if local.is_file():
            path = local
        else:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(model, "config.json")
        cfg = json.loads(Path(path).read_text())
        text = cfg.get("text_config", cfg)
        return "linear_attention" in set(text.get("layer_types") or [])
    except Exception:
        return False
