# backends/hf_backend.py
"""HuggingFace transformers + accelerate backend.

The compatibility backend: supports every architecture transformers can
load (including ones vLLM cannot), full logprob control, device_map=auto
sharding. Much slower than vLLM - use for unsupported models or small
exact runs.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import Backend, GenResult

_DEFAULT_BATCH = 8


class HfBackend(Backend):
    name = "hf"
    capabilities = frozenset({"generate", "logprob_token", "logprob_seq"})

    def __init__(self):
        self.model = None
        self.tokenizer = None

    # ---- lifecycle ----

    def load(self, model: str, cfg, quantization: str | None = None) -> None:
        kwargs = {"torch_dtype": "auto", "device_map": "auto",
                  "trust_remote_code": True}
        if quantization == "bitsandbytes":
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        elif quantization:
            print(f"[WARN] hf backend ignores quantization={quantization!r} "
                  f"(only 'bitsandbytes' is supported)")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # decoder-only batching
        self.model = AutoModelForCausalLM.from_pretrained(model, **kwargs)
        self.model.eval()

    def unload(self) -> None:
        import gc
        del self.model
        self.model = None
        self.tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()

    # ---- generation ----

    def _truncate_at_stop(self, text: str, stop: list[str] | None):
        """Post-hoc stop strings (transformers has no per-string stop)."""
        if not stop:
            return text, None
        cut, matched = len(text), None
        for s in stop:
            idx = text.find(s)
            if idx != -1 and idx < cut:
                cut, matched = idx, s
        return (text[:cut], matched) if matched is not None else (text, None)

    def generate(self, prompts, *, temperature, top_p, max_tokens, stop, n,
                 seed, batch_size):
        torch.manual_seed(seed)
        do_sample = temperature > 0
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "num_return_sequences": n,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if do_sample:
            gen_kwargs.update(temperature=temperature, top_p=top_p)

        bs = batch_size or _DEFAULT_BATCH
        results: list[list[GenResult]] = []
        n_batches = (len(prompts) + bs - 1) // bs
        for i in range(0, len(prompts), bs):
            batch = prompts[i:i + bs]
            print(f"[INFO] hf generate batch {i // bs + 1}/{n_batches} "
                  f"({len(batch)} prompts)...")
            enc = self.tokenizer(batch, return_tensors="pt", padding=True,
                                 add_special_tokens=False).to(self.model.device)
            with torch.no_grad():
                out = self.model.generate(**enc, **gen_kwargs)
            new_tokens = out[:, enc["input_ids"].shape[1]:]
            for p_idx in range(len(batch)):
                group = []
                for k in range(n):
                    seq = new_tokens[p_idx * n + k]
                    hit_eos = bool((seq == self.tokenizer.eos_token_id).any())
                    text = self.tokenizer.decode(seq, skip_special_tokens=False)
                    # strip padding/eos tail noise from decoding
                    text = text.replace(self.tokenizer.pad_token or "", "")
                    text, matched = self._truncate_at_stop(text, stop)
                    if matched is not None:
                        finish = "stop"
                    elif hit_eos:
                        finish, matched = "stop", None
                    else:
                        finish = "length"
                    group.append(GenResult(text=text, finish_reason=finish,
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

        results = []
        bs = _DEFAULT_BATCH
        for i in range(0, len(prompts), bs):
            batch = prompts[i:i + bs]
            enc = self.tokenizer(batch, return_tensors="pt", padding=True,
                                 add_special_tokens=False).to(self.model.device)
            with torch.no_grad():
                logits = self.model(**enc).logits[:, -1, :]  # left padding
            logprobs = torch.log_softmax(logits.float(), dim=-1)
            for row in logprobs:
                results.append({label: float(row[tid])
                                for label, tid in label_ids.items()})
        return results

    def score_completions(self, base_prompts, choice_texts, *, seed=0):
        scores = []
        for base, choices in zip(base_prompts, choice_texts):
            base_ids = self.tokenizer.encode(base, add_special_tokens=False)
            row = []
            for choice in choices:
                full_ids = self.tokenizer.encode(base + " " + choice,
                                                 add_special_tokens=False)
                # divergence point (tokens can merge at the seam)
                n_ctx = 0
                for a, b in zip(base_ids, full_ids):
                    if a != b:
                        break
                    n_ctx += 1
                n_answer = max(len(full_ids) - n_ctx, 1)
                ids = torch.tensor([full_ids], device=self.model.device)
                with torch.no_grad():
                    logits = self.model(ids).logits[0].float()
                logprobs = torch.log_softmax(logits, dim=-1)
                total = 0.0
                # token t is predicted by logits at position t-1
                for t in range(max(n_ctx, 1), len(full_ids)):
                    total += float(logprobs[t - 1, full_ids[t]])
                row.append(total / n_answer)
            scores.append(row)
        return scores

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
