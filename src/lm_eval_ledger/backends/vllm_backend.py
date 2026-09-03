# backends/vllm_backend.py
"""vLLM backend: in-process batched inference. The reference backend -
fastest, fully deterministic, supports every eval mode."""
from __future__ import annotations

import os

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.distributed.parallel_state import destroy_model_parallel  # noqa: E402

from .base import Backend, GenResult


class VllmBackend(Backend):
    name = "vllm"
    capabilities = frozenset({"generate", "logprob_token", "logprob_seq"})

    def __init__(self):
        self.llm: LLM | None = None
        self.tokenizer = None
        self._batch_size: int | None = None

    # ---- lifecycle ----

    def load(self, model: str, cfg, quantization: str | None = None) -> None:
        import sys
        import vllm as _vllm
        if sys.version_info < (3, 13) and _vllm.__version__ < "0.28":
            # vLLM <= 0.27 imports flashinfer (which used 3.13-only syntax)
            # unconditionally during engine warmup; fixed in 0.28.
            print(f"[WARN] vLLM {_vllm.__version__} on Python < 3.13 may crash "
                  f"at engine init (flashinfer); upgrade to vllm>=0.28 or use "
                  f"a 3.13 environment")
        kwargs = {
            "model": model,
            "trust_remote_code": True,
            "gpu_memory_utilization": cfg.gpu_memory_utilization,
            "enforce_eager": cfg.enforce_eager,
            "max_logprobs": 100,  # logprob_token needs top-100
        }
        if cfg.max_model_len is not None:
            kwargs["max_model_len"] = cfg.max_model_len
        if quantization:
            kwargs["quantization"] = quantization
        self.llm = LLM(**kwargs)
        self.tokenizer = self.llm.get_tokenizer()
        self.template_kwargs = dict(cfg.chat_template_kwargs or {})

    def unload(self) -> None:
        import gc
        del self.llm
        self.llm = None
        self.tokenizer = None
        destroy_model_parallel()
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    # ---- generation ----

    def generate(self, prompts, *, temperature, top_p, max_tokens, stop, n,
                 seed, batch_size, on_result=None):
        """One engine call by default (best throughput: the engine batches
        continuously over the full set). With batch_size set, generation is
        chunked and each finished chunk streams through on_result - samples
        reach the ledger per chunk at a modest throughput cost."""
        sampling_params = SamplingParams(
            temperature=temperature, top_p=top_p, max_tokens=max_tokens,
            stop=stop or None, n=n, skip_special_tokens=False, seed=seed,
        )
        chunk = batch_size if batch_size and batch_size > 0 else len(prompts)
        num_batches = max((len(prompts) + chunk - 1) // chunk, 1)
        if num_batches > 1:
            print(f"[INFO] Processing in {num_batches} batches of size {chunk}")
        results: list[list[GenResult]] = []
        for start in range(0, len(prompts), chunk):
            if num_batches > 1:
                print(f"[INFO] Batch {start // chunk + 1}/{num_batches}...")
            outs = self.llm.generate(prompts[start:start + chunk],
                                     sampling_params=sampling_params,
                                     use_tqdm=True)
            for j, out in enumerate(outs):
                group = [GenResult(text=r.text,
                                   finish_reason=r.finish_reason or "",
                                   stop_reason=r.stop_reason,
                                   n_tokens=len(r.token_ids))
                         for r in out.outputs]
                if on_result is not None:
                    on_result(start + j, group)
                results.append(group)
        return results

    # ---- logprob primitives ----

    def first_token_logprobs(self, prompts, labels, *, seed=0):
        # Labels follow "Answer:", so tokenize with a leading space
        choice_token_ids = {}
        for label in labels:
            token_ids = self.tokenizer.encode(f" {label}", add_special_tokens=False)
            choice_token_ids[label] = token_ids[-1]
        print(f"[INFO] Choice token IDs: {choice_token_ids}")

        sampling_params = SamplingParams(
            max_tokens=1, temperature=0, logprobs=100, seed=seed)
        outputs = self.llm.generate(prompts, sampling_params=sampling_params,
                                    use_tqdm=True)
        results = []
        for output in outputs:
            gen_logprobs = output.outputs[0].logprobs[0]
            logprobs_dict = {}
            for label, token_id in choice_token_ids.items():
                if token_id in gen_logprobs:
                    lp = gen_logprobs[token_id]
                    logprobs_dict[label] = float(getattr(lp, "logprob", lp))
                else:
                    logprobs_dict[label] = float("-inf")  # not in top-100
            results.append(logprobs_dict)
        return results

    def score_completions(self, base_prompts, choice_texts, *, seed=0):
        all_prompts: list[str] = []
        prompt_map: list[tuple[int, int]] = []
        for ex_idx, (base, choices) in enumerate(zip(base_prompts, choice_texts)):
            for c_idx, choice in enumerate(choices):
                all_prompts.append(base + " " + choice)
                prompt_map.append((ex_idx, c_idx))

        sampling_params = SamplingParams(
            max_tokens=1, temperature=0, prompt_logprobs=1, seed=seed)
        outputs = self.llm.generate(all_prompts, sampling_params=sampling_params,
                                    use_tqdm=True)

        scores = [[0.0] * len(choices) for choices in choice_texts]
        for out_idx, output in enumerate(outputs):
            ex_idx, c_idx = prompt_map[out_idx]
            prompt_token_ids = output.prompt_token_ids
            prompt_logprobs = output.prompt_logprobs
            # The answer starts where the full prompt's tokens diverge from
            # the base prompt's (the seam can merge tokens, so len() is off)
            base_token_ids = self.tokenizer.encode(base_prompts[ex_idx])
            n_ctx = 0
            for base_tok, full_tok in zip(base_token_ids, prompt_token_ids):
                if base_tok != full_tok:
                    break
                n_ctx += 1
            n_answer = max(len(prompt_token_ids) - n_ctx, 1)
            total = 0.0
            for i in range(n_ctx, len(prompt_token_ids)):
                if prompt_logprobs[i] is not None:
                    token_id = prompt_token_ids[i]
                    if token_id in prompt_logprobs[i]:
                        lp = prompt_logprobs[i][token_id]
                        total += float(getattr(lp, "logprob", lp))
            scores[ex_idx][c_idx] = total / n_answer
        return scores

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
