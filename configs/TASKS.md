# Task registry

Every task string accepted in a config's `tasks:` list (or `--task`).
Generated from the live registry (`lm_eval_ledger.tasks.TASK_REGISTRY`);
`lm-eval-ledger --help` always shows the current list.

Usage forms (see `reference.yaml`):

```yaml
tasks:
  - gpqa_diamond_generate        # task-default few-shot count
  - gsm8k_main:0                 # explicit few-shot count
  - mmlu_pro_generate:0,4        # few-shot ladder (two benchmarks)
```

Eval modes (the suffix on MCQ families):

| mode | how it scores | backends |
|---|---|---|
| `generate` | free-form generation + `\boxed{}` extraction | all |
| `logprob_token` | first-token log-probability over choice letters | vllm, hf, sglang, server* |
| `logprob_seq` | completion log-likelihood of each full answer | vllm, hf |

\* server: needs an endpoint that returns logprobs.

## Math

| task | fewshot | n | dataset |
|---|---|---|---|
| `aime_2024` | 0 | 30 | HuggingFaceH4/aime_2024 |
| `aime_2025` | 0 | 30 | MathArena/aime_2025 |
| `gsm8k_main` | 8 | 1319 | openai/gsm8k [main] |
| `gsm8k_socratic` | 8 | 1319 | openai/gsm8k [socratic] |
| `hendrycks_math` | 0 | ~5000 | EleutherAI/hendrycks_math (7 subjects) |
| `math500` | 0 | 500 | HuggingFaceH4/MATH-500 |
| `olympiad_bench_math_en` | 0 | ~675 | Hothan/OlympiadBench [OE_TO_maths_en_COMP] |
| `olympiad_bench_physics_en` | 0 | ~250 | Hothan/OlympiadBench [OE_TO_physics_en_COMP] |
| `theoremqa` | 0 | 800 | TIGER-Lab/TheoremQA — 53 image questions (`modality`) |

## Science / knowledge MCQ

Each family below has all three eval-mode variants:
`<family>_generate`, `<family>_logprob_token`, `<family>_logprob_seq`.

| family | fewshot | n | dataset |
|---|---|---|---|
| `gpqa_diamond_*` | 0 | 198 | Idavidrein/gpqa [gpqa_diamond] — gated |
| `gpqa_main_*` | 0 | 448 | Idavidrein/gpqa [gpqa_main] — gated |
| `gpqa_extended_*` | 0 | 546 | Idavidrein/gpqa [gpqa_extended] — gated |
| `mmlu_pro_*` | 0 | ~12000 | TIGER-Lab/MMLU-Pro (up to 10 options) |
| `mmlu_redux_1_*` | 0 | ~3000 | edinburgh-dawg/mmlu-redux |
| `mmlu_redux_2_*` | 0 | ~5700 | edinburgh-dawg/mmlu-redux-2.0 |

## Reasoning / commonsense

| task | fewshot | n | dataset |
|---|---|---|---|
| `bbh` | 0 | ~6500 | lukaemon/bbh (27 subtasks) |
| `arc_challenge_*` (3 modes) | 0 | 1172 | allenai/ai2_arc [ARC-Challenge] |
| `arc_easy_*` (3 modes) | 0 | 2376 | allenai/ai2_arc [ARC-Easy] |
| `hellaswag_*` (3 modes) | 0 | ~10000 | Rowan/hellaswag |
| `winogrande_*` (3 modes) | 0 | 1267 | allenai/winogrande [winogrande_xl] |

## Frontier / specialty

| task | fewshot | n | dataset | notes |
|---|---|---|---|---|
| `hle` | 0 | 2500 (2158 text-only) | cais/hle | gated; 342 image questions (`modality`); string-match is a lower bound of the official LLM-judge scoring |
| `livecodebench` | 0 | ~1055 | official release jsonls (release_v6) | EXECUTES generated code locally; `max_tokens >= 2048` |
| `mrcr_2needle` / `mrcr_4needle` / `mrcr_8needle` | 0 | varies | openai/mrcr | long context: needs `apply_chat_template` and a large `max_model_len` (32k+); partial credit (SequenceMatcher ratio) |

## Notes

- **Gated datasets** (GPQA, HLE): accept the terms on the HF dataset page,
  then `hf auth login`, before first load.
- **Image-bearing tasks** (HLE, TheoremQA): `modality: text` (default) drops
  image questions; `modality: all` sends images (vision-capable
  `backend: server` model required).
- **n** values marked `~` are approximate; exact counts print at load time.
