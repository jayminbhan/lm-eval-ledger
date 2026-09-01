# Task registry

Every task string accepted in a config's `tasks:` list (or `--task`).
Generated from the live registry (`lm_eval_ledger.tasks.TASK_REGISTRY`);
`lm-eval-ledger --help` always shows the current list.

Usage forms (see `template.yaml`):

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

Few-shot: `name:k` draws the first k exemplars from the task's few-shot
source — a held-out HF split (never the eval set), or a curated local
file under `data/`. Tasks listed as **0-shot only** have no such source
(no held-out split exists, or few-shot is not meaningful for the task);
requesting `:k` on them currently runs 0-shot. The pool size below is
the maximum k, though practical values are 0-8.

## Math

| task | default k | few-shot | n | dataset |
|---|---|---|---|---|
| `aime_2024` | 0 | 0-shot only | 30 | HuggingFaceH4/aime_2024 |
| `aime_2025` | 0 | 0-shot only | 30 | MathArena/aime_2025 |
| `gsm8k_main` | 8 | train split (pool 7473) | 1319 | openai/gsm8k [main] |
| `gsm8k_socratic` | 8 | main/train split (pool 7473) | 1319 | openai/gsm8k [socratic] |
| `hendrycks_math` | 0 | algebra/train split (pool 1744) | ~5000 | EleutherAI/hendrycks_math (7 subjects) |
| `math500` | 0 | 0-shot only | 500 | HuggingFaceH4/MATH-500 |
| `olympiad_bench_math_en` | 0 | 0-shot only | ~675 | Hothan/OlympiadBench [OE_TO_maths_en_COMP] |
| `olympiad_bench_physics_en` | 0 | 0-shot only | ~250 | Hothan/OlympiadBench [OE_TO_physics_en_COMP] |
| `theoremqa` | 0 | 0-shot only | 800 | TIGER-Lab/TheoremQA — 53 image questions (`modality`) |

## Science / knowledge MCQ

Each family below has all three eval-mode variants:
`<family>_generate`, `<family>_logprob_token`, `<family>_logprob_seq`.

| family | default k | few-shot | n | dataset |
|---|---|---|---|---|
| `gpqa_diamond_*` | 0 | 0-shot only (single split, no held-out pool) | 198 | Idavidrein/gpqa [gpqa_diamond] — gated |
| `gpqa_main_*` | 0 | 0-shot only (single split, no held-out pool) | 448 | Idavidrein/gpqa [gpqa_main] — gated |
| `gpqa_extended_*` | 0 | 0-shot only (single split, no held-out pool) | 546 | Idavidrein/gpqa [gpqa_extended] — gated |
| `mmlu_pro_*` | 0 | validation split (pool 70)¹ | ~12000 | TIGER-Lab/MMLU-Pro (up to 10 options) |
| `mmlu_redux_1_*` | 0 | 0-shot only | ~3000 | edinburgh-dawg/mmlu-redux |
| `mmlu_redux_2_*` | 0 | 0-shot only | ~5700 | edinburgh-dawg/mmlu-redux-2.0 |

¹ Exemplars are the first k of the split, not per-category as in the
official MMLU-Pro protocol — comparable across your own runs, slightly
off-protocol versus the paper's 5-shot numbers.

## Reasoning / commonsense

| task | default k | few-shot | n | dataset |
|---|---|---|---|---|
| `bbh` | 0 | 0-shot only | ~6500 | lukaemon/bbh (27 subtasks) |
| `arc_challenge_*` (3 modes) | 0 | train split (pool 1119) | 1172 | allenai/ai2_arc [ARC-Challenge] |
| `arc_easy_*` (3 modes) | 0 | train split (pool 2251) | 2376 | allenai/ai2_arc [ARC-Easy] |
| `hellaswag_*` (3 modes) | 0 | train split (pool 39905) | ~10000 | Rowan/hellaswag |
| `winogrande_*` (3 modes) | 0 | train split (pool 40398) | 1267 | allenai/winogrande [winogrande_xl] |

## Frontier / specialty

| task | default k | few-shot | n | dataset | notes |
|---|---|---|---|---|---|
| `hle` | 0 | 0-shot only | 2500 (2158 text-only) | cais/hle | gated; 342 image questions (`modality`); string-match is a lower bound of the official LLM-judge scoring |
| `livecodebench` | 0 | 0-shot only (self-contained prompts) | 1055 | official release jsonls (release_v6) | EXECUTES generated code locally; `max_tokens >= 2048` |
| `mrcr_2needle` / `mrcr_4needle` / `mrcr_8needle` | 0 | 0-shot only | varies | openai/mrcr | long context: needs `apply_chat_template` and a large `max_model_len` (32k+); partial credit (SequenceMatcher ratio) |

## Notes

- **Gated datasets** (GPQA, HLE): accept the terms on the HF dataset page,
  then `hf auth login`, before first load.
- **Image-bearing tasks** (HLE, TheoremQA): `modality: text` (default) drops
  image questions; `modality: all` sends images (vision-capable
  `backend: server` model required).
- **Custom few-shot exemplars**: any task can be given a curated
  `data/<file>.jsonl` via its `fewshot_path` — the local file takes
  priority over the HF split when both exist.
- **n** values marked `~` are approximate; exact counts print at load time.
