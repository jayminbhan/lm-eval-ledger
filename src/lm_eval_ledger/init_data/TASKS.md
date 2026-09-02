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

Every bare task name scores by **generation** (free-form response +
`\boxed{}` extraction) - the mode that matches how models are used and
what leaderboards report. MCQ tasks additionally offer logprob scoring
as explicit opt-in variants, named by suffix:

| variant | how it scores | backends |
|---|---|---|
| *(bare name)* | free-form generation + `\boxed{}` extraction | all |
| `<task>_logprob_token` | first-token log-probability over choice letters | vllm, hf, sglang, server* |
| `<task>_logprob_seq` | completion log-likelihood of each full answer | vllm, hf |

\* server: needs an endpoint that returns logprobs.

Logprob modes are cheap (no generation) and useful for base models or
for measuring the scoring-method difference on the same model - e.g.
run both `gpqa_diamond` and `gpqa_diamond_logprob_token` and compare.
The old `<task>_generate` names remain accepted as aliases.

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

Bare name = generate scoring; append `_logprob_token` / `_logprob_seq`
for the logprob variants.

| task | default k | few-shot | n | dataset |
|---|---|---|---|---|
| `gpqa_diamond` | 0 | 0-shot only (single split, no held-out pool) | 198 | Idavidrein/gpqa [gpqa_diamond] — gated |
| `gpqa_main` | 0 | 0-shot only (single split, no held-out pool) | 448 | Idavidrein/gpqa [gpqa_main] — gated |
| `gpqa_extended` | 0 | 0-shot only (single split, no held-out pool) | 546 | Idavidrein/gpqa [gpqa_extended] — gated |
| `mmlu_pro` | 0 | validation split (pool 70)¹ | ~12000 | TIGER-Lab/MMLU-Pro (up to 10 options) |
| `mmlu_redux_1` | 0 | 0-shot only | ~3000 | edinburgh-dawg/mmlu-redux |
| `mmlu_redux_2` | 0 | 0-shot only | ~5700 | edinburgh-dawg/mmlu-redux-2.0 |

¹ Exemplars are the first k of the split, not per-category as in the
official MMLU-Pro protocol — comparable across your own runs, slightly
off-protocol versus the paper's 5-shot numbers.

## Reasoning / commonsense

| task | default k | few-shot | n | dataset |
|---|---|---|---|---|
| `bbh` | 0 | 0-shot only | ~6500 | lukaemon/bbh (27 subtasks) |
| `arc_challenge` | 0 | train split (pool 1119) | 1172 | allenai/ai2_arc [ARC-Challenge] |
| `arc_easy` | 0 | train split (pool 2251) | 2376 | allenai/ai2_arc [ARC-Easy] |
| `hellaswag` | 0 | train split (pool 39905) | ~10000 | Rowan/hellaswag |
| `winogrande` | 0 | train split (pool 40398) | 1267 | allenai/winogrande [winogrande_xl] |

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
