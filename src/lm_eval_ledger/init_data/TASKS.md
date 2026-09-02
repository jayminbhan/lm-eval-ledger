# Task registry

Every task string accepted in a config's `tasks:` list (or `--task`).
`lm-eval-ledger --help` shows the current list.

Usage forms (see `template.yaml`):

```yaml
tasks:
  - gpqa_diamond                 # task-default few-shot count
  - gsm8k:0                 # explicit few-shot count
  - mmlu_pro_generate:0,4        # few-shot ladder (two benchmarks)
```

Every bare task name scores by **generation** (free-form response +
`\boxed{}` answer extraction). MCQ tasks additionally offer logprob scoring. Named by suffix:

| variant | how it scores | backends |
|---|---|---|
| *(bare name)* | free-form generation + `\boxed{}` extraction | all |
| `<task>_logprob_token` | first-token log-probability over choice letters | vllm, hf, sglang, server* |
| `<task>_logprob_seq` | completion log-likelihood of each full answer | vllm, hf |

\* server: llama.cpp only 

Logprob variants exist for exactly the MCQ tasks marked **+logprob**
in the tables below; every other task
is generate-only. Logprob modes are cheap (no generation) and useful for
base models or for measuring the scoring-method difference on the same
model - e.g. run both `gpqa_diamond` and `gpqa_diamond_logprob_token`
and compare.

Few-shot: `name:k` draws the first k exemplars from the task's few-shot
source. Tasks listed as **0-shot only** have no such source. The few-shot size below is
the maximum k.

## Math

| task | default k | few-shot | n | dataset |
|---|---|---|---|---|
| `aime_2024` | 0 | 0-shot only | 30 | HuggingFaceH4/aime_2024 |
| `aime_2025` | 0 | 0-shot only | 30 | MathArena/aime_2025 |
| `gsm8k` | 8 | train split (pool 7473) | 1319 | openai/gsm8k |
| `math` | 0 | algebra/train split (pool 1744) | 5000 | EleutherAI/hendrycks_math (7 subjects; the MATH benchmark, Hendrycks et al.) |
| `math500` | 0 | 0-shot only | 500 | HuggingFaceH4/MATH-500 |
| `olympiad_bench_math_en` | 0 | 0-shot only | 674 | Hothan/OlympiadBench [OE_TO_maths_en_COMP] |
| `olympiad_bench_physics_en` | 0 | 0-shot only | 236 | Hothan/OlympiadBench [OE_TO_physics_en_COMP] |
| `theoremqa` | 0 | 0-shot only | 800 | TIGER-Lab/TheoremQA — 53 image questions (`modality`) |

## Science / knowledge MCQ

All tasks in this section are **+logprob** (both suffix variants).

| task | default k | few-shot | n | dataset |
|---|---|---|---|---|
| `gpqa_diamond` | 0 | 0-shot only (single split, no held-out pool) | 198 | Idavidrein/gpqa [gpqa_diamond] — gated |
| `gpqa_main` | 0 | 0-shot only (single split, no held-out pool) | 448 | Idavidrein/gpqa [gpqa_main] — gated |
| `gpqa_extended` | 0 | 0-shot only (single split, no held-out pool) | 546 | Idavidrein/gpqa [gpqa_extended] — gated |
| `mmlu_pro` | 0 | validation split (pool 70)¹ | 12032 | TIGER-Lab/MMLU-Pro (up to 10 options) |
| `mmlu_redux_1` | 0 | 0-shot only | 3000 | edinburgh-dawg/mmlu-redux |
| `mmlu_redux_2` | 0 | 0-shot only | 5700 | edinburgh-dawg/mmlu-redux-2.0 |

¹ Exemplars are the first k of the split, not per-category as in the
official MMLU-Pro protocol — comparable across your own runs, slightly
off-protocol versus the paper's 5-shot numbers.

## Reasoning / commonsense

| task | default k | few-shot | n | dataset |
|---|---|---|---|---|
| `bbh` | 0 | 0-shot only | 6511 | lukaemon/bbh (27 subtasks) |
| `arc_challenge` **+logprob** | 0 | train split (pool 1119) | 1172 | allenai/ai2_arc [ARC-Challenge] |
| `arc_easy` **+logprob** | 0 | train split (pool 2251) | 2376 | allenai/ai2_arc [ARC-Easy] |
| `hellaswag` **+logprob** | 0 | train split (pool 39905) | 10042 | Rowan/hellaswag |
| `winogrande` **+logprob** | 0 | train split (pool 40398) | 1267 | allenai/winogrande [winogrande_xl] |

## Frontier / specialty

| task | default k | few-shot | n | dataset | notes |
|---|---|---|---|---|---|
| `hle` | 0 | 0-shot only | 2500 (2158 text-only) | cais/hle | gated; 342 image questions (`modality`); string-match is a lower bound of the official LLM-judge scoring |
| `livecodebench` | 0 | 0-shot only (self-contained prompts) | 1055 | official release jsonls (release_v6) | EXECUTES generated code locally; `max_tokens >= 2048` |
| `livecodebench_v5_delta` | 0 | 0-shot only | 167 | test5.jsonl (problems added in release_v5) | contests 2024-09 to 2025-01 |
| `livecodebench_v6_delta` | 0 | 0-shot only | 175 | test6.jsonl (problems added in release_v6) | contests 2025-01 to 2025-04; the newest, most contamination-safe slice |
| `mrcr_2needle` / `mrcr_4needle` / `mrcr_8needle` | 0 | 0-shot only | 800 each | openai/mrcr | long context: needs `apply_chat_template` and a large `max_model_len` (32k+); partial credit (SequenceMatcher ratio) |

The `livecodebench_vN_delta` variants load exactly one upstream release
file (`testN.jsonl` = the problems ADDED in release N), so the slices
are the dataset's own units, not ranges we invented. Pick the newest
delta published after your model's training cutoff and state which one
you ran. Bare `livecodebench` is the full v6 archive: higher,
contamination-inflated numbers for recent models.

## Notes

- **Gated datasets** (GPQA, HLE): accept the terms on the HF dataset page,
  then `hf auth login`, before first load.
- **Image-bearing tasks** (HLE, TheoremQA): `modality: text` (default) drops
  image questions; `modality: all` sends images (vision-capable
  `backend: server` model required).
