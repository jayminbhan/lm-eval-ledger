# lm-eval-ledger
**lm-eval-ledger** is an LLM evaluation harness with a built-in web app for browsing and comparing what models generate. Every generation is logged to SQLite in real time, so you can read the prompt, response, extracted answer, and score for any sample while the run is still going. Models and tasks are defined in one YAML file: benchmark many models × many tasks in a single run.


Live demo: [lm-eval-ledger on Hugging Face Spaces](https://huggingface.co/spaces/...)


**Features**

- **One YAML, many runs** — benchmark any number of models × tasks in a single command
- **Per-sample SQLite logging** — prompt, response, gold, extracted answer, stop reason, and score, written as each sample finishes
- **Web app** — browse, compare, and delete runs and samples, even mid-benchmark
- **Backends** — vLLM, SGLang, HF transformers, or any OpenAI-compatible server (e.g. llama.cpp)
- **Interactive config** — unset configs like thinking mode are detected from the model and offered as a menu
- **Custom tasks** — add your own tasks and get the same per-sample logging and inspection.

<p align="center">
  <img src="src/lm_eval_ledger/webapp/static/sample-inspection.png" width="100%" alt="Sample Inspection">
</p>

**Sample Inspection** — every prompt, response, extracted answer, and score, color-coded by outcome. Filter by model, task, or right/wrong, and compare how different models answered the same question.

<table>
  <tr>
    <td width="50%" valign="top">
      <img src="src/lm_eval_ledger/webapp/static/leaderboard.png" alt="Leaderboard"><br>
      <b>Leaderboard</b> — pick a task and see every model ranked by accuracy, across all runs. One click to keep only the best per model.
    </td>
    <td width="50%" valign="top">
      <img src="src/lm_eval_ledger/webapp/static/run-history.png" alt="Run History"><br>
      <b>Run History</b> — browse all runs, check accuracy per model and task, grab the exact YAML that produced them, or delete a whole run or a single benchmark.
    </td>
  </tr>
</table>

## Install

```bash
pip install lm-eval-ledger

# Or install from source:
git clone https://github.com/jayminbhan/lm-eval-ledger
cd lm-eval-ledger
pip install -e .   # add a backend: pip install -e ".[vllm]"

```

Inference backends are optional extras — install the one you will use:

```bash
pip install "lm-eval-ledger[vllm]"     # vLLM
pip install "lm-eval-ledger[sglang]"   # SGLang
pip install "lm-eval-ledger[hf]"       # HF transformers
```

## Quickstart

```bash
lm-eval-ledger init            # write template.yaml, create results/ and logs/
lm-eval-ledger -c bench.yaml   # run benchmarks
lm-eval-ledger serve           # browse at http://localhost:8090
```

YAML format: [template.yaml](#yaml-template), available tasks: [Task Registry](#task-registry)

## Backends

One config format, four engines:

| backend  | install                              | OS             |
|----------|--------------------------------------|----------------|
| `vllm`   | `pip install "lm-eval-ledger[vllm]"`   | Linux          |
| `sglang` | `pip install "lm-eval-ledger[sglang]"` | Linux          |
| `hf`     | `pip install "lm-eval-ledger[hf]"`     | Linux, Windows |
| `server` | `pip install lm-eval-ledger` *(no extra)* | Linux, Windows |

The first three run the model in-process. `server` talks to any OpenAI-compatible
endpoint instead — local (llama.cpp, ollama, LM Studio) or hosted (OpenAI...).

### Example: llama.cpp

Start a llama.cpp server:

```bash
llama-server -hf unsloth/Qwen3.8-27B-GGUF:Q4_K_XL -ngl 999 -c 65536 -np 4 --jinja
```


Point the config at it:

```yaml
backend: server
server_url: http://localhost:8080/v1
server_concurrency: 4        # match the server's -np slots
```

Any OpenAI-compatible endpoint works the same way.

## Thinking mode

Set `chat_template_kwargs` per model to control thinking. If it is left unset, the harness reads the model's chat template (from the HF cache, or a llama.cpp server's `/props`), detects the available knobs, offers a menu, and prints the YAML to make the choice permanent.

```yaml
models:
  - name: Qwen/Qwen3.8-27B
    chat_template_kwargs: {enable_thinking: true, reasoning_effort: high}
  - google/gemma-3-12b-it        # no thinking knob; global settings apply
```

## YAML template

Every option with its default in shared and per-backend blocks. Uncomment what you need.

<summary><b>template.yaml</b></summary>

```yaml
# lm-eval-ledger config template
# Copy this template or write directly in this template for benchmark run.
# Any field is also a CLI flag (--max-tokens 4096). 
# The config of every run is stored in the SQLite for reproduction.
# Layout: SHARED behaves identically on every backend; PER-BACKEND is one
# block per backend - keep the block you use, comment out the rest.

# ════════════════════════════════════════════════════════════
# SHARED - backend-agnostic
# ════════════════════════════════════════════════════════════

# List model names to evaluate. Each model is run sequentially on every task.
# Model name depends on the backend:
#   vllm / sglang / hf:  an HF repo id or a local checkpoint path
#   server:              the name the endpoint reports - copy it verbatim
#                        from `curl <server_url>/models`. Multi-model
#                        servers (ollama, hosted APIs) switch models per
#                        request, so several entries work in one run.

# Config under model entry overrides globals for that model only
# (chat_template_kwargs, quantization, apply_chat_template, ...).
models:
  # - Qwen/Qwen3.5-2B
  # - name: Qwen/Qwen3.5-9B
  #   chat_template_kwargs: {enable_thinking: true}
  # - google/gemma-4-12B-it-qat-w4a16-ct 


# "name":(task-default few-shot), "name:4", or "name:0,4" (0-shot and 4-shot).
# Full list: lm-eval-ledger --help or the README's Tasks section. MCQ tasks:
# bare name = generate scoring; _logprob_token | _logprob_seq variants.
tasks:
  - gsm8k:0
  - gpqa_diamond:0

max_examples: null   # per-task cap; null = all (set ~20 for a smoke test)

# For image-bearing questions: text = drop them;
# all = send images (needs a vision-capable model on backend: server).
modality: text

apply_chat_template: true   # true for instruct/chat, false for base

# Thinking-mode control (model-family specific chat-template kwargs,
# e.g. {enable_thinking: false}). Left unset, a terminal launch offers
# a menu per model and prints the YAML to pin the choice.
# chat_template_kwargs: {}   # uncomment to silence the menu

# Sampling (thinking models need sampling - check the model card).
temperature: 0.6
top_p: 0.95
max_tokens: 2048     # generation budget; thinking modes want 8192+
pass_k: 1            # best-of-k scoring; >1 needs temperature > 0

logs_dir: logs
db_path: null        # THE ledger; null = ./results/ledger.sqlite3

# ════════════════════════════════════════════════════════════
# PER-BACKEND - backend-specific: keep ONE block, comment out the rest
# ════════════════════════════════════════════════════════════

# ── vllm (in-process, fastest; pip install lm-eval-ledger[vllm]) ────
backend: vllm
gpu_memory_utilization: 0.90
max_model_len: 8192   # context window (prompt + max_tokens must fit)
enforce_eager: true
gpu_ids: null         # [3] = pin GPU; [0,1] = one worker per GPU
tensor_parallel_size: 1   # GPUs per model; gpu_ids is split into groups of this size
quantization: null    # "bitsandbytes" | "awq" | "gptq" | "fp8" | {tag: method}
batch_size: 100       # write to the ledger every N samples (throughput
                      # is unaffected; null = single engine call, results
                      # land only at task end)

# ── hf (transformers; every architecture, slow; [hf] extra) ─────────
# backend: hf
# max_model_len: 8192
# gpu_ids: null
# tensor_parallel_size: 1  # GPUs per model; gpu_ids is split into groups of this size (hf: layer split)
# quantization: null  # "bitsandbytes" only
# batch_size: 8       # true VRAM knob here - keep small

# ── sglang (in-process; [sglang] extra) ─────────────────────────────
# backend: sglang
# gpu_memory_utilization: 0.90
# max_model_len: 8192
# gpu_ids: null
# tensor_parallel_size: 1  # GPUs per model; gpu_ids is split into groups of this size
# batch_size: 100

# ── server (any OpenAI-compatible endpoint: llama.cpp, ollama,
#    hosted APIs; no extra install). The server owns model loading,
#    context size, and quantization. models: entries are the names the
#    endpoint serves (see models: above);
# backend: server
# server_url: http://localhost:8080/v1
# api_key: null
# server_concurrency: 4    # = llama-server -np slots
# request_timeout: 600     # seconds; thinking modes can take minutes
# server_extra_body: null  # extra JSON per request, e.g. {top_k: 20}

```

</details>

# Task registry

Every task string accepted in a config's `tasks:` list (or `--task`).

Usage forms (see `template.yaml`):

```yaml
tasks:
  - gpqa_diamond      # task-default few-shot count
  - gsm8k:0           # explicit few-shot count
  - mmlu_pro:0,4      # few-shot ladder (two benchmarks)
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

## Science / knowledge MCQ

All tasks in this section are **+logprob** (both suffix variants).

| task | default k | few-shot | n | dataset |
|---|---|---|---|---|
| `gpqa_diamond` | 0 | 0-shot only (single split, no held-out pool) | 198 | Idavidrein/gpqa [gpqa_diamond] — gated |
| `gpqa_main` | 0 | 0-shot only (single split, no held-out pool) | 448 | Idavidrein/gpqa [gpqa_main] — gated |
| `gpqa_extended` | 0 | 0-shot only (single split, no held-out pool) | 546 | Idavidrein/gpqa [gpqa_extended] — gated |
| `mmlu_pro` | 0 | validation split (pool 70)¹ | 12032 | TIGER-Lab/MMLU-Pro (up to 10 options) |
| `mmlu_redux_1` | 0 | 0-shot only | 2801 | edinburgh-dawg/mmlu-redux (3000 minus questions the dataset flags as flawed; wrong-groundtruth golds remapped) |
| `mmlu_redux_2` | 0 | 0-shot only | 5431 | edinburgh-dawg/mmlu-redux-2.0 (5700 minus flagged-flawed questions; wrong-groundtruth golds remapped) |

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
| `livecodebench` | 0 | 0-shot only (self-contained prompts) | 1055 | official release jsonls (release_v6) | EXECUTES generated code locally; `max_tokens >= 2048` |
| `livecodebench_v1` … `livecodebench_v6` | 0 | 0-shot only | 400 / 111 / 101 / 101 / 167 / 175 | testN.jsonl | the problems ADDED in release N (upstream's own slices); highest N = newest = most contamination-safe |

`livecodebench_vN` (N = 1-6) loads exactly one upstream release file. Bare `livecodebench` is the full archive (all six).

## Notes

- **Gated datasets** (GPQA): accept the terms on the HF dataset page,
  then `hf auth login`, before first load.
