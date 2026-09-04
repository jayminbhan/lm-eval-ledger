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
lm-eval-ledger init                                 # write template.yaml + TASKS.md, create results/ and logs/
lm-eval-ledger -c config.yaml                       # run benchmarks
lm-eval-ledger serve --db results/ledger.sqlite3    # browse at http://localhost:8090
```

YAML format: [template.yaml](#yaml-template), available tasks: [TASKS.md](src/lm_eval_ledger/init_data/TASKS.md)

## Backends

One config format, four engines:

| backend  | install                              | OS             |
|----------|--------------------------------------|----------------|
| `vllm`   | `pip install "lm-eval-ledger[vllm]"`   | Linux          |
| `sglang` | `pip install "lm-eval-ledger[sglang]"` | Linux          |
| `hf`     | `pip install "lm-eval-ledger[hf]"`     | Linux, Windows |
| `server` | `pip install lm-eval-ledger` *(no extra)* | Linux, Windows |

The first three run the model in-process. `server` talks to any OpenAI-compatible
endpoint instead — local (llama.cpp, ollama, LM Studio) or hosted (OpenAI, Together, ...).

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

Set `chat_template_kwargs` per model to control thinking. If it is left unset, the harness reads the model's chat template (from the HF cache, or a llama.cpp server's `/props`), detects the available knobs, offers a menu, and prints the YAML to make the choice permanent. Set `chat_template_kwargs: {}` to skip the prompt and use the template defaults.

```yaml
models:
  - name: Qwen/Qwen3.8-27B
    chat_template_kwargs: {enable_thinking: true, reasoning_effort: high}
  - google/gemma-3-12b-it        # no thinking knob; global settings apply
```

## YAML template

Every option with its default in shared and per-backend blocks. Uncomment what you need.

<details>
<summary><b>template.yaml</b> (click to expand)</summary>

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
  - Qwen/Qwen2.5-7B-Instruct
  # - name: Qwen/Qwen3-8B
  #   chat_template_kwargs: {enable_thinking: true}
  # - google/gemma-4-12B-it-qat-w4a16-ct 


# "name" (task-default few-shot), "name:4", or "name:0,4" (ladder).
# Full list: lm-eval-ledger --help or ./TASKS.md (written by init). MCQ tasks:
# bare name = generate scoring; _logprob_token | _logprob_seq variants.
tasks:
  - gsm8k:0
  - gpqa_diamond:0

max_examples: null   # per-task cap; null = all (set ~20 for a smoke test)

# Image-bearing questions (HLE, TheoremQA): text = drop them;
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

# LLM judge for free-form answers - applies ONLY to HLE and TheoremQA
# (other tasks score exactly and are never judged). Runs after the task
# with CompassVerifier; the official protocol uses a GPT-4o judge, only
# CompassVerifier is implemented here.
verifier: off         # 7b | 3b | off

# ════════════════════════════════════════════════════════════
# PER-BACKEND - backend-specific: keep ONE block, comment out the rest
# ════════════════════════════════════════════════════════════

# ── vllm (in-process, fastest; pip install lm-eval-ledger[vllm]) ────
backend: vllm
gpu_memory_utilization: 0.90
max_model_len: 8192   # context window (prompt + max_tokens must fit)
enforce_eager: true
gpu_ids: null         # [3] = pin GPU; [0,1] = one worker per GPU
quantization: null    # "bitsandbytes" | "awq" | "gptq" | "fp8" | {tag: method}
batch_size: 100       # write to the ledger every N samples (throughput
                      # is unaffected; null = single engine call, results
                      # land only at task end)

# ── hf (transformers; every architecture, slow; [hf] extra) ─────────
# backend: hf
# max_model_len: 8192
# gpu_ids: null
# quantization: null  # "bitsandbytes" only
# batch_size: 8       # true VRAM knob here - keep small

# ── sglang (in-process; [sglang] extra) ─────────────────────────────
# backend: sglang
# gpu_memory_utilization: 0.90
# max_model_len: 8192
# gpu_ids: null
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