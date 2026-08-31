# lm-eval-ledger

**A YAML-driven LLM benchmark harness where every run appends to one SQLite ledger** — with a built-in web viewer, cross-run comparison, LLM-judge verification, and four inference backends behind one config format.

Most harnesses treat a benchmark run as a pile of output files. lm-eval-ledger treats it as a **row in a database you keep forever**: every run, benchmark, and individual sample (prompt, full response, extracted answer, score) lands in one SQLite file, with the exact config that produced it stored alongside. Three weeks later you can diff two runs sample-by-sample, re-judge old responses with a better verifier, or open the viewer and read the one response that ruined your average.

```bash
pip install lm-eval-ledger[vllm]
lm-eval-ledger -c bench.yaml
lm-eval-ledger serve          # browse results at http://localhost:8090
```

## Quickstart

```yaml
# bench.yaml
models:
  - Qwen/Qwen2.5-7B-Instruct
tasks:
  - gsm8k_main:0
  - gpqa_diamond_generate:0
apply_chat_template: true
temperature: 0.6
top_p: 0.95
max_tokens: 2048
```

```bash
lm-eval-ledger -c bench.yaml          # run; results append to results/ledger.sqlite3
lm-eval-ledger runs                   # list all runs and scores
lm-eval-ledger compare 3 7 --samples  # accuracy deltas + exactly which samples flipped
lm-eval-ledger config 3 > rerun.yaml  # reproduce any past run from its stored config
lm-eval-ledger serve                  # web viewer (leaderboard, sample inspection)
```

Every field is documented in [`configs/reference.yaml`](configs/reference.yaml) — a valid, loadable config that doubles as the reference manual. Every task string, with few-shot support and sample counts, is in [`configs/TASKS.md`](configs/TASKS.md). Any YAML field also works as a CLI flag (`--max-tokens 4096`); unknown keys are rejected loudly.

## Backends

One config format, four engines — pick at install time:

| backend | install | eval modes | OS | notes |
|---|---|---|---|---|
| `vllm` | `pip install lm-eval-ledger[vllm]` | generate, logprob_token, logprob_seq | Linux | in-process, fastest |
| `hf` | `pip install lm-eval-ledger[hf]` | all three | Linux, Windows | transformers + accelerate; every architecture; slow |
| `sglang` | `pip install lm-eval-ledger[sglang]` | generate, logprob_token | Linux | in-process sglang.Engine |
| `server` | *(no extra)* | generate, logprob_token* | Linux, Windows | any OpenAI-compatible endpoint |

`backend: server` benchmarks **llama.cpp, ollama, LM Studio, remote vLLM/SGLang, or hosted APIs** — anything speaking the OpenAI chat API. llama.cpp example:

```bash
llama-server -hf unsloth/Qwen3-30B-GGUF:Q4_K_XL -ngl 999 -c 65536 -np 4 --jinja
```

```yaml
backend: server
server_url: http://localhost:8080/v1
server_concurrency: 4        # match the server's -np slots
```

\* logprob mode requires an endpoint that returns logprobs.

On Windows, the base install (`server` backend) and `[hf]` are fully
supported; `[vllm]`/`[sglang]` skip their Linux-only engines at install
time (so `[all]` still resolves) and explain if selected at runtime.

## The ledger

One SQLite file, three tables — `runs` (config provenance, including your YAML byte-for-byte as written), `benchmarks` (one row per model × task × few-shot with all scores), `samples` (every prompt, every full response, every extracted answer). WAL mode means you can browse results **while a run is writing them**.

```bash
lm-eval-ledger serve --db results/ledger.sqlite3
```

The viewer is a small Flask app: run history (with per-run storage sizes, config view/download, delete/compact), a leaderboard with per-task best-score chips, and **Sample Inspection** — browse any benchmark's samples with inline prompt/response and wrong-answer shading, open any sample for the full text (thinking collapsed, answer prominent, byte-exact raw view), diff two benchmarks sample-by-sample, or find the questions *every* model gets wrong. Read-only by design; bind to localhost, or share with `--host 0.0.0.0 --token SECRET` behind a reverse proxy.

## Thinking-mode control

Reasoning models are controlled through their chat template, and templates differ by family — so the knob is the template's own kwargs, recorded verbatim in the run's provenance:

```yaml
models:
  - name: Qwen/Qwen3-8B
    chat_template_kwargs: {enable_thinking: true, reasoning_effort: high}
  - google/gemma-3-12b-it        # no thinking knob; global settings apply
```

Leave `chat_template_kwargs` unset and launch from a terminal: the harness reads each model's template (from the HF cache, or a llama.cpp server's `/props`), detects the knobs, offers a menu, and prints the YAML to make your choice permanent. `{}` opts out.

Per-model mappings override any global setting except sampling (kept global so models in one run stay comparable) — including `backend`, so one run can mix a local vLLM model with an API-served one.

## Scoring you can trust

- **Strict extraction**: the *last* `\boxed{}` in the response (reasoning models restate instructions and box intermediate values; the final box is the answer).
- **Task stop strings apply only to base-model completion mode** — chat mode ends on EOS, so thinking models are never cut mid-reasoning by a few-shot separator.
- **`stop_reason` on every sample** distinguishes "ran out of budget" from "answered and stopped" — truncation problems are visible, not laundered into accuracy.
- **Optional LLM-judge verification**: `verifier_model: opencompass/CompassVerifier-7B` re-judges string-match failures after the run (or retroactively: `lm-eval-ledger-verify ledger.sqlite3 --run 3`) and writes `verified_accuracy` *alongside* the strict score — never over it.
- **Code execution for LiveCodeBench** runs generated code against the full test suites in local subprocesses — run untrusted models in a container/VM.

## Tasks

45 registered tasks: GSM8K, MATH-500, Hendrycks MATH, AIME 2024/2025, OlympiadBench, TheoremQA, GPQA (diamond/main/extended), MMLU-Pro, MMLU-Redux 1&2, BBH, ARC, HellaSwag, WinoGrande, **HLE**, **LiveCodeBench** (release_v6), **MRCR** (2/4/8-needle) — each with official datasets and paper-faithful prompts. MCQ families come in three eval modes (`_generate`, `_logprob_token`, `_logprob_seq`). Image-bearing benchmarks (HLE, TheoremQA) run text-only by default; `modality: all` sends images to vision-capable server models, stores them deduplicated in the ledger, and shows them in the viewer.

See [`configs/TASKS.md`](configs/TASKS.md) for the full table.

## More

- **Multi-GPU**: `gpu_ids: [0, 1, 2]` runs one worker per GPU, models distributed round-robin, all appending to the same ledger.
- **veRL checkpoints**: `- verl:/path/to/run` expands to every `global_step_*` checkpoint — benchmark a whole training run in one config.
- **Python API**: `from lm_eval_ledger import RunConfig, run; run(RunConfig(models=[...], tasks=[...]))`.
- **Live streaming**: server/hf backends write each sample to the ledger as it completes; set `batch_size` to stream from vllm/sglang too — refresh the viewer mid-run and watch rows land.

## License

MIT
