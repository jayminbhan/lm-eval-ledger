# lm-eval-ledger


**lm-eval-ledger** is a benchmark harness designed to benchmark LLMs and easily navigate and compare model output.
This project logs all model outputs on a sqlite database and provide navigation app that lets you easily look at each model generation, compare score, compare model answer across different samples! 

Interactive sample navigation demo: [demo app](demoapp.link)

Features:
- YAML-driven LLM benchmark harness where you can benchmark across many models x tasks in one run
- Sqlite logging of every sample model generates of benchmark and native web app that lets you navigate model generation with easy compare and delete
- Support various backends such as vllm, sglang, hf accelerate and api(openai, llama.cpp)
- Interactive interaction for config selection if not provided such as thinking mode, context etc
- Easily add custom tasks to log all model generation and naviagate model generation

Available Tasks: [TASKS.md](TASKS.md)

<p align="center">
  <img src="src/lm_eval_ledger/webapp/static/sample-inspection.png" width="100%" alt="Sample Inspection">
</p>

<table>
  <tr>
    <td width="50%"><img src="src/lm_eval_ledger/webapp/static/leaderboard.png" alt="Leaderboard"></td>
    <td width="50%"><img src="src/lm_eval_ledger/webapp/static/run-history.png" alt="Run History"></td>
  </tr>
  <tr>
    <td><b>Leaderboard</b> — Benchmark results. One click deduplication</td>
    <td><b>Run History</b> — every run with its config yaml, timing, and status.</td>
  </tr>
</table>



## Install

```bash
pip install lm-eval-ledger
```

Inference backends are optional extras — pick the one you'll actually use:

```bash
pip install "lm-eval-ledger[vllm]"     # vLLM
pip install "lm-eval-ledger[sglang]"   # SGLang
pip install "lm-eval-ledger[hf]"       # HF transformers (slow, but always works)
```

## Quickstart

```bash
lm-eval-ledger init                                 # annotated starter config + results/logs dirs
lm-eval-ledger -c bench.yaml                        # run benchmarks from a YAML config
lm-eval-ledger serve --db results/ledger.sqlite3    # browse results at http://localhost:8090
```

`init` drops `reference.yaml` - a fully annotated config that doubles
as the field manual and, as shipped, a 20-example smoke run - into the
current directory (never overwrites).

See [Configuration](#configuration) for the YAML format.


## Reference yaml
```yaml
reference yaml code example..
```


## Backends

One config format, four engines — pick at install time:

| backend | install | OS | notes |
|---|---|---|---|
| `vllm` | `pip install lm-eval-ledger[vllm]` | Linux | in-process, fastest |
| `hf` | `pip install lm-eval-ledger[hf]`  | Linux, Windows | transformers + accelerate; every architecture; slow |
| `sglang` | `pip install lm-eval-ledger[sglang]`  | Linux | in-process sglang.Engine |
| `api` | `pip install lm-eval-ledger` *(no extra)*  | Linux, Windows | any OpenAI-compatible endpoint (ex. llama.cpp, ollama or hosted APIs) |


Benchmarking with llama.cpp example:

```bash
llama-server -hf unsloth/Qwen3-30B-GGUF:Q4_K_XL -ngl 999 -c 65536 -np 4 --jinja
```

```yaml
backend: server
server_url: http://localhost:8080/v1
server_concurrency: 4        # match the server's -np slots
```


## Thinking-mode control
Provide thinking mode in yaml config. When left unset, the harness reads each model's template (from the HF cache, or a llama.cpp server's `/props`), detects the knobs, offers a menu, and prints the YAML to make your choice permanent.`{}` opts out.

```yaml
models:
  - name: Qwen/Qwen3-8B
    chat_template_kwargs: {enable_thinking: true, reasoning_effort: high}
  - google/gemma-3-12b-it        # no thinking knob; global settings apply
```
