# runner.py
"""Benchmark execution: backend-agnostic task/model loops and the
multi-GPU coordinator. Engine specifics live in lm_eval_ledger.backends."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path

from .backends import get_backend
from .config import RunConfig
from .db import DEFAULT_LEDGER_NAME, LedgerDatabase
from .fewshot import (
    build_fewshot_block,
    build_fewshot_block_logprob_seq,
    build_fewshot_block_logprob_token,
    build_fewshot_chat_messages,
    build_fewshot_chat_messages_logprob_seq,
    build_fewshot_chat_messages_logprob_token,
)
from .runlog import OutputLogger
from .tasks import get_task, get_available_tasks
from .tasks.base import load_from_hf


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 0:
        return "--:--:--"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    elif minutes > 0:
        return f"{minutes}m {secs:02d}s"
    else:
        return f"{secs}s"



def load_examples(task) -> list[dict]:
    """Load evaluation examples from HuggingFace (or the task's custom loader)."""
    if task.load_fn is not None:
        examples = task.load_fn()
        print(f"[INFO] Loaded {len(examples)} examples via custom loader ({task.name})")
        return examples
    examples = load_from_hf(task, split=task.hf_split)
    print(f"[INFO] Loaded {len(examples)} examples from HF ({task.hf_repo})")
    return examples


# ======================================================
# RUN SINGLE TASK
# ======================================================

def build_messages(task, example: dict, prompt_raw: str,
                   fewshot_chat_prefix: list[dict] | None) -> list[dict]:
    """The chat messages for one example: the task's own multi-turn
    messages, or few-shot user/assistant turns plus the test question,
    or a single user message wrapping the raw prompt."""
    if task.build_messages is not None:
        return task.build_messages(example)
    if fewshot_chat_prefix:
        test_question = task.build_prompt(example, "")
        return fewshot_chat_prefix + [{"role": "user", "content": test_question}]
    return [{"role": "user", "content": prompt_raw}]


def apply_chat_template(backend, task, example: dict, prompt_raw: str,
                        fewshot_chat_prefix: list[dict] | None,
                        enabled: bool) -> str:
    """Render one example through the model's chat template (if enabled).

    Falls back to the raw prompt when the backend cannot template
    client-side (server backends template server-side instead)."""
    if not enabled:
        return prompt_raw
    rendered = backend.apply_chat_template(
        build_messages(task, example, prompt_raw, fewshot_chat_prefix))
    return rendered if rendered is not None else prompt_raw


def _error_summary(cfg, task, fewshot_k, model_name, model_tag, timestamp,
                   error_msg: str) -> dict:
    return {
        "task": task.name,
        "fewshot_k": fewshot_k,
        "eval_mode": task.eval_mode,
        "model": model_name,
        "model_tag": model_tag,
        "error": error_msg,
        "pass_k": cfg.pass_k,
        "total_examples": 0,
        "correct": 0,
        "accuracy": 0.0,
        "no_answer_count": 0,
        "stop_reason_counts": {},
        "timestamp": timestamp,
        "settings": {
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "max_tokens": cfg.max_tokens,
        },
    }


def run_task(
    cfg: RunConfig,
    task_name: str,
    fewshot_k: int | None,
    backend,
    model_name: str,
    data_dir: Path,
    model_tag: str,
    timestamp: str,
) -> tuple[dict, list[dict]]:
    """
    Run a single benchmark task.

    Returns:
        tuple of (summary_dict, log_entries_list)
    """
    task_start = time.time()

    # ---------- get task ----------
    task = get_task(task_name)
    print(f"\n{'='*60}")
    print(f"[TASK] {task.name} - {task.description}")
    print(f"{'='*60}")

    # ---------- backend capability gate ----------
    if task.eval_mode not in backend.capabilities:
        fewshot_k = fewshot_k if fewshot_k is not None else task.default_fewshot_k
        error_msg = (f"eval mode '{task.eval_mode}' is not supported by "
                     f"backend '{backend.name}'; use one of: vllm, hf")
        print(f"[WARN] {error_msg}")
        return _error_summary(cfg, task, fewshot_k, model_name, model_tag,
                              timestamp, error_msg), []

    # ---------- load examples ----------
    eval_examples = load_examples(task)

    if not eval_examples:
        print(f"[WARN] No examples found for {task_name}, skipping")
        return {"task": task_name, "error": "No examples found"}, []

    if cfg.max_examples is not None:
        eval_examples = eval_examples[:cfg.max_examples]
        print(f"[INFO] Limited to {cfg.max_examples} examples")

    # ---------- build few-shot block ----------
    # Use provided fewshot_k, or fall back to task default if None
    if fewshot_k is None:
        fewshot_k = task.default_fewshot_k

    # Use appropriate few-shot format for each eval mode
    if task.eval_mode == "logprob_token":
        fewshot_block = build_fewshot_block_logprob_token(data_dir, task, fewshot_k)
    elif task.eval_mode == "logprob_seq":
        fewshot_block = build_fewshot_block_logprob_seq(data_dir, task, fewshot_k)
    else:
        fewshot_block = build_fewshot_block(data_dir, task, fewshot_k)
    print(f"[INFO] Few-shot examples: {fewshot_k}")
    print(f"[INFO] Eval mode: {task.eval_mode}")

    # Build structured few-shot chat messages for multi-turn chat template
    # (only used when apply_chat_template=true and fewshot_k > 0)
    fewshot_chat_prefix: list[dict] | None = None
    if cfg.apply_chat_template and fewshot_k > 0:
        if task.eval_mode == "logprob_token":
            fewshot_chat_prefix = build_fewshot_chat_messages_logprob_token(data_dir, task, fewshot_k)
        elif task.eval_mode == "logprob_seq":
            fewshot_chat_prefix = build_fewshot_chat_messages_logprob_seq(data_dir, task, fewshot_k)
        else:
            fewshot_chat_prefix = build_fewshot_chat_messages(data_dir, task, fewshot_k)
        if fewshot_chat_prefix:
            n_pairs = sum(1 for m in fewshot_chat_prefix if m["role"] == "assistant")
            print(f"[INFO] Using multi-turn chat format for {n_pairs} few-shot examples")

    # Common: build sample IDs and gold answers.
    # gold_answers feeds match_fn; when the task defines extract_gold_display,
    # the short display form goes in samples.gold and the full payload in
    # samples.gold_data.
    sample_ids: list[str] = []
    gold_answers: list[str] = []
    gold_displays: list[str | None] = []
    prompts_without_fewshot: list[str] = []
    for idx, ex in enumerate(eval_examples):
        sample_ids.append(str(ex.get("id", ex.get("idx", ex.get("index", idx)))))
        gold_answers.append(task.extract_gold(ex))
        gold_displays.append(
            task.extract_gold_display(ex) if task.extract_gold_display else None
        )
        prompts_without_fewshot.append(task.build_prompt(ex, ""))

    # ========================================
    # LOGPROB TOKEN EVALUATION PATH (first-token log-probability)
    # ========================================
    if task.eval_mode == "logprob_token" and task.choice_labels:
        choice_labels = task.choice_labels
        print(f"[INFO] Logprob MCQ mode: {len(choice_labels)} choices {choice_labels}")

        # Build one prompt per example (ending with "Answer:")
        prompts: list[str] = [
            apply_chat_template(backend, task, ex, task.build_prompt(ex, fewshot_block),
                                fewshot_chat_prefix, cfg.apply_chat_template)
            for ex in eval_examples
        ]

        print(f"[INFO] Built {len(prompts)} prompts (1 per example)")
        print(f"[INFO] Running logprob inference...")
        inference_start = time.time()
        all_logprobs = backend.first_token_logprobs(prompts, choice_labels,
                                                    seed=cfg.seed)
        inference_time = time.time() - inference_start
        print(f"[INFO] Logprob inference completed in {format_time(inference_time)}")

        log_entries: list[dict] = []
        correct_count = 0
        all_stop_reasons: list[str] = []

        for ex_idx, logprobs_dict in enumerate(all_logprobs):
            gold = gold_answers[ex_idx]
            # Pick the choice with highest logprob
            pred = max(logprobs_dict, key=logprobs_dict.get)
            is_correct = task.match_fn(gold, pred)
            if is_correct:
                correct_count += 1

            all_stop_reasons.append("logprob_token")

            # Store the per-choice logprobs dict as the response text (JSON)
            logprobs_json = json.dumps(logprobs_dict)
            log_entries.append({
                "sample_id": sample_ids[ex_idx],
                "prompt": prompts_without_fewshot[ex_idx],
                "prompt_full": prompts[ex_idx],
                "gold": gold_displays[ex_idx] or gold,
                "gold_data": gold if gold_displays[ex_idx] else None,
                "score": float(is_correct),
                "responses": [{"text": logprobs_json, "extracted": pred,
                               "stop_reason": "logprob_token",
                               "correct": float(is_correct)}],
            })

    # ========================================
    # LOGPROB SEQ EVALUATION PATH (completion log-likelihood)
    # ========================================
    elif task.eval_mode == "logprob_seq" and task.choice_labels and task.get_choice_texts:
        choice_labels = task.choice_labels
        print(f"[INFO] Logprob-seq (completion log-likelihood) mode: {len(choice_labels)} choices {choice_labels}")

        # Build base prompts and per-example choice texts
        base_prompts: list[str] = []
        choices_per_example: list[list[str]] = []
        for ex in eval_examples:
            base_prompts.append(apply_chat_template(
                backend, task, ex, task.build_prompt(ex, fewshot_block),
                fewshot_chat_prefix, cfg.apply_chat_template,
            ))
            choices_per_example.append(
                task.get_choice_texts(ex)[:len(choice_labels)])

        n_pairs = sum(len(c) for c in choices_per_example)
        print(f"[INFO] Scoring {n_pairs} (example, choice) pairs...")
        inference_start = time.time()
        all_scores = backend.score_completions(base_prompts, choices_per_example,
                                               seed=cfg.seed)
        inference_time = time.time() - inference_start
        print(f"[INFO] Logprob-seq inference completed in {format_time(inference_time)}")

        # Pick best choice per example and compare with gold
        log_entries: list[dict] = []
        correct_count = 0
        all_stop_reasons: list[str] = []

        for ex_idx in range(len(eval_examples)):
            gold = gold_answers[ex_idx]
            ex_scores = list(zip(choice_labels, all_scores[ex_idx]))

            if ex_scores:
                pred = max(ex_scores, key=lambda x: x[1])[0]
            else:
                pred = ""

            is_correct = task.match_fn(gold, pred)
            if is_correct:
                correct_count += 1

            all_stop_reasons.append("logprob_seq")

            # Store per-choice scores as the response text (JSON) for inspection
            scores_dict = {lbl: sc for lbl, sc in ex_scores}
            scores_json = json.dumps(scores_dict)

            log_entries.append({
                "sample_id": sample_ids[ex_idx],
                "prompt": prompts_without_fewshot[ex_idx],
                "prompt_full": base_prompts[ex_idx],
                "gold": gold_displays[ex_idx] or gold,
                "gold_data": gold if gold_displays[ex_idx] else None,
                "score": float(is_correct),
                "responses": [{"text": scores_json, "extracted": pred,
                               "stop_reason": "logprob_seq",
                               "correct": float(is_correct)}],
            })

    # ========================================
    # GENERATE EVALUATION PATH (default)
    # ========================================
    else:
        if cfg.pass_k > 1:
            print(f"[INFO] Pass@{cfg.pass_k} mode: generating {cfg.pass_k} responses per sample")

        # ---------- build prompts ----------
        print(f"[INFO] Building prompts...")

        prompts: list[str] = [
            apply_chat_template(backend, task, ex, task.build_prompt(ex, fewshot_block),
                                fewshot_chat_prefix, cfg.apply_chat_template)
            for ex in eval_examples
        ]
        # Server-side templating: pass structured messages instead
        use_messages = cfg.apply_chat_template and backend.prefers_messages
        messages_list: list[list[dict]] | None = None
        if use_messages:
            messages_list = [
                build_messages(task, ex, prompts_without_fewshot[i],
                               fewshot_chat_prefix)
                for i, ex in enumerate(eval_examples)
            ]

        # Drop prompts that don't fit the context window (the engine would
        # abort the whole task otherwise). Needs a backend tokenizer; server
        # backends skip this and rely on the server's own handling.
        if cfg.max_model_len is not None and backend.count_tokens(" ") is not None:
            budget = cfg.max_model_len - cfg.max_tokens
            keep = [i for i, p in enumerate(prompts)
                    if backend.count_tokens(p) <= budget]
            if len(keep) < len(prompts):
                print(f"[WARN] Skipping {len(prompts) - len(keep)} examples whose "
                      f"prompts exceed max_model_len - max_tokens = {budget} tokens")
                prompts = [prompts[i] for i in keep]
                eval_examples = [eval_examples[i] for i in keep]
                sample_ids = [sample_ids[i] for i in keep]
                gold_answers = [gold_answers[i] for i in keep]
                gold_displays = [gold_displays[i] for i in keep]
                prompts_without_fewshot = [prompts_without_fewshot[i] for i in keep]
                if messages_list is not None:
                    messages_list = [messages_list[i] for i in keep]

            if not prompts:
                # Record an error rather than a misleading 0.0 accuracy over
                # zero examples.
                error_msg = (f"All examples exceed the context budget "
                             f"(max_model_len {cfg.max_model_len} - "
                             f"max_tokens {cfg.max_tokens} = {budget} tokens); "
                             f"raise max_model_len")
                print(f"[WARN] {error_msg}")
                return {
                    "task": task.name,
                    "fewshot_k": fewshot_k,
                    "eval_mode": task.eval_mode,
                    "model": model_name,
                    "model_tag": model_tag,
                    "error": error_msg,
                    "pass_k": cfg.pass_k,
                    "total_examples": 0,
                    "correct": 0,
                    "accuracy": 0.0,
                    "no_answer_count": 0,
                    "stop_reason_counts": {},
                    "timestamp": timestamp,
                    "settings": {
                        "temperature": cfg.temperature,
                        "top_p": cfg.top_p,
                        "max_tokens": cfg.max_tokens,
                    },
                }, []

        print(f"[INFO] Running batch inference on {len(prompts)} examples...")
        inference_start = time.time()
        gen_kwargs = dict(
            temperature=cfg.temperature, top_p=cfg.top_p,
            max_tokens=cfg.max_tokens,
            stop=task.stop_strings if task.stop_strings else None,
            n=cfg.pass_k, seed=cfg.seed, batch_size=cfg.batch_size,
        )
        if use_messages:
            outputs = backend.chat_generate(messages_list, **gen_kwargs)
        else:
            outputs = backend.generate(prompts, **gen_kwargs)
        inference_time = time.time() - inference_start
        print(f"[INFO] Batch inference completed in {format_time(inference_time)}")

        # Process outputs and build log entries
        log_entries: list[dict] = []
        correct_count = 0
        all_stop_reasons: list[str] = []

        for idx, (all_responses, ex) in enumerate(zip(outputs, eval_examples)):
            gold = gold_answers[idx]
            # match_fn may return bool (binary tasks) or a float score in [0, 1]
            # (partial-credit tasks like MRCR); pass@k keeps the best score.
            best_score = 0.0
            responses_list = []

            for result in all_responses:  # GenResult
                pred_raw = result.text
                finish_reason = result.finish_reason or ""
                stop_reason = result.stop_reason
                if stop_reason:
                    stop_reason_str = f"{finish_reason}:{stop_reason}"
                else:
                    stop_reason_str = f"{finish_reason}:-"

                pred = task.extract_pred(pred_raw)
                score = float(task.match_fn(gold, pred))

                responses_list.append({
                    "text": pred_raw,
                    "extracted": pred,
                    "stop_reason": stop_reason_str,
                    "correct": score,
                })
                all_stop_reasons.append(stop_reason_str)

                best_score = max(best_score, score)

            correct_count += best_score

            log_entries.append({
                "sample_id": sample_ids[idx],
                "prompt": prompts_without_fewshot[idx],
                "prompt_full": prompts[idx],
                "gold": gold_displays[idx] or gold,
                "gold_data": gold if gold_displays[idx] else None,
                "score": best_score,
                "responses": responses_list,
            })

    # ---------- sort entries ----------
    try:
        log_entries.sort(key=lambda x: int(x['sample_id']))
    except (ValueError, TypeError):
        log_entries.sort(key=lambda x: x['sample_id'])

    # ---------- compute summary ----------
    duration = time.time() - task_start
    # total_examples is the number of unique samples
    total = len(eval_examples)
    correct = correct_count
    acc = correct / total if total > 0 else 0.0

    # Count how many times the model didn't follow the answer format
    # (empty extracted answer on the first response of each sample)
    no_answer_count = sum(
        1 for entry in log_entries
        if entry["responses"] and entry["responses"][0].get("extracted") == ""
    )

    # Count occurrences of each stop_reason across all responses
    stop_reason_counts = dict(Counter(all_stop_reasons))

    summary = {
        "model": model_name,
        "model_tag": model_tag,
        "task": task.name,
        "fewshot_k": fewshot_k,
        "eval_mode": task.eval_mode,
        "pass_k": cfg.pass_k,
        "total_examples": total,
        "correct": correct,
        "no_answer_count": no_answer_count,
        "stop_reason_counts": stop_reason_counts,
        "accuracy": acc,
        "duration_seconds": duration,
        "duration_human": format_time(duration),
        "timestamp": timestamp,
        "settings": {
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "max_tokens": cfg.max_tokens,
            "max_examples": cfg.max_examples,
        }
    }

    # ---------- print results ----------
    print(f"[{task.name}({fewshot_k})] Accuracy: {correct:g}/{total} = {acc:.4f} | Duration: {format_time(duration)}")

    return summary, log_entries


# ======================================================
# RUN ALL TASKS FOR ONE MODEL
# ======================================================

def run_model(
    cfg: RunConfig,
    model_name: str,
    tasks_to_run: list[tuple[str, int | None]],
    data_dir: Path,
    timestamp: str,
    ledger: LedgerDatabase,
    run_id: int,
    backend,
) -> dict:
    """Run all tasks for a single model and return combined summary."""
    model_start = time.time()
    model_tag = Path(model_name).name

    # If model_tag is just "huggingface" (veRL checkpoint), use parent dir info
    # GRPO: .../run_name/global_step_200/actor/huggingface -> "run_name_global_step_200"
    # SFT:  .../run_name/global_step_600/huggingface       -> "run_name_global_step_600"
    if model_tag == "huggingface":
        model_path = Path(model_name)
        parent = model_path.parent
        if parent.name == "actor" and parent.parent.name.startswith("global_step"):
            # GRPO layout: huggingface -> actor -> global_step_XXX -> run_name
            step_dir = parent.parent.name
            run_dir = parent.parent.parent.name
            model_tag = f"{run_dir}_{step_dir}"
        elif parent.name.startswith("global_step"):
            # SFT layout: huggingface -> global_step_XXX -> run_name
            step_dir = parent.name
            run_dir = parent.parent.name
            model_tag = f"{run_dir}_{step_dir}"

    print(f"\n{'#'*60}")
    print(f"# MODEL: {model_tag}")
    print(f"{'#'*60}")

    # ---------- load model ----------
    print(f"\n[INFO] Loading model: {model_name} (backend: {backend.name})")

    # Determine quantization for this model
    quantization = cfg.quantization_for(model_tag)

    if quantization:
        print(f"[INFO] Using quantization: {quantization}")
    else:
        print(f"[INFO] Using model default precision (no quantization override)")

    backend.load(model_name, cfg, quantization=quantization)

    # ---------- run all tasks and collect results ----------
    task_summaries: list[dict] = []

    for i, (task_name, fewshot_k) in enumerate(tasks_to_run, 1):
        fewshot_str = f" (fewshot={fewshot_k})" if fewshot_k is not None else " (default fewshot)"
        print(f"\n[{i}/{len(tasks_to_run)}] Running {task_name}{fewshot_str}...")

        try:
            summary, log_entries = run_task(
                cfg=cfg,
                task_name=task_name,
                fewshot_k=fewshot_k,
                backend=backend,
                model_name=model_name,
                data_dir=data_dir,
                model_tag=model_tag,
                timestamp=timestamp,
            )
        except Exception as e:
            print(f"\n[ERROR] Task {task_name} failed: {e}")
            traceback.print_exc()
            summary = {
                "task": task_name,
                "fewshot_k": fewshot_k,
                "model": model_name,
                "model_tag": model_tag,
                "error": str(e),
                "pass_k": cfg.pass_k,
                "total_examples": 0,
                "correct": 0,
                "accuracy": 0.0,
                "no_answer_count": 0,
                "stop_reason_counts": {},
                "timestamp": timestamp,
                "settings": {
                    "temperature": cfg.temperature,
                    "top_p": cfg.top_p,
                    "max_tokens": cfg.max_tokens,
                },
            }
            log_entries = []
            print(f"[INFO] Continuing to next task...")

        task_summaries.append(summary)

        # Append to the ledger
        benchmark_id = ledger.add_benchmark(run_id, summary)
        if log_entries:
            ledger.add_samples(benchmark_id, log_entries)

    # ---------- unload model to free resources ----------
    print(f"\n[INFO] Unloading model: {model_tag}")
    backend.unload()

    # ---------- compute model summary ----------
    model_duration = time.time() - model_start

    model_summary = {
        "model": model_name,
        "model_tag": model_tag,
        "timestamp": timestamp,
        "duration_seconds": model_duration,
        "duration_human": format_time(model_duration),
        "total_examples": sum(s.get("total_examples", 0) for s in task_summaries if "error" not in s),
        "tasks": task_summaries,
    }

    # ---------- print model summary ----------
    print(f"\n{'='*60}")
    print(f"MODEL COMPLETE: {model_tag}")
    print(f"{'='*60}")
    print(f"Duration: {format_time(model_duration)}")
    print(f"Results:")
    for summary in task_summaries:
        label = f"{summary['task']}({summary.get('fewshot_k')})"
        if "error" in summary:
            print(f"  {label}: ERROR - {summary['error']}")
        else:
            print(f"  {label}: {summary['accuracy']:.4f} ({summary['correct']:g}/{summary['total_examples']})")

    return model_summary


# ======================================================
# MAIN
# ======================================================

def _make_run_name(cfg: RunConfig, num_models: int, num_tasks: int) -> str:
    """Build the run name used for the DB and log files."""
    examples_str = "N" if cfg.max_examples is None else str(cfg.max_examples)
    db_timestamp = datetime.now().strftime("%m-%d_%H%M")
    pass_k_str = f"-P{cfg.pass_k}" if cfg.pass_k > 1 else ""
    return f"{db_timestamp}_{num_models}M-{num_tasks}T-{examples_str}E{pass_k_str}"


def _resolve_tasks_to_run(cfg: RunConfig) -> list[tuple[str, int | None]]:
    """Config tasks, or ALL registered tasks with defaults when none configured."""
    if cfg.tasks:
        return cfg.tasks
    return [(task, None) for task in get_available_tasks()]


def _ledger_path(cfg: RunConfig) -> Path:
    """The ledger DB file: cfg.db_path, or <results_dir>/ledger.sqlite3."""
    if cfg.db_path:
        return Path(cfg.db_path)
    return Path(cfg.results_dir) / DEFAULT_LEDGER_NAME


def _harness_version() -> str:
    from . import __version__
    return __version__


def _prefetch_verifier(cfg: RunConfig) -> None:
    """Download the verifier model BEFORE benchmarking, so a typo'd name or
    an undownloadable model surfaces immediately instead of after hours."""
    if not cfg.verifier_model or Path(cfg.verifier_model).exists():
        return
    from huggingface_hub import snapshot_download
    print(f"[INFO] Prefetching verifier model {cfg.verifier_model} ...")
    try:
        snapshot_download(cfg.verifier_model)
        print(f"[INFO] Verifier model ready")
    except Exception as e:
        print(f"[WARN] Could not prefetch verifier model {cfg.verifier_model}: {e}")
        print(f"[WARN] The post-run verification pass will likely fail; "
              f"benchmark results are unaffected either way")


def _maybe_verify(cfg: RunConfig, db_path: Path | None, run_id: int | None) -> None:
    """Run the post-run LLM verification pass if configured.

    Never raises: a verifier failure must not eat the benchmark run's
    final output - results can always be re-verified with
    `lm-eval-ledger-verify`.
    """
    if not cfg.verifier_model or db_path is None or run_id is None:
        return
    from .verifier import verify_run
    try:
        verify_run(
            db_path, cfg.verifier_model,
            run_id=run_id,
            mode=cfg.verifier_mode,
            max_model_len=cfg.verifier_max_model_len,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            enforce_eager=cfg.enforce_eager,
            seed=cfg.seed,
        )
    except Exception as e:
        print(f"\n[ERROR] Verifier pass failed: {e}")
        traceback.print_exc()
        print(f"[INFO] Benchmark results are unaffected. Re-run the pass with:\n"
              f"       lm-eval-ledger-verify {db_path} --run {run_id} "
              f"--model {cfg.verifier_model}")


def _run_coordinator(cfg: RunConfig) -> Path:
    """Multi-GPU coordinator: spawn parallel workers and collect results."""
    total_start = time.time()
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(exist_ok=True, parents=True)
    logs_dir = Path(cfg.logs_dir)
    logs_dir.mkdir(exist_ok=True, parents=True)

    # Determine tasks (for run_name generation and header display)
    tasks_to_run = _resolve_tasks_to_run(cfg)

    # Generate shared run name (all workers append to the same run row)
    run_name = _make_run_name(cfg, len(cfg.models), len(tasks_to_run))
    ledger_path = _ledger_path(cfg)

    gpu_ids = [str(g) for g in cfg.gpu_ids]
    num_workers = min(len(gpu_ids), len(cfg.models))

    # Pre-create the ledger and the run row before workers connect. The run
    # row's resolved config is the reproducibility artifact; workers load
    # their config from it (retrievable later via `lm-eval-ledger config`).
    ledger = LedgerDatabase(ledger_path)
    run_id = ledger.create_run(run_name, cfg.to_yaml(), _harness_version())
    ledger.close()

    # ---------- print header ----------
    task_strs = [f"{name}({k})" if k is not None else name for name, k in tasks_to_run]
    print(f"{'='*60}")
    print(f"LLM BENCHMARK RUNNER - MULTI-GPU ({num_workers} GPUs)")
    print(f"{'='*60}")
    print(f"Models: {len(cfg.models)} (distributed across {num_workers} GPUs)")
    for i, m in enumerate(cfg.models):
        print(f"  GPU {gpu_ids[i % num_workers]}: {Path(m).name}")
    print(f"Tasks: {', '.join(task_strs)}")
    print(f"Ledger: {ledger_path} (run {run_name}, id {run_id})")
    print(f"{'='*60}")

    # Fail fast on an unavailable verifier before hours of benchmarking
    _prefetch_verifier(cfg)

    # ---------- spawn worker processes ----------
    processes = []
    for worker_id in range(num_workers):
        gpu_id = gpu_ids[worker_id]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        cmd = [
            sys.executable, "-m", "lm_eval_ledger.cli",
            "--db-path", str(ledger_path),
            "--shard", f"{worker_id}/{num_workers}",
            "--run-name", run_name,
        ]
        print(f"\n[COORDINATOR] Spawning worker {worker_id} on GPU {gpu_id} ...")
        proc = subprocess.Popen(cmd, env=env)
        processes.append((worker_id, gpu_id, proc))

    # ---------- wait for all workers ----------
    print(f"\n[COORDINATOR] Waiting for {num_workers} workers to finish ...")
    failed = []
    for worker_id, gpu_id, proc in processes:
        rc = proc.wait()
        if rc != 0:
            print(f"[COORDINATOR] Worker {worker_id} (GPU {gpu_id}) FAILED (exit code {rc})")
            failed.append(worker_id)
        else:
            print(f"[COORDINATOR] Worker {worker_id} (GPU {gpu_id}) completed successfully")

    # ---------- print combined summary from DB ----------
    total_duration = time.time() - total_start

    print(f"\n{'#'*60}")
    print(f"# ALL BENCHMARKS COMPLETE ({num_workers} GPUs)")
    print(f"{'#'*60}")
    print(f"Total duration: {format_time(total_duration)}")

    ledger = LedgerDatabase(ledger_path)
    rows = ledger.conn.execute(
        "SELECT task, fewshot_k, model_tag, accuracy, total_examples, correct, error "
        "FROM benchmarks WHERE run_id = ? ORDER BY benchmark_id", (run_id,)
    ).fetchall()

    if rows:
        print(f"\nResults by model:")
        print(f"{'-'*60}")
        current_model = None
        for row in rows:
            if row["model_tag"] != current_model:
                current_model = row["model_tag"]
                print(f"\n{current_model}:")
            label = f"{row['task']}({row['fewshot_k']})"
            if row["error"]:
                print(f"  {label}: ERROR - {row['error']}")
            else:
                print(f"  {label}: {row['accuracy']:.4f} ({row['correct']:g}/{row['total_examples']})")

    # All workers have exited, so it's safe to clean up the WAL/SHM files
    ledger.close(remove_sidecars=True)

    if failed:
        print(f"\n[WARN] {len(failed)} worker(s) failed: {failed}")

    # Optional LLM verification pass (workers are done; pin to first GPU)
    if cfg.verifier_model and cfg.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_ids[0])
    _maybe_verify(cfg, ledger_path, run_id)

    print(f"\nResults appended to ledger: {ledger_path} (run id {run_id})")
    print(f"Logs saved to: {logs_dir}")
    print(f"{'#'*60}")
    return ledger_path


def run(cfg: RunConfig, *, shard: str | None = None, run_name: str | None = None) -> Path:
    """Run all configured benchmarks and return the results database path.

    This is the library entry point: build a RunConfig and call run(cfg).
    shard and run_name are internal parameters for multi-GPU worker processes.
    """
    # Multi-GPU coordinator mode: gpu_ids has 2+ GPUs and not already a worker
    if cfg.gpu_ids and len(cfg.gpu_ids) > 1 and shard is None:
        return _run_coordinator(cfg)

    # ---------- single-GPU / worker mode ----------

    # Pin to specific GPU if gpu_ids has exactly one entry (e.g., gpu_ids: [7])
    if cfg.gpu_ids and len(cfg.gpu_ids) == 1 and shard is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_ids[0])
        print(f"[INFO] Pinning to GPU {cfg.gpu_ids[0]} (CUDA_VISIBLE_DEVICES={cfg.gpu_ids[0]})")

    total_start = time.time()

    # ---------- determine model subset for this worker ----------
    models = cfg.models
    shard_idx = None
    num_shards = None
    if shard is not None:
        shard_idx, num_shards = (int(x) for x in shard.split("/"))
        models = cfg.models[shard_idx::num_shards]
        if not models:
            print(f"[WORKER {shard_idx}] No models assigned, exiting.")
            return None

    # ---------- setup paths ----------
    data_dir = Path(cfg.data_dir)
    timestamp = datetime.now().strftime("%m%d_%H%M")

    # ---------- setup results directory ----------
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(exist_ok=True, parents=True)

    # ---------- determine tasks to run ----------
    tasks_to_run = _resolve_tasks_to_run(cfg)

    # ---------- open the ledger and register the run ----------
    if run_name is None:
        # Standalone mode: generate run name (workers get it from the coordinator)
        run_name = _make_run_name(cfg, len(models), len(tasks_to_run))

    ledger_path = _ledger_path(cfg)
    ledger = LedgerDatabase(ledger_path)

    if shard is None:
        # Standalone mode: register the run; the run row's resolved config
        # is the reproducibility artifact (`lm-eval-ledger config <id>`).
        run_id = ledger.create_run(run_name, cfg.to_yaml(), _harness_version())
    else:
        # Worker mode: the coordinator already registered the run.
        run_id = ledger.get_run_id(run_name)
        if run_id is None:
            raise RuntimeError(f"Run {run_name!r} not found in ledger {ledger_path}")

    # ---------- setup output logging ----------
    logs_dir = Path(cfg.logs_dir)
    logs_dir.mkdir(exist_ok=True, parents=True)
    # Suffix log files with shard ID to avoid collisions in multi-GPU mode
    log_name = f"{run_name}_gpu{shard_idx}" if shard_idx is not None else run_name
    output_logger = OutputLogger(logs_dir, log_name)
    output_logger.start()

    try:
        if shard is None:
            _prefetch_verifier(cfg)
        _run_models(cfg, models, tasks_to_run, data_dir, timestamp, ledger,
                    run_id, shard_idx, num_shards, total_start)
        # Optional LLM verification pass, inside the logging scope so its
        # output (and any failure) lands in the run's logs. Standalone mode
        # only; the coordinator runs it once for multi-GPU runs.
        if shard is None:
            _maybe_verify(cfg, ledger_path, run_id)
    finally:
        # In multi-GPU mode other workers may still hold the ledger open;
        # only the last close of a run may delete the WAL/SHM sidecars.
        ledger.close(remove_sidecars=shard is None)
        output_logger.stop()

    print(f"\nResults appended to ledger: {ledger_path} (run id {run_id})")
    print(f"Logs saved to: {logs_dir / log_name}_stdout.log and _combined.log")
    print(f"{'#'*60}")
    return ledger_path

def _run_models(
    cfg: RunConfig,
    models: list[str],
    tasks_to_run: list[tuple[str, int | None]],
    data_dir: Path,
    timestamp: str,
    ledger: LedgerDatabase,
    run_id: int,
    shard_idx: int | None,
    num_shards: int | None,
    total_start: float,
) -> None:
    """Run all models/tasks for this process and write results to the DB."""
    # ---------- print header ----------
    worker_str = f" (worker {shard_idx}/{num_shards})" if shard_idx is not None else ""
    print(f"{'='*60}")
    print(f"LLM BENCHMARK RUNNER{worker_str}")
    print(f"{'='*60}")
    print(f"Models: {len(models)}")
    for m in models:
        print(f"  - {Path(m).name}")
    # Format tasks with their fewshot values for display
    task_strs = [f"{name}({k})" if k is not None else name for name, k in tasks_to_run]
    print(f"Tasks: {', '.join(task_strs)}")
    print(f"Timestamp: {timestamp}")
    print(f"Ledger: {ledger.db_path} (run id {run_id})")
    print(f"{'='*60}")

    # ---------- run all models ----------
    backend = get_backend(cfg.backend)
    all_model_summaries: list[dict] = []

    for i, model_name in enumerate(models, 1):
        print(f"\n[MODEL {i}/{len(models)}]")
        try:
            model_summary = run_model(
                cfg=cfg,
                model_name=model_name,
                tasks_to_run=tasks_to_run,
                data_dir=data_dir,
                timestamp=timestamp,
                ledger=ledger,
                run_id=run_id,
                backend=backend,
            )
            all_model_summaries.append(model_summary)
        except Exception as e:
            print(f"\n[ERROR] Model {model_name} failed: {e}")
            print(f"[INFO] Continuing to next model...")
            traceback.print_exc()
            # Add error summary for this model
            all_model_summaries.append({
                "model": model_name,
                "model_tag": Path(model_name).name,
                "timestamp": timestamp,
                "error": str(e),
                "tasks": [],
            })

    # ---------- print final summary ----------
    total_duration = time.time() - total_start

    print(f"\n{'#'*60}")
    print(f"# BENCHMARKS COMPLETE{worker_str}")
    print(f"{'#'*60}")
    print(f"Duration: {format_time(total_duration)}")
    print(f"\nResults by model:")
    print(f"{'-'*60}")

    for model_summary in all_model_summaries:
        print(f"\n{model_summary['model_tag']}:")
        for task_summary in model_summary["tasks"]:
            label = f"{task_summary.get('task')}({task_summary.get('fewshot_k')})"
            if "error" in task_summary:
                print(f"  {label}: ERROR")
            else:
                print(f"  {label}: {task_summary['accuracy']:.4f}")
