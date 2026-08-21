# cli.py
"""
LLM Benchmark Runner

Usage:
    lm-eval-ledger                        # settings from ./bench.yaml
    lm-eval-ledger -c my_run.yaml         # explicit config file
    lm-eval-ledger -c my_run.yaml --max-examples 10   # flag overrides file
    lm-eval-ledger --model Qwen/Qwen2.5-0.5B-Instruct --task gsm8k_main:0
    lm-eval-ledger --help                 # all flags

Settings precedence: RunConfig defaults < YAML config < CLI flags (see config.py).

Available tasks:
    - gsm8k_main: GSM8K Main version
    - gsm8k_socratic: GSM8K Socratic version
    - aime_2024: AIME 2024
    - aime_2025: AIME 2025
    - math500: MATH-500
    - hendrycks_math: Hendrycks MATH (7 subjects)
    - mmlu_redux_2_generate / _logprob_token / _logprob_seq: MMLU Redux 2.0
    - mmlu_redux_1_generate / _logprob_token / _logprob_seq: MMLU Redux 1.0
    - mmlu_pro_generate / _logprob_token / _logprob_seq: MMLU-Pro (up to 10 options)
    - arc_easy_generate / _logprob_token / _logprob_seq: ARC-Easy
    - arc_challenge_generate / _logprob_token / _logprob_seq: ARC-Challenge
    - hellaswag_generate / _logprob_token / _logprob_seq: HellaSwag
    - winogrande_generate / _logprob_token / _logprob_seq: WinoGrande
    - olympiad_bench_math_en: OlympiadBench Math (English)
    - olympiad_bench_physics_en: OlympiadBench Physics (English)
    - theoremqa: TheoremQA (text-only)
    - gpqa_diamond_generate / _logprob_token / _logprob_seq: GPQA Diamond (198)
    - gpqa_main_generate / _logprob_token / _logprob_seq: GPQA Main (448)
    - gpqa_extended_generate / _logprob_token / _logprob_seq: GPQA Extended (546)
    - bbh: BIG-Bench Hard (27 subtasks)
"""

# TODO: add IFEval, HumanEval/MBPP
from __future__ import annotations


import gc
import json
import os
import sqlite3
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel

from .config import RunConfig, build_arg_parser, resolve_config
from .tasks import get_task, get_available_tasks
from .tasks.base import load_from_hf, load_jsonl


os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
# ======================================================
# PROGRESS TRACKING
# ======================================================

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


# ======================================================
# CONSOLIDATED SQLITE DATABASE
# ======================================================

class BenchmarkDatabase:
    """
    Consolidated SQLite database for all benchmark data.

    Tables:
    - summaries: Per-task accuracy, timing, and settings
    - results: Individual example results (one row per sample, with response_1..response_k columns)
    - (gpu_metrics table removed)
    """

    def __init__(self, db_path: Path, pass_k: int = 1):
        self.db_path = db_path
        self.pass_k = pass_k
        # Use autocommit mode (isolation_level=None) for immediate writes
        # busy_timeout lets concurrent writers retry instead of failing immediately
        self.conn = sqlite3.connect(db_path, isolation_level=None, timeout=60)
        self.conn.row_factory = sqlite3.Row
        # WAL mode allows concurrent reads + writes from multiple processes
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self):
        """Create all tables with proper schema."""
        cursor = self.conn.cursor()

        # Summaries table - per-task results with all metadata
        # stop_reason_counts is JSON: {"stop:-": 450, "stop:####": 50, "length:-": 10}
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_name TEXT NOT NULL,
                model_tag TEXT NOT NULL,
                total_examples INTEGER,
                correct INTEGER,
                accuracy REAL,
                no_answer_count INTEGER,
                stop_reason_counts TEXT,
                duration_human TEXT,
                pass_k INTEGER,
                temperature REAL,
                top_p REAL,
                max_tokens INTEGER,
                error TEXT,
                model TEXT NOT NULL
            )
        """)

        # Results table - individual example results (one row per sample)
        # For pass@k, responses are stored in response_1, response_2, ... response_k columns
        # extracted_answers and stop_reasons are JSON arrays with all k values
        response_columns = []
        for i in range(1, self.pass_k + 1):
            response_columns.append(f"response_{i} TEXT")
        response_columns_sql = ",\n                ".join(response_columns)

        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                task_name TEXT NOT NULL,
                sample_id TEXT,
                prompt_actual TEXT,
                prompt_full TEXT,
                gold_answer TEXT,
                is_correct INTEGER,
                extracted_answers TEXT,
                stop_reasons TEXT,
                {response_columns_sql}
            )
        """)

        # Create indexes for common queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_summaries_model ON summaries(model_tag)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_summaries_task ON summaries(task_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_results_model ON results(model_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_results_task ON results(task_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_results_correct ON results(is_correct)")

        self.conn.commit()

    def add_summary(self, summary: dict):
        """Add a task summary record."""
        cursor = self.conn.cursor()
        settings = summary.get("settings", {})
        # Convert stop_reason_counts dict to JSON string
        stop_reason_counts = summary.get("stop_reason_counts", {})
        stop_reason_counts_json = json.dumps(stop_reason_counts) if stop_reason_counts else "{}"
        cursor.execute("""
            INSERT INTO summaries (
                task_name, model_tag, total_examples, correct, accuracy,
                no_answer_count, stop_reason_counts, duration_human,
                pass_k, temperature, top_p, max_tokens, error, model
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            summary.get("task", ""),
            summary.get("model_tag", ""),
            summary.get("total_examples", 0),
            summary.get("correct", 0),
            summary.get("accuracy", 0.0),
            summary.get("no_answer_count", 0),
            stop_reason_counts_json,
            summary.get("duration_human", ""),
            summary.get("pass_k", 1),
            settings.get("temperature"),
            settings.get("top_p"),
            settings.get("max_tokens"),
            summary.get("error"),
            summary.get("model", ""),
        ))
        self.conn.commit()

    def add_results(self, task_name: str, model_name: str, entries: list[dict]):
        """Add multiple result entries for a task.

        Each entry should have:
        - sample_id, prompt_actual, prompt_full, gold_answer, is_correct (overall pass@k result)
        - extracted_answers: list of extracted answers for each k (stored as JSON)
        - stop_reasons: list of stop reasons for each k (stored as JSON)
        - responses: list of response strings for each k
        """
        # Build column names and placeholders dynamically based on pass_k
        base_columns = ["model_name", "task_name", "sample_id", "prompt_actual", "prompt_full",
                        "gold_answer", "is_correct", "extracted_answers", "stop_reasons"]
        response_columns = [f"response_{i}" for i in range(1, self.pass_k + 1)]

        all_columns = base_columns + response_columns
        placeholders = ", ".join(["?"] * len(all_columns))
        columns_sql = ", ".join(all_columns)

        rows = []
        for entry in entries:
            responses = entry.get("responses", [])
            # Pad responses so every response_i column gets a value
            response_values = [
                responses[i] if i < len(responses) else ""
                for i in range(self.pass_k)
            ]
            rows.append((
                model_name,
                task_name,
                entry.get("sample_id", ""),
                entry.get("prompt_actual", ""),
                entry.get("prompt_full", ""),
                entry.get("gold_answer", ""),
                1 if entry.get("is_correct") else 0,
                json.dumps(entry.get("extracted_answers", [])),
                json.dumps(entry.get("stop_reasons", [])),
                *response_values,
            ))

        self.conn.executemany(
            f"INSERT INTO results ({columns_sql}) VALUES ({placeholders})", rows
        )
        self.conn.commit()

    def save_run_config(self, config_yaml: str):
        """Record the resolved run configuration (as YAML text) in the database."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS run_config (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                config_yaml TEXT NOT NULL
            )
        """)
        self.conn.execute(
            "INSERT INTO run_config (created_at, config_yaml) VALUES (?, ?)",
            (datetime.now().isoformat(timespec="seconds"), config_yaml),
        )
        self.conn.commit()

    def close(self, remove_sidecars: bool = False):
        """Close the database connection.

        Args:
            remove_sidecars: Also delete the -wal/-shm files. Only safe when no
                other process still has the database open (e.g., the last close
                of a run) - deleting a live WAL file can corrupt the database.
        """
        try:
            self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.OperationalError:
            pass  # Another writer is active; SQLite will checkpoint later
        self.conn.close()
        if remove_sidecars:
            for suffix in ("-wal", "-shm"):
                Path(str(self.db_path) + suffix).unlink(missing_ok=True)


# ======================================================
# OUTPUT LOGGING (TEE TO FILES)
# ======================================================

class TeeWriter:
    """Write to multiple destinations (file + original stream)."""

    def __init__(self, original, *files):
        self.original = original
        self.files = files

    def write(self, data):
        self.original.write(data)
        for f in self.files:
            f.write(data)
            f.flush()

    def flush(self):
        self.original.flush()
        for f in self.files:
            f.flush()

    def fileno(self):
        """Return file descriptor of the original stream (needed by vLLM)."""
        return self.original.fileno()

    def isatty(self):
        """Return whether original stream is a tty."""
        return self.original.isatty()


class OutputLogger:
    """
    Capture stdout and stderr to log files.

    Creates two log files:
    - stdout_only.log: Only stdout
    - combined.log: Both stdout and stderr
    """

    def __init__(self, logs_dir: Path, timestamp: str):
        self.logs_dir = logs_dir
        self.timestamp = timestamp
        self._stdout_file = None
        self._combined_file = None
        self._original_stdout = None
        self._original_stderr = None

    def start(self):
        self.logs_dir.mkdir(exist_ok=True, parents=True)

        stdout_path = self.logs_dir / f"{self.timestamp}_stdout.log"
        combined_path = self.logs_dir / f"{self.timestamp}_combined.log"

        self._stdout_file = stdout_path.open("w", encoding="utf-8")
        self._combined_file = combined_path.open("w", encoding="utf-8")

        # Write header
        header = f"==== Benchmark Run: {datetime.now().isoformat()} ====\n\n"
        self._stdout_file.write(header)
        self._combined_file.write(header)

        # Save originals
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr

        # Redirect stdout to: console + stdout_file + combined_file
        sys.stdout = TeeWriter(
            self._original_stdout,
            self._stdout_file,
            self._combined_file,
        )

        # Redirect stderr to: console + combined_file only
        sys.stderr = TeeWriter(
            self._original_stderr,
            self._combined_file,
        )

    def stop(self):
        # Restore original streams
        if self._original_stdout:
            sys.stdout = self._original_stdout
            self._original_stdout = None
        if self._original_stderr:
            sys.stderr = self._original_stderr
            self._original_stderr = None

        # Close files
        if self._stdout_file:
            self._stdout_file.close()
            self._stdout_file = None
        if self._combined_file:
            self._combined_file.close()
            self._combined_file = None


# ======================================================
# DATA LOADING
# ======================================================

def load_examples(task) -> list[dict]:
    """Load evaluation examples from HuggingFace."""
    examples = load_from_hf(task, split=task.hf_split)
    print(f"[INFO] Loaded {len(examples)} examples from HF ({task.hf_repo})")
    return examples


def build_fewshot_block(data_dir: Path, task, k: int) -> str:
    """Build few-shot prompt block from curated few-shot file or training data.

    Uses task.build_prompt to format each example properly (includes MCQ choices
    for MMLU, proper question format for other tasks), then appends the answer.

    Supports two formats:
    1. Simple answer-only: fewshot_answer_field contains the complete answer
    2. Solution+Answer: fewshot_solution_field has reasoning, fewshot_final_answer_field has the answer
       Format: {solution} \boxed{answer}
    """
    # Format instruction - always included even for 0-shot
    format_instruction = r"Output format: end your response with \boxed{<answer>} where <answer> is the final answer."

    if k <= 0:
        return format_instruction

    examples = _load_fewshot_raw(data_dir, task, k)
    if not examples:
        print(f"[WARN] No few-shot examples available for {task.name}; "
              f"falling back to 0-shot prompt")
        return format_instruction

    blocks = []

    # Add format instruction at the start
    blocks.append(format_instruction)

    # Check which format to use
    use_solution_answer_format = bool(task.fewshot_solution_field)

    for ex in examples:
        # Use task's build_prompt to format properly (includes MCQ choices, etc.)
        question_part = task.build_prompt(ex, "")

        if use_solution_answer_format:
            # Solution + Answer format: {reasoning} \boxed{answer}
            solution = ex.get(task.fewshot_solution_field, "")
            # Handle list values (e.g., OlympiadBench solution field is a list)
            if isinstance(solution, list):
                solution = solution[0] if solution else ""
            solution = str(solution).strip()

            # Get final answer if available
            final_answer = ""
            if task.fewshot_final_answer_field:
                final_answer = ex.get(task.fewshot_final_answer_field, "")
                if isinstance(final_answer, list):
                    final_answer = final_answer[0] if final_answer else ""
                final_answer = str(final_answer).strip() if final_answer else ""

            if question_part and solution:
                if final_answer:
                    # Full format with reasoning and answer
                    blocks.append(f"{question_part}\n{solution}\n\n\\boxed{{{final_answer}}}")
                else:
                    # Proof-type problems without numeric answer (just solution)
                    blocks.append(f"{question_part}\n{solution}")
        else:
            # Simple answer-only format (e.g., GSM8K where answer contains "solution text\n#### number")
            answer = ex.get(task.fewshot_answer_field, "")
            # Handle list values
            if isinstance(answer, list):
                answer = answer[0] if answer else ""
            answer = str(answer).strip()
            if question_part and answer:
                # Check if answer contains #### delimiter (like GSM8K format)
                if "####" in answer:
                    # Split into solution and final answer
                    parts = answer.split("####")
                    solution_part = parts[0].strip()
                    final_answer = parts[-1].strip()
                    # Don't double-wrap if solution already contains \boxed{}
                    if "\\boxed{" in solution_part:
                        blocks.append(f"{question_part}\n{solution_part}")
                    else:
                        blocks.append(f"{question_part}\n{solution_part}\n\n\\boxed{{{final_answer}}}")
                else:
                    # Plain answer without solution
                    blocks.append(f"{question_part}\n\\boxed{{{answer}}}")

    return "\n\n".join(blocks)


def build_fewshot_block_logprob_token(data_dir: Path, task, k: int) -> str:
    """Build few-shot prompt block for first-token log-probability MCQ evaluation.

    Simpler than generate mode: no format instruction, no \\boxed{}.
    Each few-shot example shows the question followed by the gold answer letter.
    """
    if k <= 0:
        return ""

    examples = _load_fewshot_raw(data_dir, task, k)
    if not examples:
        return ""

    blocks = []

    for ex in examples:
        # Build question prompt (e.g., "Question: ...\nA. ...\nB. ...\nAnswer:")
        question_part = task.build_prompt(ex, "")
        # Get the gold answer letter (e.g., "A", "B", "C", "D")
        gold = task.extract_gold(ex)
        if question_part and gold:
            blocks.append(f"{question_part} {gold}")

    return "\n\n".join(blocks)


def build_fewshot_block_logprob_seq(data_dir: Path, task, k: int) -> str:
    """Build few-shot prompt block for completion log-likelihood MCQ evaluation.

    Each few-shot example shows the question followed by the full answer text
    (not the letter), consistent with the evaluation that scores full answer texts.
    """
    if k <= 0:
        return ""

    examples = _load_fewshot_raw(data_dir, task, k)
    if not examples:
        return ""

    blocks = []

    for ex in examples:
        question_part = task.build_prompt(ex, "")
        gold_label = task.extract_gold(ex)
        # Map gold label to full answer text
        if task.get_choice_texts and task.choice_labels:
            choice_texts = task.get_choice_texts(ex)
            if gold_label in task.choice_labels:
                gold_idx = task.choice_labels.index(gold_label)
                if gold_idx < len(choice_texts):
                    answer_text = choice_texts[gold_idx]
                    if question_part and answer_text:
                        blocks.append(f"{question_part} {answer_text}")
                    continue
        # Fallback: use the label itself
        if question_part and gold_label:
            blocks.append(f"{question_part} {gold_label}")

    return "\n\n".join(blocks)


# ======================================================
# MULTI-TURN CHAT FEW-SHOT (for instruct/chat models)
# ======================================================

def _load_fewshot_raw(data_dir: Path, task, k: int) -> list[dict]:
    """Load few-shot examples from local file or HF. Returns [] if nothing configured."""
    if k <= 0:
        return []

    # Prefer local fewshot file if configured
    if task.fewshot_path:
        local_path = data_dir / task.fewshot_path
        if local_path.exists():
            examples = load_jsonl(local_path)
            print(f"[INFO] Loaded {len(examples)} few-shot examples from {local_path}")
            return examples[:k]

    if task.hf_repo and task.hf_fewshot_split:
        from datasets import load_dataset
        config = task.hf_fewshot_config or task.hf_config
        ds = load_dataset(task.hf_repo, config, split=task.hf_fewshot_split)
        return [dict(row) for row in ds][:k]

    return []


def build_fewshot_chat_messages(data_dir: Path, task, k: int) -> list[dict] | None:
    """Build few-shot examples as multi-turn chat messages for generate eval mode.

    Returns list of message dicts (system + user/assistant pairs) or None if no examples.
    The caller should append the final test question as a user message.
    """
    format_instruction = r"Output format: end your response with \boxed{<answer>} where <answer> is the final answer."

    examples = _load_fewshot_raw(data_dir, task, k)
    if not examples:
        return None

    messages: list[dict] = [{"role": "system", "content": format_instruction}]
    use_solution_answer_format = bool(task.fewshot_solution_field)

    for ex in examples:
        user_msg = task.build_prompt(ex, "")
        if not user_msg:
            continue

        if use_solution_answer_format:
            solution = ex.get(task.fewshot_solution_field, "")
            if isinstance(solution, list):
                solution = solution[0] if solution else ""
            solution = str(solution).strip()
            if not solution:
                continue

            final_answer = ""
            if task.fewshot_final_answer_field:
                final_answer = ex.get(task.fewshot_final_answer_field, "")
                if isinstance(final_answer, list):
                    final_answer = final_answer[0] if final_answer else ""
                final_answer = str(final_answer).strip() if final_answer else ""

            if final_answer:
                assistant_msg = f"{solution}\n\n\\boxed{{{final_answer}}}"
            else:
                assistant_msg = solution
        else:
            answer = ex.get(task.fewshot_answer_field, "")
            if isinstance(answer, list):
                answer = answer[0] if answer else ""
            answer = str(answer).strip()
            if not answer:
                continue

            if "####" in answer:
                parts = answer.split("####")
                solution_part = parts[0].strip()
                final_answer = parts[-1].strip()
                if "\\boxed{" in solution_part:
                    assistant_msg = solution_part
                else:
                    assistant_msg = f"{solution_part}\n\n\\boxed{{{final_answer}}}"
            else:
                assistant_msg = f"\\boxed{{{answer}}}"

        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})

    # Only return if we have actual example pairs (not just system message)
    return messages if len(messages) > 1 else None


def build_fewshot_chat_messages_logprob_token(data_dir: Path, task, k: int) -> list[dict] | None:
    """Build few-shot examples as multi-turn chat messages for logprob_token eval mode.

    Returns list of message dicts (user/assistant pairs) or None if no examples.
    """
    examples = _load_fewshot_raw(data_dir, task, k)
    if not examples:
        return None

    messages: list[dict] = []
    for ex in examples:
        user_msg = task.build_prompt(ex, "")
        gold = task.extract_gold(ex)
        if user_msg and gold:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": gold})

    return messages if messages else None


def build_fewshot_chat_messages_logprob_seq(data_dir: Path, task, k: int) -> list[dict] | None:
    """Build few-shot examples as multi-turn chat messages for logprob_seq eval mode.

    Returns list of message dicts (user/assistant pairs) or None if no examples.
    """
    examples = _load_fewshot_raw(data_dir, task, k)
    if not examples:
        return None

    messages: list[dict] = []
    for ex in examples:
        user_msg = task.build_prompt(ex, "")
        gold_label = task.extract_gold(ex)

        answer_text = gold_label  # fallback
        if task.get_choice_texts and task.choice_labels:
            choice_texts = task.get_choice_texts(ex)
            if gold_label in task.choice_labels:
                gold_idx = task.choice_labels.index(gold_label)
                if gold_idx < len(choice_texts):
                    answer_text = choice_texts[gold_idx]

        if user_msg and answer_text:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": answer_text})

    return messages if messages else None


# ======================================================
# RUN SINGLE TASK
# ======================================================

def apply_chat_template(tokenizer, task, example: dict, prompt_raw: str,
                        fewshot_chat_prefix: list[dict] | None,
                        enabled: bool) -> str:
    """Wrap a raw prompt in the model's chat template (if enabled and available).

    With a few-shot chat prefix, the few-shot examples become user/assistant
    turns and only the test question goes in the final user message.
    Falls back to the raw prompt if the tokenizer has no template or it fails.
    """
    if not enabled or not hasattr(tokenizer, "apply_chat_template"):
        return prompt_raw
    try:
        if fewshot_chat_prefix:
            test_question = task.build_prompt(example, "")
            messages = fewshot_chat_prefix + [{"role": "user", "content": test_question}]
        else:
            messages = [{"role": "user", "content": prompt_raw}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        return prompt_raw


def generate_batched(llm: LLM, prompts: list[str], sampling_params: SamplingParams,
                     batch_size: int | None) -> list:
    """Run llm.generate, optionally splitting prompts into batch_size chunks."""
    if not batch_size or batch_size <= 0:
        return llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)

    outputs = []
    num_batches = (len(prompts) + batch_size - 1) // batch_size
    print(f"[INFO] Processing in {num_batches} batches of size {batch_size}")
    for batch_idx in range(0, len(prompts), batch_size):
        batch = prompts[batch_idx:batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1
        print(f"[INFO] Batch {batch_num}/{num_batches} ({len(batch)} prompts)...")
        outputs.extend(llm.generate(batch, sampling_params=sampling_params, use_tqdm=True))
    return outputs


def run_task(
    cfg: RunConfig,
    task_name: str,
    fewshot_k: int | None,
    llm: LLM,
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

    # Get tokenizer (needed for chat template and logprob token ID lookup)
    tokenizer = llm.get_tokenizer()

    # Common: build sample IDs and gold answers
    sample_ids: list[str] = []
    gold_answers: list[str] = []
    prompts_without_fewshot: list[str] = []
    for idx, ex in enumerate(eval_examples):
        sample_ids.append(str(ex.get("id", ex.get("idx", ex.get("index", idx)))))
        gold_answers.append(task.extract_gold(ex))
        prompts_without_fewshot.append(task.build_prompt(ex, ""))

    # ========================================
    # LOGPROB TOKEN EVALUATION PATH (first-token log-probability)
    # ========================================
    if task.eval_mode == "logprob_token" and task.choice_labels:
        choice_labels = task.choice_labels
        num_choices = len(choice_labels)
        print(f"[INFO] Logprob MCQ mode: {num_choices} choices {choice_labels}")

        # Pre-tokenize choice labels (with space prefix, as they follow "Answer:")
        # e.g., " A" -> token_id, " B" -> token_id, etc.
        choice_token_ids: dict[str, int] = {}
        for label in choice_labels:
            token_ids = tokenizer.encode(f" {label}", add_special_tokens=False)
            choice_token_ids[label] = token_ids[-1]
        print(f"[INFO] Choice token IDs: {choice_token_ids}")

        # Build one prompt per example (ending with "Answer:")
        prompts: list[str] = [
            apply_chat_template(tokenizer, task, ex, task.build_prompt(ex, fewshot_block),
                                fewshot_chat_prefix, cfg.apply_chat_template)
            for ex in eval_examples
        ]

        print(f"[INFO] Built {len(prompts)} prompts (1 per example)")

        # Single-prompt logprob: generate 1 token, get top-100 logprobs
        sampling_params = SamplingParams(
            max_tokens=1,
            temperature=0,
            logprobs=100,
            seed=cfg.seed,
        )

        print(f"[INFO] Running logprob inference...")
        inference_start = time.time()
        outputs = generate_batched(llm, prompts, sampling_params, cfg.batch_size)
        inference_time = time.time() - inference_start
        print(f"[INFO] Logprob inference completed in {format_time(inference_time)}")

        # Process outputs: extract logprobs for each choice label from generated token logprobs
        log_entries: list[dict] = []
        correct_count = 0
        all_stop_reasons: list[str] = []

        for ex_idx, output in enumerate(outputs):
            gold = gold_answers[ex_idx]

            # output.outputs[0].logprobs[0] = logprobs dict for the 1st generated token
            # Maps token_id -> Logprob(logprob, rank, decoded_token)
            gen_logprobs = output.outputs[0].logprobs[0]

            # Extract logprob for each choice label
            logprobs_dict = {}
            for label, token_id in choice_token_ids.items():
                if token_id in gen_logprobs:
                    lp = gen_logprobs[token_id]
                    logprobs_dict[label] = float(getattr(lp, "logprob", lp))
                else:
                    # Not in top-100 → negligible probability
                    logprobs_dict[label] = float("-inf")

            # Pick the choice with highest logprob
            pred = max(logprobs_dict, key=logprobs_dict.get)
            is_correct = task.match_fn(gold, pred)
            if is_correct:
                correct_count += 1

            all_stop_reasons.append("logprob_token")

            # Store logprobs dict as the "response" (JSON string)
            logprobs_json = json.dumps(logprobs_dict)
            log_entries.append({
                "sample_id": sample_ids[ex_idx],
                "prompt_actual": prompts_without_fewshot[ex_idx],
                "prompt_full": prompts[ex_idx],
                "gold_answer": gold,
                "is_correct": is_correct,
                "extracted_answers": [pred],
                "stop_reasons": ["logprob_token"],
                "responses": [logprobs_json],
            })

    # ========================================
    # LOGPROB SEQ EVALUATION PATH (completion log-likelihood)
    # ========================================
    elif task.eval_mode == "logprob_seq" and task.choice_labels and task.get_choice_texts:
        choice_labels = task.choice_labels
        print(f"[INFO] Logprob-seq (completion log-likelihood) mode: {len(choice_labels)} choices {choice_labels}")

        # Build base prompts and full prompts (base + each choice text) for all examples
        all_prompts: list[str] = []       # Flat list: N_examples * N_choices
        prompt_map: list[tuple[int, int]] = []  # (example_idx, choice_idx)
        base_prompts: list[str] = []      # One per example (after chat template)

        for ex_idx, ex in enumerate(eval_examples):
            base_prompt = apply_chat_template(
                tokenizer, task, ex, task.build_prompt(ex, fewshot_block),
                fewshot_chat_prefix, cfg.apply_chat_template,
            )
            base_prompts.append(base_prompt)
            choice_texts = task.get_choice_texts(ex)

            for c_idx, choice_text in enumerate(choice_texts):
                if c_idx >= len(choice_labels):
                    break
                full_prompt = base_prompt + " " + choice_text
                all_prompts.append(full_prompt)
                prompt_map.append((ex_idx, c_idx))

        print(f"[INFO] Built {len(all_prompts)} prompts ({len(eval_examples)} examples x {len(choice_labels)} choices)")

        # Run inference with prompt_logprobs to score each continuation
        sampling_params = SamplingParams(
            max_tokens=1,
            temperature=0,
            prompt_logprobs=1,
            seed=cfg.seed,
        )

        print(f"[INFO] Running logprob-seq inference...")
        inference_start = time.time()
        outputs = generate_batched(llm, all_prompts, sampling_params, cfg.batch_size)
        inference_time = time.time() - inference_start
        print(f"[INFO] Logprob-seq inference completed in {format_time(inference_time)}")

        # Score each (example, choice) pair:
        # Tokenize base_prompt to find boundary, sum logprobs for answer tokens, normalize
        scores: dict[int, list[tuple[str, float]]] = {}  # ex_idx -> [(label, norm_score), ...]

        for out_idx, output in enumerate(outputs):
            ex_idx, c_idx = prompt_map[out_idx]
            label = choice_labels[c_idx]

            # Get prompt token IDs and per-token logprobs from vLLM output
            prompt_token_ids = output.prompt_token_ids
            prompt_logprobs = output.prompt_logprobs  # list[dict | None], one per token

            # Find the context/answer boundary: the answer starts where the full
            # prompt's tokens diverge from the base prompt's tokens (tokenizing
            # base+answer can merge tokens at the seam, so len(base) alone is off)
            base_token_ids = tokenizer.encode(base_prompts[ex_idx])
            n_ctx = 0
            for base_tok, full_tok in zip(base_token_ids, prompt_token_ids):
                if base_tok != full_tok:
                    break
                n_ctx += 1

            # Sum logprobs for answer tokens (positions n_ctx onward)
            n_answer = len(prompt_token_ids) - n_ctx
            if n_answer <= 0:
                n_answer = 1  # Safety: avoid division by zero

            total_logprob = 0.0
            for i in range(n_ctx, len(prompt_token_ids)):
                if prompt_logprobs[i] is not None:
                    token_id = prompt_token_ids[i]
                    if token_id in prompt_logprobs[i]:
                        lp = prompt_logprobs[i][token_id]
                        total_logprob += float(getattr(lp, "logprob", lp))

            normalized_score = total_logprob / n_answer

            if ex_idx not in scores:
                scores[ex_idx] = []
            scores[ex_idx].append((label, normalized_score))

        # Pick best choice per example and compare with gold
        log_entries: list[dict] = []
        correct_count = 0
        all_stop_reasons: list[str] = []

        for ex_idx in range(len(eval_examples)):
            gold = gold_answers[ex_idx]
            ex_scores = scores.get(ex_idx, [])

            if ex_scores:
                pred = max(ex_scores, key=lambda x: x[1])[0]
            else:
                pred = ""

            is_correct = task.match_fn(gold, pred)
            if is_correct:
                correct_count += 1

            all_stop_reasons.append("logprob_seq")

            # Store scores as JSON for inspection
            scores_dict = {lbl: sc for lbl, sc in ex_scores}
            scores_json = json.dumps(scores_dict)

            log_entries.append({
                "sample_id": sample_ids[ex_idx],
                "prompt_actual": prompts_without_fewshot[ex_idx],
                "prompt_full": base_prompts[ex_idx],
                "gold_answer": gold,
                "is_correct": is_correct,
                "extracted_answers": [pred],
                "stop_reasons": ["logprob_seq"],
                "responses": [scores_json],
            })

    # ========================================
    # GENERATE EVALUATION PATH (default)
    # ========================================
    else:
        # ---------- sampling params ----------
        sampling_params = SamplingParams(
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_tokens,
            stop=task.stop_strings if task.stop_strings else None,
            n=cfg.pass_k,
            skip_special_tokens=False,
            seed=cfg.seed,
        )
        if cfg.pass_k > 1:
            print(f"[INFO] Pass@{cfg.pass_k} mode: generating {cfg.pass_k} responses per sample")

        # ---------- run batch inference ----------
        print(f"[INFO] Building prompts...")

        prompts: list[str] = [
            apply_chat_template(tokenizer, task, ex, task.build_prompt(ex, fewshot_block),
                                fewshot_chat_prefix, cfg.apply_chat_template)
            for ex in eval_examples
        ]

        print(f"[INFO] Running batch inference on {len(prompts)} examples...")
        inference_start = time.time()
        outputs = generate_batched(llm, prompts, sampling_params, cfg.batch_size)
        inference_time = time.time() - inference_start
        print(f"[INFO] Batch inference completed in {format_time(inference_time)}")

        # Process outputs and build log entries
        log_entries: list[dict] = []
        correct_count = 0
        all_stop_reasons: list[str] = []

        for idx, (output, ex) in enumerate(zip(outputs, eval_examples)):
            gold = gold_answers[idx]
            all_responses = output.outputs
            any_correct = False
            responses_list = []

            for resp_idx, result in enumerate(all_responses):
                pred_raw = result.text
                finish_reason = result.finish_reason or ""
                stop_reason = result.stop_reason
                if stop_reason:
                    stop_reason_str = f"{finish_reason}:{stop_reason}"
                else:
                    stop_reason_str = f"{finish_reason}:-"

                pred = task.extract_pred(pred_raw)
                is_correct = task.match_fn(gold, pred)

                responses_list.append({
                    "response": pred_raw,
                    "extracted": pred,
                    "correct": is_correct,
                    "stop_reason": stop_reason_str,
                })
                all_stop_reasons.append(stop_reason_str)

                if is_correct:
                    any_correct = True

            if any_correct:
                correct_count += 1

            log_entries.append({
                "sample_id": sample_ids[idx],
                "prompt_actual": prompts_without_fewshot[idx],
                "prompt_full": prompts[idx],
                "gold_answer": gold,
                "is_correct": any_correct,
                "extracted_answers": [r["extracted"] for r in responses_list],
                "stop_reasons": [r["stop_reason"] for r in responses_list],
                "responses": [r["response"] for r in responses_list],
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
        if entry["extracted_answers"] and entry["extracted_answers"][0] == ""
    )

    # Count occurrences of each stop_reason across all responses
    stop_reason_counts = dict(Counter(all_stop_reasons))

    # Include fewshot in task name for unique identification
    task_name_with_fewshot = f"{task.name}({fewshot_k})"

    summary = {
        "model": model_name,
        "model_tag": model_tag,
        "task": task_name_with_fewshot,
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
    print(f"[{task_name_with_fewshot}] Accuracy: {correct}/{total} = {acc:.4f} | Duration: {format_time(duration)}")

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
    benchmark_db: BenchmarkDatabase,
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
    print(f"\n[INFO] Loading model: {model_name}")

    # Determine quantization for this model
    quantization = cfg.quantization_for(model_tag)

    if quantization:
        print(f"[INFO] Using quantization: {quantization}")
    else:
        print(f"[INFO] Using model default precision (no quantization override)")

    llm_kwargs = {
        "model": model_name,
        "trust_remote_code": True,
        "gpu_memory_utilization": cfg.gpu_memory_utilization,
        "enforce_eager": cfg.enforce_eager,
        "max_logprobs": 100,  # Default is 20; need 100 for logprob MCQ evaluation
    }
    if cfg.max_model_len is not None:
        llm_kwargs["max_model_len"] = cfg.max_model_len
    if quantization:
        llm_kwargs["quantization"] = quantization
    llm = LLM(**llm_kwargs)

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
                llm=llm,
                model_name=model_name,
                data_dir=data_dir,
                model_tag=model_tag,
                timestamp=timestamp,
            )
        except Exception as e:
            print(f"\n[ERROR] Task {task_name} failed: {e}")
            traceback.print_exc()
            fewshot_k_actual = fewshot_k if fewshot_k is not None else 0
            summary = {
                "task": f"{task_name}({fewshot_k_actual})",
                "model": model_name,
                "model_tag": model_tag,
                "error": str(e),
                "total_examples": 0,
                "correct": 0,
                "accuracy": 0.0,
                "no_answer_count": 0,
                "stop_reason_counts": {},
                "duration_human": "",
                "settings": {
                    "temperature": cfg.temperature,
                    "top_p": cfg.top_p,
                    "max_tokens": cfg.max_tokens,
                },
            }
            log_entries = []
            print(f"[INFO] Continuing to next task...")

        task_summaries.append(summary)

        # Write to consolidated benchmark database
        benchmark_db.add_summary(summary)

        if log_entries:
            # Write results to benchmark database (use task name with fewshot from summary)
            benchmark_db.add_results(summary["task"], model_tag, log_entries)

    # ---------- unload model to free GPU memory ----------
    print(f"\n[INFO] Unloading model: {model_tag}")
    del llm
    destroy_model_parallel()
    gc.collect()

    # Try to clear CUDA cache if available
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

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
        if "error" in summary:
            print(f"  {summary['task']}: ERROR - {summary['error']}")
        else:
            print(f"  {summary['task']}: {summary['accuracy']:.4f} ({summary['correct']}/{summary['total_examples']})")

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


def _run_coordinator(cfg: RunConfig) -> None:
    """Multi-GPU coordinator: spawn parallel workers and collect results."""
    total_start = time.time()
    script_dir = Path(__file__).parent
    results_dir = script_dir / "results"
    results_dir.mkdir(exist_ok=True, parents=True)
    logs_dir = script_dir / "logs"
    logs_dir.mkdir(exist_ok=True, parents=True)

    # Determine tasks (for run_name generation and header display)
    tasks_to_run = _resolve_tasks_to_run(cfg)

    # Generate shared run name (all workers use the same DB)
    run_name = _make_run_name(cfg, len(cfg.models), len(tasks_to_run))
    benchmark_db_path = results_dir / f"{run_name}.sqlite3"

    gpu_ids = [str(g) for g in cfg.gpu_ids]
    num_workers = min(len(gpu_ids), len(cfg.models))

    # Write the RESOLVED config next to the DB: it's the run's reproducibility
    # artifact, and workers load it so they see exactly the coordinator's config
    # (including any CLI overrides).
    resolved_config_path = results_dir / f"{run_name}_config.yaml"
    resolved_config_path.write_text(cfg.to_yaml(), encoding="utf-8")

    # Pre-create the DB so WAL mode and tables are ready before workers connect
    benchmark_db = BenchmarkDatabase(benchmark_db_path, pass_k=cfg.pass_k)
    benchmark_db.save_run_config(cfg.to_yaml())
    benchmark_db.close()

    # ---------- print header ----------
    task_strs = [f"{name}({k})" if k is not None else name for name, k in tasks_to_run]
    print(f"{'='*60}")
    print(f"LLM BENCHMARK RUNNER - MULTI-GPU ({num_workers} GPUs)")
    print(f"{'='*60}")
    print(f"Models: {len(cfg.models)} (distributed across {num_workers} GPUs)")
    for i, m in enumerate(cfg.models):
        print(f"  GPU {gpu_ids[i % num_workers]}: {Path(m).name}")
    print(f"Tasks: {', '.join(task_strs)}")
    print(f"Database: {benchmark_db_path}")
    print(f"Config: {resolved_config_path}")
    print(f"{'='*60}")

    # ---------- spawn worker processes ----------
    processes = []
    for worker_id in range(num_workers):
        gpu_id = gpu_ids[worker_id]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        cmd = [
            sys.executable, "-m", "lm_eval_ledger.cli",
            "--config", str(resolved_config_path),
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

    benchmark_db = BenchmarkDatabase(benchmark_db_path, pass_k=cfg.pass_k)
    rows = benchmark_db.conn.execute(
        "SELECT task_name, model_tag, accuracy, total_examples, correct, error "
        "FROM summaries WHERE task_name != 'TOTAL' ORDER BY id"
    ).fetchall()

    if rows:
        print(f"\nResults by model:")
        print(f"{'-'*60}")
        current_model = None
        for row in rows:
            if row["model_tag"] != current_model:
                current_model = row["model_tag"]
                print(f"\n{current_model}:")
            if row["error"]:
                print(f"  {row['task_name']}: ERROR - {row['error']}")
            else:
                print(f"  {row['task_name']}: {row['accuracy']:.4f} ({row['correct']}/{row['total_examples']})")

    # Add TOTAL summary row
    total_examples = sum(r["total_examples"] or 0 for r in rows if not r["error"])
    total_correct = sum(r["correct"] or 0 for r in rows if not r["error"])
    total_accuracy = total_correct / total_examples if total_examples > 0 else 0.0
    model_tags = sorted(set(r["model_tag"] for r in rows))
    task_names_set = sorted(set(r["task_name"] for r in rows))

    total_summary = {
        "task": "TOTAL",
        "model_tag": f"{len(model_tags)} models",
        "total_examples": total_examples,
        "correct": total_correct,
        "accuracy": total_accuracy,
        "no_answer_count": 0,
        "stop_reason_counts": {},
        "duration_human": format_time(total_duration),
        "settings": {
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "max_tokens": cfg.max_tokens,
        },
        "error": None,
        "model": f"Models: {', '.join(model_tags)} | Tasks: {', '.join(task_names_set)}",
    }
    benchmark_db.add_summary(total_summary)
    # All workers have exited, so it's safe to clean up the WAL/SHM files
    benchmark_db.close(remove_sidecars=True)

    if failed:
        print(f"\n[WARN] {len(failed)} worker(s) failed: {failed}")

    print(f"\nResults saved to: {benchmark_db_path}")
    print(f"Logs saved to: {logs_dir}")
    print(f"{'#'*60}")


def main() -> None:
    args = build_arg_parser().parse_args()
    try:
        cfg = resolve_config(args)
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    # Multi-GPU coordinator mode: gpu_ids has 2+ GPUs and not already a worker
    if cfg.gpu_ids and len(cfg.gpu_ids) > 1 and args.shard is None:
        _run_coordinator(cfg)
        return

    # ---------- single-GPU / worker mode ----------

    # Pin to specific GPU if gpu_ids has exactly one entry (e.g., gpu_ids: [7])
    if cfg.gpu_ids and len(cfg.gpu_ids) == 1 and args.shard is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_ids[0])
        print(f"[INFO] Pinning to GPU {cfg.gpu_ids[0]} (CUDA_VISIBLE_DEVICES={cfg.gpu_ids[0]})")

    total_start = time.time()

    # ---------- determine model subset for this worker ----------
    models = cfg.models
    shard_idx = None
    num_shards = None
    if args.shard is not None:
        shard_idx, num_shards = (int(x) for x in args.shard.split("/"))
        models = cfg.models[shard_idx::num_shards]
        if not models:
            print(f"[WORKER {shard_idx}] No models assigned, exiting.")
            return

    # ---------- setup paths ----------
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    timestamp = datetime.now().strftime("%m%d_%H%M")

    # ---------- setup results directory ----------
    results_dir = script_dir / "results"
    results_dir.mkdir(exist_ok=True, parents=True)

    # ---------- determine tasks to run ----------
    tasks_to_run = _resolve_tasks_to_run(cfg)

    # ---------- setup consolidated benchmark database and logs ----------
    if args.run_name:
        # Worker mode: use shared run name from coordinator
        run_name = args.run_name
    else:
        # Standalone mode: generate run name
        run_name = _make_run_name(cfg, len(models), len(tasks_to_run))

    benchmark_db_path = results_dir / f"{run_name}.sqlite3"
    benchmark_db = BenchmarkDatabase(benchmark_db_path, pass_k=cfg.pass_k)

    # Standalone mode: record the resolved config (file + DB).
    # In worker mode the coordinator already did both.
    if args.shard is None:
        resolved_config_path = results_dir / f"{run_name}_config.yaml"
        resolved_config_path.write_text(cfg.to_yaml(), encoding="utf-8")
        benchmark_db.save_run_config(cfg.to_yaml())

    # ---------- setup output logging ----------
    logs_dir = script_dir / "logs"
    logs_dir.mkdir(exist_ok=True, parents=True)
    # Suffix log files with shard ID to avoid collisions in multi-GPU mode
    log_name = f"{run_name}_gpu{shard_idx}" if shard_idx is not None else run_name
    output_logger = OutputLogger(logs_dir, log_name)
    output_logger.start()

    try:
        _run_models(cfg, models, tasks_to_run, data_dir, timestamp, benchmark_db,
                    shard_idx, num_shards, total_start)
    finally:
        # In multi-GPU mode other workers may still hold the DB open;
        # only the last close of a run may delete the WAL/SHM sidecars.
        benchmark_db.close(remove_sidecars=args.shard is None)
        output_logger.stop()

    print(f"\nResults saved to: {benchmark_db_path}")
    print(f"Logs saved to: {logs_dir / log_name}_stdout.log and _combined.log")
    print(f"{'#'*60}")


def _run_models(
    cfg: RunConfig,
    models: list[str],
    tasks_to_run: list[tuple[str, int | None]],
    data_dir: Path,
    timestamp: str,
    benchmark_db: BenchmarkDatabase,
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
    print(f"Database: {benchmark_db.db_path}")
    print(f"{'='*60}")

    # ---------- run all models ----------
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
                benchmark_db=benchmark_db,
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
            if "error" in task_summary:
                print(f"  {task_summary['task']}: ERROR")
            else:
                print(f"  {task_summary['task']}: {task_summary['accuracy']:.4f}")

    # ---------- add total summary row to database ----------
    # Only in standalone mode (single-GPU). In multi-GPU mode, coordinator adds TOTAL.
    if shard_idx is None:
        total_examples = 0
        total_correct = 0
        total_no_answer = 0
        model_tags = []
        task_names = set()

        for model_summary in all_model_summaries:
            model_tags.append(model_summary.get("model_tag", ""))
            for task_summary in model_summary.get("tasks", []):
                if "error" not in task_summary:
                    total_examples += task_summary.get("total_examples", 0)
                    total_correct += task_summary.get("correct", 0)
                    total_no_answer += task_summary.get("no_answer_count", 0)
                    task_names.add(task_summary.get("task", ""))

        total_accuracy = total_correct / total_examples if total_examples > 0 else 0.0

        # Add TOTAL row to summaries table
        total_summary = {
            "task": "TOTAL",
            "model_tag": f"{len(model_tags)} models",
            "total_examples": total_examples,
            "correct": total_correct,
            "accuracy": total_accuracy,
            "no_answer_count": total_no_answer,
            "stop_reason_counts": {},
            "duration_human": format_time(total_duration),
            "settings": {
                "temperature": cfg.temperature,
                "top_p": cfg.top_p,
                "max_tokens": cfg.max_tokens,
            },
            "error": None,  # None means completed successfully
            "model": f"Models: {', '.join(model_tags)} | Tasks: {', '.join(sorted(task_names))}",
        }
        benchmark_db.add_summary(total_summary)


if __name__ == "__main__":
    main()
