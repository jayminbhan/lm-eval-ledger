# verifier.py
"""LLM-based answer verification over a finished run's database.

Uses CompassVerifier (Liu et al., "CompassVerifier: A Unified and Robust
Verifier for LLMs Evaluation and Outcome Reward", arXiv:2508.03686;
models: opencompass/CompassVerifier-3B/-7B/-32B, Qwen2.5-Instruct based)
to judge whether each stored response answers its question correctly,
independent of the string-matching extract/match pipeline. This is a local
substitute for API-judge scoring (e.g., HLE's official o3-mini judge) -
same role, not the identical pipeline.

The verifier runs AFTER benchmarking, over the results already stored in
SQLite (so the eval model and the verifier never share GPU memory), and
writes per-response verdicts plus verified accuracies back into the DB:

- results.verifier_verdicts: JSON list of A/B/C verdicts (A=correct,
  B=incorrect, C=invalid response; B and C both count as incorrect)
- results.verified_correct: 0.0/1.0 (rows the verifier skipped stay NULL)
- summaries.verified_correct / verified_accuracy: recomputed per task

Modes:
- "fallback" (default): only responses that string-matching marked wrong
  are verified; a sample counts as verified-correct if either pipeline
  accepts it. Cheap, and recovers format-noncompliant answers.
- "all": every response is verified; the verifier's verdict alone decides.

Tasks where LLM verification is meaningless are skipped automatically:
logprob modes (responses are score dicts), MRCR (official metric is a
sequence ratio), and LiveCodeBench (correctness is execution-defined).

Activate via config (verifier_model: opencompass/CompassVerifier-7B) to
run automatically after a benchmark run, or standalone on any past DB:

    lm-eval-ledger-verify results/run.sqlite3 --model opencompass/CompassVerifier-7B
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

# Official CompassVerifier prompt (verbatim from open-compass/CompassVerifier
# src/prompts.py, CV_PROMPT).
CV_PROMPT = """
Please as a grading expert, judge whether the final answers given by the candidates below are consistent with the standard answers, that is, whether the candidates answered correctly.
Here are some evaluation criteria:
1. Please refer to the given standard answer. You don't need to re-generate the answer to the question because the standard answer has been given. You only need to judge whether the candidate's answer is consistent with the standard answer according to the form of the question. THE STANDARD ANSWER IS ALWAYS CORRECT AND THE QUESTION IS PERFECTLY VALID. NEVER QUESTION THEM.
2. ONLY compare the FINAL ANSWER - COMPLETELY IGNORE any potential errors in the REASONING PROCESSES.
3. Some answers may be expressed in different ways, such as some answers may be a mathematical expression, some answers may be a textual description, as long as the meaning expressed is the same. Before making a judgment, please understand the question and the standard answer first, and then judge whether the candidate's answer is correct.
4. Some answers may consist of multiple items, such as multiple-choice questions, multiple-select questions, fill-in-the-blank questions, etc. Regardless of the question type, the final answer will be considered correct as long as it matches the standard answer, regardless of whether the reasoning process is correct. For multiple-select questions and multi-blank fill-in-the-blank questions, all corresponding options or blanks must be answered correctly and match the standard answer exactly to be deemed correct.
5. If the prediction is given with \\boxed{{}}, please ignore the \\boxed{{}} and only judge whether the candidate's answer is consistent with the standard answer.
6. If the candidate's answer is invalid (e.g., incomplete (cut off mid-response), lots of unnormal repetitive content, or irrelevant to the question, saying it can't answer the question because some irresistible factors, like ethical issues, no enough information, etc.), select option C (INVALID).Please judge whether the following answers are consistent with the standard answer based on the above criteria. Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT
C: INVALID
Just return the letters "A", "B", or "C", with no text around it.
Here is your task. Simply reply with either CORRECT, INCORRECT, or INVALID. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
<Original Question Begin>:
{question}
<Original Question End>
<Standard Answer Begin>:
{gold_answer}
<Standard Answer End>
<Candidate's Answer Begin>:
{llm_response}
<Candidate's Answer End>
Judging the correctness of the candidate's answer:
"""

# Tasks whose correctness cannot be judged by an LLM verifier
_SKIP_TASK_PREFIXES = ("mrcr_", "livecodebench")


def process_judgment(judgment_str: str) -> str:
    """Parse the A/B/C verdict from verifier output (official logic)."""
    boxed_matches = re.findall(r"boxed{([A-C])}", judgment_str)
    if boxed_matches:
        return boxed_matches[-1]
    judgment_str = judgment_str.strip()
    if judgment_str in ("A", "B", "C"):
        return judgment_str
    final = judgment_str.split("Final Judgment:")[-1]
    matches = re.findall(r"\(([A-C])\)", final)
    if matches:
        return matches[-1]
    matches = re.findall(r"([A-C])", final)
    if matches:
        return matches[-1]
    return ""


def _truncate_response(response: str, max_chars: int) -> str:
    """Trim over-long responses, keeping the head and (crucially) the tail
    where the final answer lives."""
    if len(response) <= max_chars:
        return response
    head = max_chars // 4
    tail = max_chars - head
    return response[:head] + "\n...[truncated]...\n" + response[-tail:]


def _skip_task(task_name: str) -> bool:
    return task_name.startswith(_SKIP_TASK_PREFIXES) or task_name == "TOTAL"


def _ensure_columns(conn: sqlite3.Connection) -> None:
    for table, col, decl in [
        ("results", "verifier_verdicts", "TEXT"),
        ("results", "verified_correct", "REAL"),
        ("summaries", "verified_correct", "REAL"),
        ("summaries", "verified_accuracy", "REAL"),
    ]:
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
        except sqlite3.OperationalError:
            pass  # column already exists


def verify_run(
    db_path: Path | str,
    verifier_model: str,
    mode: str = "fallback",
    max_model_len: int = 16384,
    gpu_memory_utilization: float = 0.95,
    enforce_eager: bool = True,
    seed: int = 42,
    _generate=None,
) -> None:
    """Verify a run's stored responses with an LLM verifier and record
    verified accuracies in the database.

    _generate is an internal test seam: a callable prompts -> list[str]
    replacing the vLLM engine.
    """
    if mode not in ("fallback", "all"):
        raise ValueError(f"verifier mode must be 'fallback' or 'all', got {mode!r}")

    db_path = Path(db_path)
    conn = sqlite3.connect(db_path, isolation_level=None, timeout=60)
    conn.row_factory = sqlite3.Row
    _ensure_columns(conn)

    # Discover response_1..k columns
    cols = [r["name"] for r in conn.execute("PRAGMA table_info(results)")]
    response_cols = sorted(
        (c for c in cols if re.fullmatch(r"response_\d+", c)),
        key=lambda c: int(c.split("_")[1]),
    )

    rows = conn.execute(
        f"SELECT id, task_name, prompt_actual, gold_answer, is_correct, "
        f"stop_reasons, {', '.join(response_cols)} FROM results"
    ).fetchall()

    # Select (row, response) pairs to judge
    jobs: list[tuple[int, list[str]]] = []  # (row_id, responses to judge)
    skipped_tasks: set[str] = set()
    for row in rows:
        if _skip_task(row["task_name"]):
            skipped_tasks.add(row["task_name"])
            continue
        if "logprob" in (row["stop_reasons"] or ""):
            skipped_tasks.add(row["task_name"])
            continue
        if mode == "fallback" and row["is_correct"]:
            continue
        responses = [row[c] for c in response_cols if row[c]]
        if responses:
            jobs.append((row["id"], responses))

    if skipped_tasks:
        print(f"[VERIFY] Skipping non-verifiable tasks: {sorted(skipped_tasks)}")
    n_prompts = sum(len(r) for _, r in jobs)
    print(f"[VERIFY] Mode: {mode} | {len(jobs)} samples, {n_prompts} responses to judge")

    if jobs:
        # Response char budget: leave room for the template, question, and gold
        max_chars = max(4000, (max_model_len - 1500) * 3)

        row_by_id = {row["id"]: row for row in rows}
        prompts: list[str] = []
        prompt_map: list[tuple[int, int]] = []  # (job_idx, response_idx)
        raw_prompts: list[str] = []
        for job_idx, (row_id, responses) in enumerate(jobs):
            row = row_by_id[row_id]
            for resp_idx, response in enumerate(responses):
                raw_prompts.append(CV_PROMPT.format(
                    question=row["prompt_actual"],
                    gold_answer=row["gold_answer"],
                    llm_response=_truncate_response(response, max_chars),
                ))
                prompt_map.append((job_idx, resp_idx))

        if _generate is None:
            _generate = _make_vllm_generate(
                verifier_model, max_model_len, gpu_memory_utilization,
                enforce_eager, seed,
            )
        outputs = _generate(raw_prompts)

        # Collect verdicts per job
        verdicts_by_job: dict[int, list[str]] = {i: [] for i in range(len(jobs))}
        for (job_idx, _resp_idx), out_text in zip(prompt_map, outputs):
            verdicts_by_job[job_idx].append(process_judgment(out_text) or "?")

        # Write per-row verdicts and verified_correct
        for job_idx, (row_id, _responses) in enumerate(jobs):
            verdicts = verdicts_by_job[job_idx]
            verifier_ok = 1.0 if "A" in verdicts else 0.0
            if mode == "fallback":
                verified = max(float(row_by_id[row_id]["is_correct"] or 0.0), verifier_ok)
            else:
                verified = verifier_ok
            conn.execute(
                "UPDATE results SET verifier_verdicts = ?, verified_correct = ? WHERE id = ?",
                (json.dumps(verdicts), verified, row_id),
            )

    # In fallback mode, rows that were already correct keep their score
    if mode == "fallback":
        conn.execute(
            "UPDATE results SET verified_correct = is_correct "
            "WHERE verified_correct IS NULL"
        )

    # ---------- recompute summaries ----------
    print(f"\n[VERIFY] Verified accuracies ({mode} mode):")
    summary_rows = conn.execute(
        "SELECT id, task_name, model_tag, accuracy FROM summaries "
        "WHERE task_name != 'TOTAL' AND (error IS NULL OR error = '')"
    ).fetchall()
    for srow in summary_rows:
        agg = conn.execute(
            "SELECT SUM(COALESCE(verified_correct, is_correct)) AS c, COUNT(*) AS n "
            "FROM results WHERE task_name = ? AND model_name = ?",
            (srow["task_name"], srow["model_tag"]),
        ).fetchone()
        if not agg["n"]:
            continue
        verified_acc = (agg["c"] or 0.0) / agg["n"]
        conn.execute(
            "UPDATE summaries SET verified_correct = ?, verified_accuracy = ? WHERE id = ?",
            (agg["c"], verified_acc, srow["id"]),
        )
        delta = verified_acc - (srow["accuracy"] or 0.0)
        print(f"  {srow['model_tag']} / {srow['task_name']}: "
              f"{srow['accuracy']:.4f} -> {verified_acc:.4f} ({delta:+.4f})")

    # Record verifier settings alongside the run config
    try:
        conn.execute(
            "INSERT INTO run_config (created_at, config_yaml) VALUES (?, ?)",
            (datetime.now().isoformat(timespec="seconds"),
             f"# verifier pass\nverifier_model: {verifier_model}\n"
             f"verifier_mode: {mode}\n"),
        )
    except sqlite3.OperationalError:
        pass  # very old DB without run_config table

    conn.close()
    print(f"\n[VERIFY] Done. Verdicts and verified accuracies written to {db_path}")


def _make_vllm_generate(model: str, max_model_len: int,
                        gpu_memory_utilization: float, enforce_eager: bool,
                        seed: int):
    """Build a prompts -> outputs callable backed by a vLLM engine."""
    from vllm import LLM, SamplingParams

    print(f"[VERIFY] Loading verifier model: {model}")
    llm = LLM(
        model=model,
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
    )
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(temperature=0.0, max_tokens=32, seed=seed)

    def _generate(raw_prompts: list[str]) -> list[str]:
        chat_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False, add_generation_prompt=True,
            )
            for p in raw_prompts
        ]
        outputs = llm.generate(chat_prompts, sampling_params=sampling, use_tqdm=True)
        return [o.outputs[0].text for o in outputs]

    return _generate


def main() -> None:
    """Standalone entry point: verify any past run's database."""
    p = argparse.ArgumentParser(
        prog="lm-eval-ledger-verify",
        description="LLM-verify the responses stored in a benchmark run's "
                    "SQLite database and record verified accuracies.",
    )
    p.add_argument("db", type=Path, help="path to the run's .sqlite3 database")
    p.add_argument("--model", default="opencompass/CompassVerifier-7B",
                   help="verifier model HF id or local path")
    p.add_argument("--mode", choices=["fallback", "all"], default="fallback",
                   help="fallback: only re-judge string-match failures; "
                        "all: verifier verdict replaces string matching")
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    args = p.parse_args()

    if not args.db.exists():
        print(f"[ERROR] Database not found: {args.db}", file=sys.stderr)
        sys.exit(1)

    verify_run(
        args.db, args.model, mode=args.mode,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )


if __name__ == "__main__":
    main()
