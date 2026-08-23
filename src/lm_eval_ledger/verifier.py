# verifier.py
"""LLM-based answer verification over one run in the ledger.

Uses CompassVerifier (Liu et al., "CompassVerifier: A Unified and Robust
Verifier for LLMs Evaluation and Outcome Reward", arXiv:2508.03686;
models: opencompass/CompassVerifier-3B/-7B/-32B, Qwen2.5-Instruct based)
to judge whether each stored response answers its question correctly,
independent of the string-matching extract/match pipeline. This is a local
substitute for API-judge scoring (e.g., HLE's official o3-mini judge) -
same role, not the identical pipeline.

The verifier runs AFTER benchmarking, over samples already stored in the
ledger (so the eval model and the verifier never share GPU memory), scoped
to one run, and writes verdicts back:

- samples.verifier_verdicts: JSON list of A/B/C verdicts per judged
  response (A=correct, B=incorrect, C=invalid; B and C count as incorrect)
- samples.verified_score: 0.0/1.0 (skipped samples stay NULL; queries use
  COALESCE(verified_score, score))
- benchmarks.verifier_model/verifier_mode/verified_correct/verified_accuracy

Modes:
- "fallback" (default): only responses that string-matching marked wrong
  are re-judged; a sample counts as verified-correct if either pipeline
  accepts any of its responses. Cheap, recovers format-noncompliant answers.
- "all": every response is judged; the verifier's verdict alone decides.

Benchmarks where LLM verification is meaningless are skipped automatically:
logprob eval modes (responses are score dicts), MRCR (official metric is a
sequence ratio), and LiveCodeBench (correctness is execution-defined).

Activate via config (verifier_model: opencompass/CompassVerifier-7B) to run
automatically after a benchmark run, or standalone on any run in a ledger:

    lm-eval-ledger-verify results/ledger.sqlite3 --run 3 \\
        --model opencompass/CompassVerifier-7B
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
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

# Benchmarks whose correctness cannot be judged by an LLM verifier
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


def verify_run(
    db_path: Path | str,
    verifier_model: str,
    run_id: int | None = None,
    mode: str = "fallback",
    max_model_len: int = 16384,
    gpu_memory_utilization: float = 0.95,
    enforce_eager: bool = True,
    seed: int = 42,
    _generate=None,
) -> None:
    """Judge one run's stored responses with an LLM verifier and record
    verified scores in the ledger.

    run_id None means the newest run in the ledger. _generate is an internal
    test seam: a callable prompts -> list[str] replacing the vLLM engine.
    """
    if mode not in ("fallback", "all"):
        raise ValueError(f"verifier mode must be 'fallback' or 'all', got {mode!r}")

    db_path = Path(db_path)
    conn = sqlite3.connect(db_path, isolation_level=None, timeout=60)
    conn.row_factory = sqlite3.Row

    if run_id is None:
        row = conn.execute("SELECT MAX(run_id) AS m FROM runs").fetchone()
        run_id = row["m"]
        if run_id is None:
            print("[VERIFY] Ledger has no runs; nothing to do")
            return
        print(f"[VERIFY] No run specified; using newest run id {run_id}")

    # Benchmarks of this run that an LLM can judge
    bench_rows = conn.execute(
        "SELECT benchmark_id, task, model_tag, eval_mode, accuracy FROM benchmarks "
        "WHERE run_id = ? AND (error IS NULL OR error = '')", (run_id,)
    ).fetchall()
    judgeable, skipped = [], []
    for b in bench_rows:
        if b["eval_mode"] != "generate" or b["task"].startswith(_SKIP_TASK_PREFIXES):
            skipped.append(f"{b['task']} ({b['model_tag']})")
        else:
            judgeable.append(b)
    if skipped:
        print(f"[VERIFY] Skipping non-verifiable benchmarks: {skipped}")
    if not judgeable:
        print("[VERIFY] No verifiable benchmarks in this run")
        conn.close()
        return

    bench_ids = [b["benchmark_id"] for b in judgeable]
    placeholders = ",".join("?" * len(bench_ids))
    samples = conn.execute(
        f"SELECT sample_pk, benchmark_id, prompt, gold, responses, score "
        f"FROM samples WHERE benchmark_id IN ({placeholders})", bench_ids
    ).fetchall()

    # Build judging jobs: (sample_pk, [response indices to judge])
    max_chars = max(4000, (max_model_len - 1500) * 3)
    prompts: list[str] = []
    prompt_map: list[tuple[int, int]] = []  # (sample_pk, response_idx)
    responses_by_pk: dict[int, list[dict]] = {}
    for s in samples:
        if mode == "fallback" and (s["score"] or 0) >= 1:
            continue
        responses = json.loads(s["responses"] or "[]")
        responses_by_pk[s["sample_pk"]] = responses
        for r_idx, resp in enumerate(responses):
            if mode == "fallback" and float(resp.get("correct") or 0) >= 1:
                continue
            text = resp.get("text", "")
            if not text:
                continue
            prompts.append(CV_PROMPT.format(
                question=s["prompt"],
                gold_answer=s["gold"],
                llm_response=_truncate_response(text, max_chars),
            ))
            prompt_map.append((s["sample_pk"], r_idx))

    print(f"[VERIFY] Run {run_id} | mode: {mode} | "
          f"{len(responses_by_pk)} samples, {len(prompts)} responses to judge")

    if prompts:
        if _generate is None:
            _generate = _make_vllm_generate(
                verifier_model, max_model_len, gpu_memory_utilization,
                enforce_eager, seed,
            )
        outputs = _generate(prompts)

        verdicts_by_pk: dict[int, list[str]] = {}
        for (pk, _r_idx), out_text in zip(prompt_map, outputs):
            verdicts_by_pk.setdefault(pk, []).append(process_judgment(out_text) or "?")

        sample_by_pk = {s["sample_pk"]: s for s in samples}
        for pk, verdicts in verdicts_by_pk.items():
            verifier_ok = 1.0 if "A" in verdicts else 0.0
            if mode == "fallback":
                verified = max(float(sample_by_pk[pk]["score"] or 0.0), verifier_ok)
            else:
                verified = verifier_ok
            conn.execute(
                "UPDATE samples SET verifier_verdicts = ?, verified_score = ? "
                "WHERE sample_pk = ?",
                (json.dumps(verdicts), verified, pk),
            )

    # ---------- recompute benchmark scores ----------
    # COALESCE keeps the original score for samples the verifier didn't touch
    # (already-correct samples in fallback mode, empty responses).
    print(f"\n[VERIFY] Verified accuracies ({mode} mode):")
    for b in judgeable:
        agg = conn.execute(
            "SELECT SUM(COALESCE(verified_score, score)) AS c, COUNT(*) AS n "
            "FROM samples WHERE benchmark_id = ?", (b["benchmark_id"],)
        ).fetchone()
        if not agg["n"]:
            continue
        verified_acc = (agg["c"] or 0.0) / agg["n"]
        conn.execute(
            "UPDATE benchmarks SET verifier_model = ?, verifier_mode = ?, "
            "verified_correct = ?, verified_accuracy = ? WHERE benchmark_id = ?",
            (verifier_model, mode, agg["c"], verified_acc, b["benchmark_id"]),
        )
        delta = verified_acc - (b["accuracy"] or 0.0)
        print(f"  {b['model_tag']} / {b['task']}: "
              f"{b['accuracy']:.4f} -> {verified_acc:.4f} ({delta:+.4f})")

    conn.close()
    print(f"\n[VERIFY] Done. Verdicts written to {db_path} (run id {run_id})")


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
    """Standalone entry point: verify any run in a ledger."""
    p = argparse.ArgumentParser(
        prog="lm-eval-ledger-verify",
        description="LLM-verify the responses of one run in a ledger database "
                    "and record verified accuracies.",
    )
    p.add_argument("db", type=Path, help="path to the ledger .sqlite3 database")
    p.add_argument("--run", type=int, default=None, metavar="RUN_ID",
                   help="run to verify (default: the newest run; "
                        "see `lm-eval-ledger runs`)")
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
        args.db, args.model, run_id=args.run, mode=args.mode,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )


if __name__ == "__main__":
    main()
