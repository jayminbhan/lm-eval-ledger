# queries.py
"""Canned ledger queries: list runs and diff two runs.

    lm-eval-ledger runs                       # list runs and their scores
    lm-eval-ledger compare 3 7                # accuracy deltas per benchmark
    lm-eval-ledger compare 3 7 --samples      # exactly which samples flipped
    lm-eval-ledger compare 3 7 --match task   # ignore model_tag when pairing

Benchmarks are paired between the two runs on (task, fewshot_k, model_tag)
by default; --match task drops model_tag, for comparing different models or
checkpoints on the same tasks. Scores use COALESCE(verified_score, score),
so verifier results are reflected when present.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

from .db import DEFAULT_LEDGER_NAME


def _connect(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        print(f"[ERROR] Ledger not found: {db_path}", file=sys.stderr)
        sys.exit(1)
    conn = sqlite3.connect(db_path, timeout=60)
    conn.row_factory = sqlite3.Row
    return conn


def cmd_runs(db_path: Path) -> None:
    """List all runs with their benchmark scores."""
    conn = _connect(db_path)
    runs = conn.execute(
        "SELECT run_id, run_name, started_at, harness_version FROM runs ORDER BY run_id"
    ).fetchall()
    if not runs:
        print("Ledger is empty.")
        return
    for r in runs:
        print(f"\nrun {r['run_id']}: {r['run_name']}  ({r['started_at']}"
              f"{', v' + r['harness_version'] if r['harness_version'] else ''})")
        benches = conn.execute(
            "SELECT model_tag, task, fewshot_k, accuracy, verified_accuracy, "
            "total_examples, error FROM benchmarks WHERE run_id = ? "
            "ORDER BY benchmark_id", (r["run_id"],)
        ).fetchall()
        for b in benches:
            label = f"{b['model_tag']} / {b['task']}({b['fewshot_k']})"
            if b["error"]:
                print(f"  {label}: ERROR - {b['error']}")
            else:
                verified = (f"  [verified {b['verified_accuracy']:.4f}]"
                            if b["verified_accuracy"] is not None else "")
                acc = (f"{b['accuracy']:.4f}" if b['accuracy'] is not None
                       else "(in progress)")
                print(f"  {label}: {acc} "
                      f"({b['total_examples']} examples){verified}")
    conn.close()


def cmd_compare(db_path: Path, run_a: int, run_b: int,
                samples: bool = False, match: str = "task,model") -> None:
    """Diff two runs: per-benchmark accuracy deltas, optionally sample flips."""
    conn = _connect(db_path)
    for rid in (run_a, run_b):
        if not conn.execute("SELECT 1 FROM runs WHERE run_id = ?", (rid,)).fetchone():
            print(f"[ERROR] Run {rid} not found (see `lm-eval-ledger runs`)",
                  file=sys.stderr)
            sys.exit(1)

    join = "a.task = b.task AND a.fewshot_k IS b.fewshot_k"
    if match == "task,model":
        join += " AND a.model_tag = b.model_tag"

    pairs = conn.execute(
        f"""SELECT a.benchmark_id AS id_a, b.benchmark_id AS id_b,
                   a.task, a.fewshot_k, a.model_tag AS model_a, b.model_tag AS model_b,
                   COALESCE(a.verified_accuracy, a.accuracy) AS acc_a,
                   COALESCE(b.verified_accuracy, b.accuracy) AS acc_b
            FROM benchmarks a JOIN benchmarks b ON {join}
            WHERE a.run_id = ? AND b.run_id = ?
              AND (a.error IS NULL OR a.error = '')
              AND (b.error IS NULL OR b.error = '')
            ORDER BY a.benchmark_id""",
        (run_a, run_b),
    ).fetchall()

    if not pairs:
        print(f"No matching benchmarks between runs {run_a} and {run_b} "
              f"(match mode: {match}; try --match task).")
        return

    print(f"\nrun {run_a} -> run {run_b}  (match: {match})")
    print(f"{'-'*72}")
    for p in pairs:
        models = (p["model_a"] if p["model_a"] == p["model_b"]
                  else f"{p['model_a']} -> {p['model_b']}")
        delta = p["acc_b"] - p["acc_a"]
        print(f"  {p['task']}({p['fewshot_k']}) [{models}]: "
              f"{p['acc_a']:.4f} -> {p['acc_b']:.4f} ({delta:+.4f})")

    if samples:
        print(f"\nSample-level flips (score changed between runs):")
        print(f"{'-'*72}")
        any_flips = False
        for p in pairs:
            flips = conn.execute(
                """SELECT sa.sample_id,
                          COALESCE(sa.verified_score, sa.score) AS score_a,
                          COALESCE(sb.verified_score, sb.score) AS score_b,
                          json_extract(sa.responses, '$[0].extracted') AS ans_a,
                          json_extract(sb.responses, '$[0].extracted') AS ans_b,
                          sa.gold
                   FROM samples sa JOIN samples sb ON sa.sample_id = sb.sample_id
                   WHERE sa.benchmark_id = ? AND sb.benchmark_id = ?
                     AND COALESCE(sa.verified_score, sa.score)
                         != COALESCE(sb.verified_score, sb.score)
                   ORDER BY sa.sample_id""",
                (p["id_a"], p["id_b"]),
            ).fetchall()
            for f in flips:
                any_flips = True
                direction = "FIXED" if f["score_b"] > f["score_a"] else "REGRESSED"
                print(f"  [{direction}] {p['task']}({p['fewshot_k']}) "
                      f"sample {f['sample_id']}: gold={f['gold']!r} "
                      f"answered {f['ans_a']!r} -> {f['ans_b']!r} "
                      f"({f['score_a']:g} -> {f['score_b']:g})")
        if not any_flips:
            print("  (none)")
    conn.close()


def cmd_config(db_path: Path, run_id: int | None) -> None:
    """Print a run's resolved config YAML (newest run when no id given).

    Round-trips: `lm-eval-ledger config 3 > rerun.yaml` then
    `lm-eval-ledger -c rerun.yaml` reproduces the run.
    """
    conn = _connect(db_path)
    if run_id is None:
        row = conn.execute(
            "SELECT run_id, config_yaml FROM runs ORDER BY run_id DESC LIMIT 1"
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT run_id, config_yaml FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
    conn.close()
    if row is None:
        print(f"[ERROR] Run {run_id if run_id is not None else '(latest)'} "
              f"not found (see `lm-eval-ledger runs`)", file=sys.stderr)
        sys.exit(1)
    print(f"# resolved config of run {row['run_id']}")
    print(row["config_yaml"], end="")


def main_query(argv: list[str]) -> None:
    """Dispatch the `runs`, `compare`, and `config` subcommands."""
    p = argparse.ArgumentParser(prog="lm-eval-ledger")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_runs = sub.add_parser("runs", help="list runs in the ledger")
    p_runs.add_argument("--db", type=Path,
                        default=Path("results") / DEFAULT_LEDGER_NAME)

    p_cmp = sub.add_parser("compare", help="diff two runs")
    p_cmp.add_argument("run_a", type=int)
    p_cmp.add_argument("run_b", type=int)
    p_cmp.add_argument("--db", type=Path,
                       default=Path("results") / DEFAULT_LEDGER_NAME)
    p_cmp.add_argument("--samples", action="store_true",
                       help="also list samples whose score changed")
    p_cmp.add_argument("--match", choices=["task,model", "task"],
                       default="task,model",
                       help="how to pair benchmarks between the runs")

    p_cfg = sub.add_parser("config",
                           help="print a run's resolved config YAML "
                                "(pipe to a file to re-run it)")
    p_cfg.add_argument("run_id", type=int, nargs="?", default=None,
                       help="run id (default: newest run)")
    p_cfg.add_argument("--db", type=Path,
                       default=Path("results") / DEFAULT_LEDGER_NAME)

    args = p.parse_args(argv)
    if args.cmd == "runs":
        cmd_runs(args.db)
    elif args.cmd == "config":
        cmd_config(args.db, args.run_id)
    else:
        cmd_compare(args.db, args.run_a, args.run_b,
                    samples=args.samples, match=args.match)
