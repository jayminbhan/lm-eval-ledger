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
    - hle: Humanity's Last Exam (text-only; gated dataset)
    - livecodebench: LiveCodeBench code generation (executes generated code)
    - mrcr_2needle / _4needle / _8needle: MRCR long-context (partial credit)
"""

from __future__ import annotations

import sys

from .config import build_arg_parser, resolve_config


def main() -> None:
    # Ledger query subcommands (everything else is a benchmark run)
    if len(sys.argv) > 1 and sys.argv[1] in ("runs", "compare", "config"):
        from .queries import main_query
        main_query(sys.argv[1:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        from .webapp import main as serve_main
        serve_main(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "init":
        from .init_cmd import main as init_main
        init_main(sys.argv[2:])
        return

    args = build_arg_parser().parse_args()

    if args.shard is not None:
        # Multi-GPU worker: the coordinator registered the run; load its
        # resolved config from the ledger's run row.
        if not args.run_name or not args.db_path:
            print("[ERROR] worker mode requires --db-path and --run-name",
                  file=sys.stderr)
            sys.exit(1)
        from .config import config_from_resolved_yaml
        from .db import LedgerDatabase
        ledger = LedgerDatabase(args.db_path)
        row = ledger.conn.execute(
            "SELECT config_yaml FROM runs WHERE run_name = ? "
            "ORDER BY run_id DESC LIMIT 1", (args.run_name,),
        ).fetchone()
        ledger.close()
        if row is None:
            print(f"[ERROR] run {args.run_name!r} not found in {args.db_path}",
                  file=sys.stderr)
            sys.exit(1)
        cfg = config_from_resolved_yaml(row["config_yaml"])
    else:
        try:
            cfg = resolve_config(args)
        except ValueError as e:
            print(f"[ERROR] {e}", file=sys.stderr)
            sys.exit(1)

    # Interactive thinking-mode selection when the config left it unset
    # (TTY only; workers inherit the coordinator's resolved choice)
    if args.shard is None and cfg.models:
        from .thinking import maybe_prompt_thinking_mode
        maybe_prompt_thinking_mode(cfg)

    # Context-budget preflight: catch zero/thin prompt budgets and slot
    # mismatches before any GPU time is spent (workers inherit the
    # coordinator's already-checked config).
    if args.shard is None and cfg.models:
        from .preflight import preflight_context
        preflight_context(cfg)

    from .runner import run  # deferred: pulls in vLLM
    run(cfg, shard=args.shard, run_name=args.run_name)


if __name__ == "__main__":
    main()
