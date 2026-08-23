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

from __future__ import annotations

import sys

from .config import build_arg_parser, resolve_config
from .runner import run


def main() -> None:
    args = build_arg_parser().parse_args()
    try:
        cfg = resolve_config(args)
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    run(cfg, shard=args.shard, run_name=args.run_name)


if __name__ == "__main__":
    main()
