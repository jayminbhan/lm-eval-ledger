# tasks/livecodebench.py
"""LiveCodeBench code-generation benchmark.

Paper: Jain et al., "LiveCodeBench: Holistic and Contamination Free
Evaluation of Large Language Models for Code", arXiv:2403.07974 (2024).
Dataset: livecodebench/code_generation_lite (official).

Competitive-programming problems continuously collected from LeetCode,
AtCoder, and Codeforces after model cutoffs (contest_date field enables
contamination-free windows). The official repo is a script-based HF dataset,
which `datasets>=3` can no longer load, so the release jsonl files are
downloaded directly from the official repo via huggingface_hub.

Evaluation EXECUTES model-generated code in local subprocesses (one per
test case, 6s timeout) - standard practice for code benchmarks, but run
inside a container/VM if you don't trust the model being evaluated.
A problem counts as solved only if every public and private test passes.

Two problem styles, matching the official harness:
- stdin/stdout: program reads stdin, output compared line-by-line
- functional (LeetCode-style): Solution().func(*args) called per test

Use a generous max_tokens (>= 2048 recommended).
"""
from __future__ import annotations

import base64
import json
import os
import pickle
import re
import subprocess
import sys
import zlib

from .base import TaskConfig, load_jsonl

_REPO_ID = "livecodebench/code_generation_lite"

# release_vN = union of the first N jsonl files (mirrors the official loader)
_ALL_FILES = ["test.jsonl", "test2.jsonl", "test3.jsonl",
              "test4.jsonl", "test5.jsonl", "test6.jsonl"]
_VERSION_FILES = {f"release_v{i}": _ALL_FILES[:i] for i in range(1, 7)}

_TEST_TIMEOUT_S = 6

_CODE_BLOCK_RE = re.compile(r"```(?:python3?|py)?\s*\n(.*?)```", re.DOTALL)
_ANY_FENCE_RE = re.compile(r"```\S*\s*\n(.*?)```", re.DOTALL)

# Prepended before the model's code, mirroring the official harness: LeetCode
# solutions routinely use List/deque/Counter/etc. without importing them.
_PRELUDE = """\
import sys as _lel_prelude_sys
_lel_prelude_sys.setrecursionlimit(600000)
import collections, functools, heapq, itertools, math, random, re, string, sys
from collections import Counter, OrderedDict, defaultdict, deque
from functools import lru_cache, cache, reduce
from heapq import heappush, heappop, heapify
from itertools import accumulate, combinations, permutations, product
from math import ceil, floor, gcd, inf, sqrt
from typing import *
try:
    from sortedcontainers import SortedList, SortedDict, SortedSet
except ImportError:
    pass
"""

# Appended below the model's code; reads JSON args (one per line) from stdin,
# calls Solution().<func> (or a bare function of that name), prints JSON result.
_FUNCTIONAL_HARNESS = """

import sys as _lel_sys, json as _lel_json
_lel_args = [_lel_json.loads(_l) for _l in _lel_sys.stdin.read().splitlines() if _l.strip()]
_lel_name = _lel_sys.argv[1]
if "Solution" in globals():
    _lel_target = getattr(Solution(), _lel_name)
else:
    _lel_target = globals()[_lel_name]
print(_lel_json.dumps(_lel_target(*_lel_args)))
"""


# ============================================================
# Loading
# ============================================================

def _load(release: str) -> list[dict]:
    """Download the official release jsonl files and concatenate them."""
    return _load_files(_VERSION_FILES[release])


def _load_files(files: list[str]) -> list[dict]:
    from huggingface_hub import hf_hub_download

    rows: list[dict] = []
    for i, fname in enumerate(files, 1):
        print(f"  [{i}/{len(files)}] Loading {_REPO_ID}/{fname} "
              f"(downloaded on first use, cached after)")
        path = hf_hub_download(_REPO_ID, fname, repo_type="dataset")
        rows.extend(load_jsonl(path))
    for row in rows:
        row["id"] = row.get("question_id", "")
    return rows


def _decode_private_tests(raw: str) -> list[dict]:
    """Private tests are JSON, or base64+zlib+pickle-compressed JSON."""
    if not raw:
        return []
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return json.loads(pickle.loads(zlib.decompress(base64.b64decode(raw.encode("utf-8")))))


# ============================================================
# Prompt / extraction
# ============================================================

def build_prompt(example: dict, fewshot_block: str) -> str:
    """Official-style codegen prompt. Ignores fewshot_block (the generic
    \\boxed{} format instruction would conflict with the code-block format)."""
    prompt = (
        "You are an expert Python programmer. You will be given a question "
        "(problem specification) and will generate a correct Python program "
        "that matches the specification and passes all tests.\n\n"
        f"### Question:\n{example['question_content'].strip()}\n\n"
    )
    starter = (example.get("starter_code") or "").strip()
    if starter:
        prompt += (
            "### Format: You will use the following starter code to write the "
            "solution to the problem and enclose your code within delimiters.\n"
            f"```python\n{starter}\n```\n\n"
        )
    else:
        prompt += (
            "### Format: Read the inputs from stdin solve the problem and "
            "write the answer to stdout (do not directly test on the sample "
            "inputs). Enclose your code within delimiters as follows.\n"
            "```python\n# YOUR CODE HERE\n```\n\n"
        )
    prompt += "### Answer: (use the provided format with backticks)\n\n"
    return prompt


def extract_gold(example: dict) -> str:
    """Pack the tests (private ones still compressed) and metadata as JSON."""
    return json.dumps({
        "public": example.get("public_test_cases", ""),
        "private": example.get("private_test_cases", ""),
        "meta": example.get("metadata", ""),
    })


def extract_gold_display(example: dict) -> str:
    """Short human-readable gold for the ledger (full tests go to gold_data)."""
    try:
        n_public = len(json.loads(example.get("public_test_cases") or "[]"))
    except (json.JSONDecodeError, ValueError):
        n_public = 0
    try:
        n_private = len(_decode_private_tests(example.get("private_test_cases", "")))
    except Exception:
        n_private = 0
    return (f"pass all {n_public + n_private} tests "
            f"({n_public} public, {n_private} private) - "
            f"{example.get('platform', '')} {example.get('question_title', '')}")


def extract_pred(model_output: str) -> str:
    """Extract the last ```python ...``` code block (official behavior);
    fall back to the last fenced block of any language tag."""
    matches = _CODE_BLOCK_RE.findall(model_output)
    if not matches:
        matches = _ANY_FENCE_RE.findall(model_output)
    return matches[-1].strip() if matches else ""


# ============================================================
# Execution-based matching
# ============================================================

def _values_match(actual, expected) -> bool:
    """Structural equality with float tolerance."""
    if isinstance(actual, float) or isinstance(expected, float):
        try:
            return abs(float(actual) - float(expected)) < 1e-6
        except (TypeError, ValueError):
            return False
    if isinstance(actual, list) and isinstance(expected, list):
        return (len(actual) == len(expected)
                and all(_values_match(a, e) for a, e in zip(actual, expected)))
    return actual == expected


def _stdout_matches(actual: str, expected: str) -> bool:
    """Line-by-line comparison, ignoring trailing whitespace; numeric
    tokens compared with float tolerance."""
    a_lines = [l.rstrip() for l in actual.strip().splitlines()]
    e_lines = [l.rstrip() for l in expected.strip().splitlines()]
    if a_lines == e_lines:
        return True
    if len(a_lines) != len(e_lines):
        return False
    for a, e in zip(a_lines, e_lines):
        if a == e:
            continue
        a_tok, e_tok = a.split(), e.split()
        if len(a_tok) != len(e_tok):
            return False
        for at, et in zip(a_tok, e_tok):
            if at == et:
                continue
            try:
                if abs(float(at) - float(et)) >= 1e-6:
                    return False
            except ValueError:
                return False
    return True


def _run_one_test(code: str, test: dict, func_name: str | None) -> bool:
    """Execute the code against one test case in a subprocess."""
    if test.get("testtype") == "functional":
        if not func_name:
            return False
        script = _PRELUDE + code + _FUNCTIONAL_HARNESS
        argv = [sys.executable, "-c", script, func_name]
    else:
        argv = [sys.executable, "-c", _PRELUDE + code]

    try:
        proc = subprocess.run(
            argv, input=test.get("input", ""), capture_output=True,
            text=True, timeout=_TEST_TIMEOUT_S,
            # untrusted code: do not inherit API keys/tokens from our env
            env={"PATH": os.environ.get("PATH", ""),
                 "PYTHONIOENCODING": "utf-8", "PYTHONHASHSEED": "0"},
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    if proc.returncode != 0:
        return False

    expected = test.get("output", "")
    if test.get("testtype") == "functional":
        try:
            return _values_match(json.loads(proc.stdout.strip()),
                                 json.loads(expected))
        except (json.JSONDecodeError, ValueError):
            return proc.stdout.strip() == expected.strip()
    return _stdout_matches(proc.stdout, expected)


def match_fn(gold: str, pred: str) -> bool:
    """A problem is solved only if the extracted code passes every test."""
    if not pred:
        return False
    data = json.loads(gold)
    try:
        tests = json.loads(data["public"] or "[]")
        tests += _decode_private_tests(data["private"])
    except Exception:
        return False
    if not tests:
        return False
    meta = json.loads(data["meta"] or "{}")
    func_name = meta.get("func_name")

    return all(_run_one_test(pred, test, func_name) for test in tests)


# ============================================================
# Task factory
# ============================================================

def get_task() -> TaskConfig:
    """Get LiveCodeBench code-generation task (release_v6)."""
    return TaskConfig(
        name="livecodebench",
        build_prompt=build_prompt,
        extract_gold=extract_gold,
        extract_gold_display=extract_gold_display,
        extract_pred=extract_pred,
        match_fn=match_fn,
        default_fewshot_k=0,
        description="LiveCodeBench (release_v6) - contamination-free code "
                    "generation; EXECUTES generated code locally; "
                    "use max_tokens >= 2048",
        load_fn=lambda: _load("release_v6"),
    )


def _load_delta(n: int) -> list[dict]:
    """Problems ADDED in release n (the upstream per-release jsonl file) -
    the dataset's own contamination-control unit."""
    ex = _load_files([_ALL_FILES[n - 1]])
    print(f"  [INFO] livecodebench release-v{n} delta: {len(ex)} problems")
    return ex


def get_task_delta(n: int) -> TaskConfig:
    """A single release-delta LiveCodeBench task (upstream file test{n})."""
    base = get_task()
    base.name = f"livecodebench_v{n}"
    base.description = (f"LiveCodeBench problems added in release_v{n} - "
                        f"upstream-defined slice; EXECUTES generated code "
                        f"locally")
    base.load_fn = lambda: _load_delta(n)
    return base
