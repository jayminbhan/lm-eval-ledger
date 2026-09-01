# init_cmd.py
"""`lm-eval-ledger init`: materialize a working directory.

Writes a starter bench.yaml plus the bundled reference material
(reference.yaml field manual, TASKS.md task list) into the target
directory and creates the results/logs/data directories. Existing
files are never overwritten.
"""
from __future__ import annotations

import argparse
from importlib import resources
from pathlib import Path

_FILES = ("bench.yaml", "reference.yaml", "TASKS.md")
_DIRS = ("results", "logs", "data")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        prog="lm-eval-ledger init",
        description="Create a starter config, reference docs, and the "
                    "results/logs/data directories.")
    p.add_argument("directory", nargs="?", default=".",
                   help="target directory (default: current)")
    args = p.parse_args(argv)
    target = Path(args.directory)
    target.mkdir(parents=True, exist_ok=True)

    pkg = resources.files("lm_eval_ledger") / "init_data"
    for name in _FILES:
        dest = target / name
        if dest.exists():
            print(f"  [SKIP] {dest} exists, not overwriting")
            continue
        dest.write_text((pkg / name).read_text())
        print(f"  [INIT] wrote {dest}")
    for d in _DIRS:
        (target / d).mkdir(exist_ok=True)
    print(f"  [INIT] directories: {', '.join(_DIRS)}")
    print("\nNext steps:")
    print("  1. edit bench.yaml (fields: reference.yaml, tasks: TASKS.md)")
    print("  2. lm-eval-ledger" + (f" -c {target}/bench.yaml"
                                   if args.directory != "." else ""))
    print("  3. lm-eval-ledger serve   # browse results")
