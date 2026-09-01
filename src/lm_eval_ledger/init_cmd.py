# init_cmd.py
"""`lm-eval-ledger init`: materialize a working directory.

Writes reference.yaml - the fully annotated config that doubles as
the field manual and a runnable 20-example smoke test - and creates
the results/logs directories. Existing files are never overwritten.
"""
from __future__ import annotations

import argparse
from importlib import resources
from pathlib import Path

_FILES = ("reference.yaml",)
_DIRS = ("results", "logs")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        prog="lm-eval-ledger init",
        description="Create the annotated starter config (reference.yaml) "
                    "and the results/logs directories.")
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
    cfg = f"{target}/reference.yaml" if args.directory != "." else "reference.yaml"
    print("\nNext steps:")
    print(f"  1. edit {cfg} (every field documented in place; task list: "
          f"lm-eval-ledger --help)")
    print(f"  2. lm-eval-ledger -c {cfg}   # 20-example smoke run as shipped")
    print("  3. lm-eval-ledger serve        # browse results")
