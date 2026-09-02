# init_cmd.py
"""`lm-eval-ledger init`: materialize a working directory.

Writes template.yaml (every config field, annotated; copy it, edit
the copy, run the copy) and TASKS.md (every task with few-shot
support, sample counts, and dataset caveats), and creates the
results/logs directories. Existing files are never overwritten.
"""
from __future__ import annotations

import argparse
from importlib import resources
from pathlib import Path

_FILES = ("template.yaml", "TASKS.md")
_DIRS = ("results", "logs")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        prog="lm-eval-ledger init",
        description="Create the annotated config template (template.yaml) "
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
    tpl = f"{target}/template.yaml" if args.directory != "." else "template.yaml"
    print("\nNext steps:")
    print(f"  1. cp {tpl} bench.yaml   # your working copy (auto-discovered)")
    print("  2. edit bench.yaml (every field documented in place; "
          "task list: lm-eval-ledger --help)")
    print("  3. lm-eval-ledger          # run it (20-example smoke as shipped)")
    print("  4. lm-eval-ledger serve    # browse results")
