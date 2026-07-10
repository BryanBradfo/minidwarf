# SPDX-License-Identifier: Apache-2.0
"""Print a problem's prompt.md to stdout.

Usage:
    python scripts/gen_prompt.py <problem_dir>

This is a thin helper for feeding a model exactly the prompt text it should
see when attempting a problem -- nothing more. `prompt.md` never contains
eval shapes or seeds (see CONTRIBUTING.md), so this script cannot leak them:
it only ever cats the file as authored.
"""
import sys
from pathlib import Path


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: gen_prompt.py <problem_dir>", file=sys.stderr)
        return 2
    problem_dir = Path(argv[0])
    prompt_path = problem_dir / "prompt.md"
    if not prompt_path.exists():
        print(f"ERROR: {prompt_path} does not exist", file=sys.stderr)
        return 1
    sys.stdout.write(prompt_path.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
