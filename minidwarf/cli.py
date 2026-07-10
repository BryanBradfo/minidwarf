# SPDX-License-Identifier: Apache-2.0
import argparse, json, tempfile
from dataclasses import asdict
from pathlib import Path
from .grade import grade_problem, ProblemResult
from .score import summarize

def main(argv=None):
    ap = argparse.ArgumentParser(prog="minidwarf")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--problem", required=True); r.add_argument("--kernel", required=True)
    s = sub.add_parser("suite"); s.add_argument("--root", required=True); s.add_argument("--kernels", required=True)
    a = ap.parse_args(argv)

    if a.cmd == "run":
        with tempfile.TemporaryDirectory() as d:
            res = grade_problem(Path(a.problem), Path(a.kernel), Path(d))
        print(json.dumps(asdict(res))); return 0

    if a.cmd == "suite":
        results = []
        for spec in sorted(Path(a.root).glob("*/*/spec.yaml")):
            pdir = spec.parent
            kern = Path(a.kernels) / (pdir.name + ".cu")
            if not kern.exists():
                results.append(ProblemResult(pdir.name, pdir.parent.name, "compile_error", False, None)); continue
            with tempfile.TemporaryDirectory() as d:
                results.append(grade_problem(pdir, kern, Path(d)))
        print(json.dumps(summarize(results))); return 0

if __name__ == "__main__":
    raise SystemExit(main())
