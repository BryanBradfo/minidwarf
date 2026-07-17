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
    ev = sub.add_parser("eval")
    ev.add_argument("--config", required=True); ev.add_argument("--split", required=True)
    ev.add_argument("--runs-dir", default="runs"); ev.add_argument("--run-id", default=None)
    ev.add_argument("--only", default=None); ev.add_argument("--canned-file", default=None)
    lb = sub.add_parser("leaderboard")
    lb.add_argument("--runs-dir", default="runs"); lb.add_argument("--out", default="LEADERBOARD.md")
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

    if a.cmd == "eval":
        from .evalconfig import load_eval_config
        from .generate import resolve_split, run_generation
        from .evaluate import score_run
        from .models.registry import create_model
        cfg = load_eval_config(Path(a.config))
        dirs = resolve_split(a.split)
        if a.only:
            dirs = [d for d in dirs if d.name == a.only]
        canned = Path(a.canned_file).read_text() if a.canned_file else None
        model = create_model(cfg, canned_text=canned) if cfg.provider == "dummy" else None
        run_id = a.run_id or f"{cfg.model_name}-{a.split}"
        run_dir = run_generation(cfg, dirs, Path(a.runs_dir), run_id, model=model)
        scores = score_run(run_dir)
        print(json.dumps({"run_dir": str(run_dir), "scores": str(scores)})); return 0

    if a.cmd == "leaderboard":
        from .leaderboard import write_leaderboard
        write_leaderboard(Path(a.runs_dir), Path(a.out))
        print(str(a.out)); return 0

if __name__ == "__main__":
    raise SystemExit(main())
