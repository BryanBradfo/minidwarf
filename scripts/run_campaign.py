# SPDX-License-Identifier: Apache-2.0
"""One quota-bounded step of a multi-day evaluation campaign (meant to run daily, e.g. from cron).

  python scripts/run_campaign.py --runs-dir runs \\
      gemini-2.5-flash:3 gemini-2.5-pro:3 gemini-3.1-pro-preview:3 gemini-3.8-flash:5

For each `<config>:<samples>` (configs/eval/<config>.yaml) it resumes generation over all problems
(stopping at the daily API quota), then grades the run if the GPU is idle, then rebuilds the figures.
Run ids are `<config>-all-n<samples>`, so every day continues the same runs.
"""
import argparse, dataclasses, sys, traceback
from datetime import datetime
from pathlib import Path
from minidwarf.evalconfig import load_eval_config
from minidwarf.generate import run_generation
from minidwarf.evaluate import score_run
from minidwarf.preflight import GpuBusyError

def all_problems(root="problems"):
    return sorted((p.parent for p in Path(root).glob("*/*/spec.yaml")), key=lambda d: d.name)

def step(name, samples, runs_dir, gen=run_generation, score=score_run):
    """Generate (resume) then grade one model; returns a short status line, never raises."""
    try:
        cfg = dataclasses.replace(load_eval_config(Path("configs/eval") / f"{name}.yaml"), n_samples=samples)
        run_dir = gen(cfg, all_problems(), Path(runs_dir), f"{name}-all-n{samples}", resume=True)
    except Exception:
        return f"{name}: generation failed\n{traceback.format_exc()}"
    try:
        score(run_dir)
        return f"{name}: generated + graded"
    except GpuBusyError as e:
        return f"{name}: generated; grading skipped ({e})"
    except Exception:
        return f"{name}: grading failed\n{traceback.format_exc()}"

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="<config>:<samples>")
    ap.add_argument("--runs-dir", default="runs"); ap.add_argument("--figures", default="paper/figures")
    a = ap.parse_args(argv)
    print(f"== campaign step {datetime.now():%Y-%m-%d %H:%M}", flush=True)
    for spec in a.models:
        name, _, n = spec.partition(":")
        print(step(name, int(n or 3), a.runs_dir), flush=True)
    import subprocess
    subprocess.run([sys.executable, "scripts/make_figures.py", "--runs-dir", a.runs_dir, "--out", a.figures])
    return 0

if __name__ == "__main__":
    sys.exit(main())
