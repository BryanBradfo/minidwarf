# SPDX-License-Identifier: Apache-2.0
"""A/A timing test: grade every problem's baseline against itself to measure timing noise eps."""
import argparse, datetime, json, math, sys, tempfile
from pathlib import Path
import numpy as np
from minidwarf.grade import grade_problem
from minidwarf.preflight import ensure_gpu_idle, env_record

def eps_from_speedups(speedups, q=0.95):
    """eps such that a (1 + eps) speedup exceeds the q-quantile of |log speedup| in an A/A test."""
    return math.exp(float(np.quantile([abs(math.log(s)) for s in speedups], q))) - 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems-root", default="problems"); ap.add_argument("--out", default="harness/noise_floor.json")
    ap.add_argument("--allow-busy-gpu", action="store_true")
    a = ap.parse_args()
    ensure_gpu_idle(a.allow_busy_gpu)
    per = {}
    for spec in sorted(Path(a.problems_root).glob("*/*/spec.yaml")):
        pdir = spec.parent
        with tempfile.TemporaryDirectory() as d:
            r = grade_problem(pdir, pdir / "baseline.cu", Path(d), trusted=True)
        if r.status != "ok":
            print(f"FAIL {pdir.name}: baseline vs itself is {r.status}", file=sys.stderr); return 1
        per[pdir.name] = r.speedup; print(f"{pdir.name:28s} {r.speedup:.4f}", flush=True)
    eps = eps_from_speedups(list(per.values()))
    Path(a.out).write_text(json.dumps({"eps": eps, "quantile": 0.95, "env": env_record(),
                                       "date": datetime.date.today().isoformat(), "per_problem": per}, indent=2) + "\n")
    print(f"eps = {eps:.4f}"); return 0

if __name__ == "__main__":
    sys.exit(main())
