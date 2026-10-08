# SPDX-License-Identifier: Apache-2.0
"""A/A timing test: grade every problem's baseline against itself to measure timing noise eps."""
import argparse, datetime, math, sys, tempfile
from pathlib import Path
import numpy as np
from minidwarf.grade import grade_problem
from minidwarf.preflight import ensure_gpu_idle, env_record
from minidwarf.report import write_report

def eps_from_speedups(speedups, q=0.95):
    """eps such that a (1 + eps) speedup exceeds the q-quantile of |log speedup| in an A/A test."""
    return math.exp(float(np.quantile([abs(math.log(s)) for s in speedups], q))) - 1

def _report(per, failed, total, done, env_start, env_end, date, repeats):
    # eps only once every problem is done and at least half succeeded; pooled over all repeats
    enough = done == total and len(per) * 2 >= total and per
    eps = eps_from_speedups([x for r in per.values() for x in r["speedups"]]) if enough else None
    return {"eps": eps, "quantile": 0.95, "repeats": repeats, "env_start": env_start, "env_end": env_end,
            "date": date, "per_problem": per, "failed": failed}

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems-root", default="problems"); ap.add_argument("--out", default="harness/noise_floor.json")
    ap.add_argument("--repeats", type=int, default=5, help="A/A runs per problem (eps pools all of them)")
    ap.add_argument("--allow-busy-gpu", action="store_true")
    a = ap.parse_args(argv)
    specs = sorted(Path(a.problems_root).glob("*/*/spec.yaml"))
    if not specs:
        print(f"ERROR: no problems/*/*/spec.yaml under {a.problems_root}", file=sys.stderr); return 2
    ensure_gpu_idle(a.allow_busy_gpu)
    env_start, env_end, date = env_record(), None, datetime.date.today().isoformat()
    per, failed = {}, {}
    for i, spec in enumerate(specs, 1):
        pdir = spec.parent
        try:
            sp = []
            for _ in range(a.repeats):
                with tempfile.TemporaryDirectory() as d:
                    r = grade_problem(pdir, pdir / "baseline.cu", Path(d), trusted=True)
                if r.status != "ok" or r.speedup is None or not r.speedup > 0:
                    raise RuntimeError(f"baseline vs itself is {r.status}, speedup={r.speedup}")
                sp.append(r.speedup)
            per[pdir.name] = {"speedups": sp, "min": min(sp), "median": float(np.median(sp)), "max": max(sp)}
            print(f"{pdir.name:28s} min {min(sp):.4f}  median {np.median(sp):.4f}  max {max(sp):.4f}", flush=True)
        except Exception as e:
            failed[pdir.name] = f"{type(e).__name__}: {e}"
            print(f"FAIL {pdir.name}: {failed[pdir.name]}", file=sys.stderr, flush=True)
        if i == len(specs): env_end = env_record()  # sampled right after the last measurement, while still warm
        write_report(a.out, _report(per, failed, len(specs), i, env_start, env_end, date, a.repeats))
    rep = _report(per, failed, len(specs), len(specs), env_start, env_end, date, a.repeats)
    if rep["eps"] is None:
        print(f"ERROR: only {len(per)}/{len(specs)} problems succeeded; no eps written", file=sys.stderr); return 1
    print(f"eps = {rep['eps']:.4f}")
    return 1 if failed else 0

if __name__ == "__main__":
    sys.exit(main())
