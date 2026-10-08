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

UNSTABLE_RATIO = 1.05  # A/A max/min above this marks bimodal/unstable timing

def problem_eps(speedups):
    """Largest A/A deviation over a problem's repeats, as a (1 + eps) factor."""
    return math.exp(max(abs(math.log(s)) for s in speedups)) - 1

def is_unstable(speedups):
    return max(speedups) / min(speedups) > UNSTABLE_RATIO

def _report(per, failed, total, done, env_start, env_end, date, repeats):
    # global eps only once every problem is done and at least half succeeded; pooled over all samples.
    # Per-problem eps = max(own largest deviation, eps_global). "eps" (= eps_global) is kept for old readers.
    enough = done == total and len(per) * 2 >= total and per
    g = eps_from_speedups([x for r in per.values() for x in r["speedups"]]) if enough else None
    out = {k: {**r, "eps": max(r["eps_raw"], g or 0.0)} for k, r in per.items()}
    m = max(r["eps"] for r in out.values()) if enough else None
    return {"eps": g, "eps_global": g, "eps_max": m, "quantile": 0.95, "repeats": repeats,
            "unstable_ratio": UNSTABLE_RATIO, "env_start": env_start, "env_end": env_end, "date": date,
            "per_problem": out, "failed": failed}

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems-root", default="problems"); ap.add_argument("--out", default="harness/noise_floor.json")
    ap.add_argument("--repeats", type=int, default=5, help="A/A runs per problem (eps pools all of them)")
    ap.add_argument("--unstable-repeats", type=int, default=20,
                    help="total A/A runs for problems whose first --repeats spread more than the unstable ratio")
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
            while len(sp) < a.repeats or (is_unstable(sp) and len(sp) < a.unstable_repeats):
                with tempfile.TemporaryDirectory() as d:
                    r = grade_problem(pdir, pdir / "baseline.cu", Path(d), trusted=True)
                if r.status != "ok" or r.speedup is None or not r.speedup > 0:
                    raise RuntimeError(f"baseline vs itself is {r.status}, speedup={r.speedup}")
                sp.append(r.speedup)
            per[pdir.name] = {"eps_raw": problem_eps(sp), "unstable": is_unstable(sp), "n_repeats": len(sp),
                              "speedups": sp, "min": min(sp), "median": float(np.median(sp)), "max": max(sp)}
            print(f"{pdir.name:28s} n {len(sp):2d}  min {min(sp):.4f}  median {np.median(sp):.4f}  max {max(sp):.4f}  "
                  f"eps_raw {per[pdir.name]['eps_raw']:.4f}{'  UNSTABLE' if is_unstable(sp) else ''}", flush=True)
        except Exception as e:
            failed[pdir.name] = f"{type(e).__name__}: {e}"
            print(f"FAIL {pdir.name}: {failed[pdir.name]}", file=sys.stderr, flush=True)
        if i == len(specs): env_end = env_record()  # post-run snapshot; the GPU is usually idle again (idle clocks)
        write_report(a.out, _report(per, failed, len(specs), i, env_start, env_end, date, a.repeats))
    rep = _report(per, failed, len(specs), len(specs), env_start, env_end, date, a.repeats)
    if rep["eps"] is None:
        print(f"ERROR: only {len(per)}/{len(specs)} problems succeeded; no eps written", file=sys.stderr); return 1
    print(f"eps_global = {rep['eps_global']:.4f}  eps_max = {rep['eps_max']:.4f}  unstable = "
          f"{[k for k, r in rep['per_problem'].items() if r['unstable']]}")
    return 1 if failed else 0

if __name__ == "__main__":
    sys.exit(main())
