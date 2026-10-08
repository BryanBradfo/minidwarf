# SPDX-License-Identifier: Apache-2.0
"""Grade each expert kernel and compare its error with the problem's tolerance.

tol_ratio = max |a - e| / (atol + rtol*|e|) over all checks; the tolerance is calibrated when
0.01 <= tol_ratio <= 1 (within [1x, 100x] of a real float32 kernel's error). Any grade status
other than "ok" (including bad calls) is a "fail" regardless of the ratio."""
import argparse, sys, tempfile
from pathlib import Path
from minidwarf.grade import grade_problem
from minidwarf.preflight import ensure_gpu_idle
from minidwarf.report import write_report
from minidwarf.spec import load_problem

def verdict(tol_ratio):
    if tol_ratio == 0: return "exact"
    if tol_ratio > 1: return "fail"
    if tol_ratio < 0.01: return "too_loose"
    return "ok"

def calibrate(pdir) -> dict:
    p = load_problem(pdir)
    with tempfile.TemporaryDirectory() as d:
        r = grade_problem(pdir, pdir / "solutions/expert_v1.cu", Path(d), reps=4, warmup=0)
    ratios = [c["tol_ratio"] for c in r.checks]
    ratio = max(ratios) if ratios and None not in ratios else float("inf")
    v = verdict(ratio) if r.status == "ok" else "fail"
    return {"problem": p.name, "rtol": p.rtol, "atol": p.atol, "status": r.status, "bad_calls": r.bad_calls,
            "tol_ratio": ratio, "verdict": v}

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems-root", default="problems"); ap.add_argument("--out", default="harness/tolerance_report.json")
    ap.add_argument("--allow-busy-gpu", action="store_true")
    a = ap.parse_args(argv)
    specs = sorted(Path(a.problems_root).glob("*/*/spec.yaml"))
    if not specs:
        print(f"ERROR: no problems/*/*/spec.yaml under {a.problems_root}", file=sys.stderr); return 2
    ensure_gpu_idle(a.allow_busy_gpu)
    rows = []
    for spec in specs:
        try:
            rows.append(calibrate(spec.parent))
            r = rows[-1]
            print(f"{r['problem']:28s} rtol={r['rtol']:.1e} atol={r['atol']:.1e} ratio={r['tol_ratio']:.3e} "
                  f"status={r['status']} bad_calls={r['bad_calls']} {r['verdict']}", flush=True)
        except Exception as e:
            rows.append({"problem": spec.parent.name, "error": f"{type(e).__name__}: {e}", "verdict": "fail"})
            print(f"FAIL {spec.parent.name}: {rows[-1]['error']}", file=sys.stderr, flush=True)
        write_report(a.out, rows)
    return 0 if all(r["verdict"] in ("ok", "exact") for r in rows) else 1

if __name__ == "__main__":
    sys.exit(main())
