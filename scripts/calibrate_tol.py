# SPDX-License-Identifier: Apache-2.0
"""Grade each expert kernel and compare its error with the problem's tolerance.

tol_ratio = max |a - e| / (atol + rtol*|e|) over all checks; the tolerance is calibrated when
0.01 <= tol_ratio <= 1 (within [1x, 100x] of a real float32 kernel's error)."""
import argparse, json, sys, tempfile
from pathlib import Path
from minidwarf.grade import grade_problem
from minidwarf.preflight import ensure_gpu_idle
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
    return {"problem": p.name, "rtol": p.rtol, "atol": p.atol, "status": r.status,
            "tol_ratio": ratio, "verdict": verdict(ratio)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems-root", default="problems"); ap.add_argument("--out", default="harness/tolerance_report.json")
    ap.add_argument("--allow-busy-gpu", action="store_true")
    a = ap.parse_args()
    ensure_gpu_idle(a.allow_busy_gpu)
    rows = []
    for spec in sorted(Path(a.problems_root).glob("*/*/spec.yaml")):
        rows.append(calibrate(spec.parent))
        r = rows[-1]; print(f"{r['problem']:28s} rtol={r['rtol']:.1e} atol={r['atol']:.1e} ratio={r['tol_ratio']:.3e} {r['verdict']}", flush=True)
    Path(a.out).write_text(json.dumps(rows, indent=2) + "\n")
    return 0 if all(r["verdict"] in ("ok", "exact") for r in rows) else 1

if __name__ == "__main__":
    sys.exit(main())
