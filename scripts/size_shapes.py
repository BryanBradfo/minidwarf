# SPDX-License-Identifier: Apache-2.0
"""Measure baseline time, device memory and case-generation cost for eval shapes.

  python scripts/size_shapes.py problems/dense/sgemm 4096,4096,4096 3000,5000,3500
  python scripts/size_shapes.py --check-all          # every problem's spec.yaml eval_shapes
"""
import argparse, json, resource, sys, tempfile, time
from pathlib import Path
from minidwarf.spec import load_problem
from minidwarf.baselines import link_flags
from minidwarf.compile import compile_binary
from minidwarf.grade import N_SETS
from minidwarf.preflight import ensure_gpu_idle
from minidwarf.refcache import cached_case
from minidwarf.report import write_report
from minidwarf.runner import run_binary

MIN_MS, MAX_DEVICE_MB, WARN_CASE_S = 1.0, 2000.0, 120.0
SEED = 12345  # grade_problem's default: warms exactly the cache entries grading will use

def row_ok(median_ms, device_mb):
    return median_ms >= MIN_MS and device_mb <= MAX_DEVICE_MB

def measure(pdir, shapes):
    p = load_problem(pdir); rows = []
    with tempfile.TemporaryDirectory() as d:
        exe = compile_binary(p.root / "baseline.cu", Path(d), extra_flags=link_flags(p.baseline))
        for i, shape in enumerate(shapes):
            t0 = time.perf_counter()
            cases = [cached_case(p.root, shape, SEED + 1000 * i + k) for k in range(N_SETS)]
            case_s = time.perf_counter() - t0
            ins, exp = [c[0] for c in cases], [c[1] for c in cases]
            r = run_binary(exe, ins, shape, [e.shape for e in exp[0]], reps=20, warmup=3,  # verified, as when graded
                           expected_sets=exp, rtol=p.rtol, atol=p.atol)
            dev_mb = (sum(a.nbytes for a in ins[0]) + sum(e.nbytes for e in exp[0])) / 1e6
            rows.append({"problem": p.name, "shape": list(shape), "base_median_ms": r.median_ms,
                         "device_mb": dev_mb, "case_s": case_s,
                         "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                         "ok": row_ok(r.median_ms, dev_mb)})
            flag = "" if rows[-1]["ok"] else "  <-- FAIL"
            warn = "  (WARN: slow case generation)" if case_s > WARN_CASE_S else ""
            print(f"{p.name:28s} {str(list(shape)):32s} {r.median_ms:9.3f} ms {dev_mb:8.1f} MB {case_s:7.1f} s{flag}{warn}", flush=True)
    return rows

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("problem", nargs="?"); ap.add_argument("shapes", nargs="*")
    ap.add_argument("--check-all", action="store_true"); ap.add_argument("--problems-root", default="problems")
    ap.add_argument("--out", default="harness/shape_report.json"); ap.add_argument("--allow-busy-gpu", action="store_true")
    a = ap.parse_args(argv)
    if a.check_all:
        specs = sorted(Path(a.problems_root).glob("*/*/spec.yaml"))
        if not specs:
            print(f"ERROR: no problems/*/*/spec.yaml under {a.problems_root}", file=sys.stderr); return 2
        ensure_gpu_idle(a.allow_busy_gpu)
        rows = []
        for spec in specs:
            try:
                rows += measure(spec.parent, load_problem(spec.parent).eval_shapes)
            except Exception as e:
                rows.append({"problem": spec.parent.name, "error": f"{type(e).__name__}: {e}", "ok": False})
                print(f"FAIL {spec.parent.name}: {rows[-1]['error']}", file=sys.stderr, flush=True)
            write_report(a.out, rows)
        return 0 if all(r.get("ok") for r in rows) else 1
    if not a.problem or not a.shapes:
        ap.error("give a problem dir and at least one shape, or --check-all")
    ensure_gpu_idle(a.allow_busy_gpu)
    measure(Path(a.problem), [[int(x) for x in s.split(",")] for s in a.shapes])
    return 0

if __name__ == "__main__":
    sys.exit(main())
