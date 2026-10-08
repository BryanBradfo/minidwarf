# SPDX-License-Identifier: Apache-2.0
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from .spec import load_problem
from .compile import compile_binary, compile_object, link_binary, object_hits, CompileError
from .baselines import link_flags, lib_flags
from .lint import lint_source
from .refcache import cached_case
from .runner import run_binary, RunError
from .correctness import compare
from .score import geomean_speedup

N_SETS = 4
CHECK_SEED_OFFSET = 500_000

@dataclass
class ProblemResult:
    name: str
    dwarf: str
    status: str
    correct: bool
    speedup: float | None
    checks: list = field(default_factory=list)
    timings: list = field(default_factory=list)
    lint: list = field(default_factory=list)
    bad_calls: int = 0
    baseline_bad_calls: int = 0  # non-zero means the problem/harness is broken, not the candidate

def _cases(root, shape, seed0):
    cases = [cached_case(root, shape, seed0 + d) for d in range(N_SETS)]
    return [c[0] for c in cases], [c[1] for c in cases]

def _checks(run, expected_sets, shape, p, kind):
    return [{"kind": kind, "shape": list(shape), "data_set": d, **compare(outs, exp, p.rtol, p.atol)}
            for d, (outs, exp) in enumerate(zip(run.outputs, expected_sets))]

def _iqr(t):
    q25, q75 = np.percentile(t, [25, 75]); return float(q75 - q25)

def grade_problem(problem_root, candidate_cu, work_dir, seed=12345, reps=20, warmup=3,
                  timeout_s=60, trusted=False) -> ProblemResult:
    """Lint, compile, check and time one candidate against a problem (harness v3).

    Status is one of "ok", "wrong_output", "compile_error", "runtime_error", "timeout" or
    "forbidden_api"; a misbehaving candidate never raises. `trusted=True` skips the lints and links
    the candidate like the baseline (used to grade baseline.cu against itself). Every candidate call is
    verified by the driver; any bad call makes the candidate incorrect."""
    p = load_problem(problem_root)
    work_dir = Path(work_dir)
    fail = lambda status, **kw: ProblemResult(p.name, p.dwarf, status, False, None, **kw)
    if not trusted:
        hits = lint_source(Path(candidate_cu).read_bytes().decode(errors="replace"), p.allowed_libs)
        if hits: return fail("forbidden_api", lint=hits)
    try:
        base_flags = link_flags(p.baseline)
        if trusted:
            cand_exe = compile_binary(Path(candidate_cu), work_dir / "cand", extra_flags=base_flags)
        else:  # compile once with the real flags, check that exact object, then link it
            obj = compile_object(Path(candidate_cu), work_dir / "cand")
            hits = object_hits(obj, p.allowed_libs)
            if hits: return fail("forbidden_api", lint=hits)
            cand_exe = link_binary(obj, work_dir / "cand", extra_flags=lib_flags(p.allowed_libs))
        base_exe = compile_binary(p.root / "baseline.cu", work_dir / "base", extra_flags=base_flags)
    except (CompileError, subprocess.TimeoutExpired):
        return fail("compile_error")

    checks, timings, bad, base_bad = [], [], 0, 0
    def cand(ins, exp, shape, shapes, n_reps, n_warm):
        nonlocal bad
        r = run_binary(cand_exe, ins, shape, shapes, n_reps, n_warm, timeout_s,
                       expected_sets=exp, rtol=p.rtol, atol=p.atol)
        if r.n_bad_calls is None: raise RunError("driver did not report n_bad_calls")
        bad += r.n_bad_calls
        return r
    try:
        for i, shape in enumerate(p.check_shapes):
            ins, exp = _cases(p.root, shape, seed + CHECK_SEED_OFFSET + 1000 * i)
            shapes = [e.shape for e in exp[0]]
            checks += _checks(cand(ins, exp, shape, shapes, N_SETS, 0), exp, shape, p, "check")
        for i, shape in enumerate(p.eval_shapes):
            ins = exp = None  # free the previous shape's data before loading the next
            ins, exp = _cases(p.root, shape, seed + 1000 * i)
            shapes = [e.shape for e in exp[0]]
            # ABBA order; check each candidate run at once and keep only times, so at most one
            # run's outputs are alive next to ins/exp (multi-GB at the large eval shapes)
            ct, bt = [], []
            for who in ("c", "b", "b", "c"):
                if who == "c":
                    r = cand(ins, exp, shape, shapes, reps, warmup)
                    checks += _checks(r, exp, shape, p, "eval"); ct += r.times_ms
                else:
                    # verified like the candidate (result unused) so both binaries see the same per-call
                    # host work between timed reps; otherwise GPU clock state differs and A/A drifts 10-30%
                    r = run_binary(base_exe, ins, shape, shapes, reps, warmup, timeout_s,
                                   expected_sets=exp, rtol=p.rtol, atol=p.atol); bt += r.times_ms
                    if r.n_bad_calls is None: raise RunError("driver did not report n_bad_calls (baseline)")
                    base_bad += r.n_bad_calls
                del r
            timings.append({"shape": list(shape), "cand_median_ms": float(np.median(ct)), "cand_iqr_ms": _iqr(ct),
                            "base_median_ms": float(np.median(bt)), "base_iqr_ms": _iqr(bt)})
    except RunError:
        return fail("runtime_error", checks=checks, timings=timings, bad_calls=bad)
    except subprocess.TimeoutExpired:
        return fail("timeout", checks=checks, timings=timings, bad_calls=bad)

    if base_bad:  # the baseline itself disagrees with the reference: no trustworthy verdict or speedup
        return fail("runtime_error", checks=checks, timings=timings, bad_calls=bad, baseline_bad_calls=base_bad)
    correct = all(c["passed"] for c in checks) and bad == 0
    speedup = geomean_speedup([t["base_median_ms"] for t in timings], [t["cand_median_ms"] for t in timings])
    return ProblemResult(p.name, p.dwarf, "ok" if correct else "wrong_output", correct, speedup, checks, timings,
                         bad_calls=bad)
