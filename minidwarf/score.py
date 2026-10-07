# SPDX-License-Identifier: Apache-2.0
import math

HARNESS_VERSION = 3
def fast_p(results, p):
    """Fraction of results that are both correct and at least `p`x faster than baseline.

    `p=0` reduces to the plain correctness rate (any non-negative speedup
    counts); `p=1`, `p=2`, ... report increasingly demanding speed bars.
    """
    if not results: return 0.0
    ok = sum(1 for r in results if r.correct and (r.speedup or 0) >= p)
    return ok / len(results)

def compile_rate(results):
    """Fraction of results whose candidate compiled (status != "compile_error")."""
    if not results: return 0.0
    return sum(1 for r in results if r.status != "compile_error") / len(results)

def correctness_rate(results):
    """Fraction of results marked correct against the reference implementation."""
    if not results: return 0.0
    return sum(1 for r in results if r.correct) / len(results)

def summarize(results, ps=(0, 1, 2, 5)):
    """Build the JSON-serializable suite summary: n, compile_rate, correctness_rate, and the fast_p curve."""
    return {"n": len(results), "compile_rate": compile_rate(results),
            "correctness_rate": correctness_rate(results),
            "fast_p": {p: fast_p(results, p) for p in ps}}

def geomean_speedup(base_ms, cand_ms):
    """Geometric mean over shapes of baseline_ms / candidate_ms (None if empty, mismatched or non-positive)."""
    if not base_ms or len(base_ms) != len(cand_ms) or min(base_ms) <= 0 or min(cand_ms) <= 0:
        return None
    return math.exp(sum(math.log(b / c) for b, c in zip(base_ms, cand_ms)) / len(base_ms))
