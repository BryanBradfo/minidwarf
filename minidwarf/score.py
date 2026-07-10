# SPDX-License-Identifier: Apache-2.0
def fast_p(results, p):
    if not results: return 0.0
    ok = sum(1 for r in results if r.correct and (r.speedup or 0) >= p)
    return ok / len(results)

def compile_rate(results):
    if not results: return 0.0
    return sum(1 for r in results if r.status != "compile_error") / len(results)

def correctness_rate(results):
    if not results: return 0.0
    return sum(1 for r in results if r.correct) / len(results)

def summarize(results, ps=(0, 1, 2, 5)):
    return {"n": len(results), "compile_rate": compile_rate(results),
            "correctness_rate": correctness_rate(results),
            "fast_p": {p: fast_p(results, p) for p in ps}}
