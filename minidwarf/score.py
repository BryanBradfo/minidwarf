# SPDX-License-Identifier: Apache-2.0
import json, math, warnings
from pathlib import Path

HARNESS_VERSION = 3
NOISE_FLOOR = Path(__file__).resolve().parents[1] / "harness" / "noise_floor.json"

def load_noise_floor(path: Path = NOISE_FLOOR) -> tuple[float, dict, set]:
    """Return (eps_global, {problem: eps}, {unstable problems}) from noise_floor.json ((0.0, {}, set()) if absent)."""
    path = Path(path)
    if not path.exists(): return 0.0, {}, set()
    data = json.loads(path.read_text())
    rows = {k: v for k, v in (data.get("per_problem") or {}).items() if isinstance(v, dict)}
    per = {k: float(v["eps"]) for k, v in rows.items() if v.get("eps") is not None}
    g = next((data[k] for k in ("eps_global", "eps") if data.get(k) is not None), None)
    if g is None: warnings.warn(f"{path}: no eps_global/eps value; using a 0.0 noise floor")
    return float(g or 0.0), per, {k for k, v in rows.items() if v.get("unstable")}

def _floor(eps, noise_floor):
    """(eps_global, per-problem eps): the measured floor, or `eps` uniformly when given."""
    return (float(eps), {}) if eps is not None else load_noise_floor(noise_floor)[:2]

def fast_p(results, p, eps=None, noise_floor=NOISE_FLOOR, _floors=None):
    """Fraction of results that are correct with speedup >= p * (1 + eps of that problem).

    eps is the problem's measured timing noise floor (eps_global if unmeasured), as on the leaderboard;
    a float `eps` overrides it for every problem. `p=0` reduces to the plain correctness rate.
    """
    if not results: return 0.0
    g, per = _floors or _floor(eps, noise_floor)
    ok = sum(1 for r in results if r.correct and (r.speedup or 0) >= p * (1 + per.get(r.name, g)))
    return ok / len(results)

def compile_rate(results):
    """Fraction of results whose candidate compiled (not compile_error; forbidden_api is rejected before compiling)."""
    if not results: return 0.0
    return sum(1 for r in results if r.status not in ("compile_error", "forbidden_api")) / len(results)

def correctness_rate(results):
    """Fraction of results marked correct against the reference implementation."""
    if not results: return 0.0
    return sum(1 for r in results if r.correct) / len(results)

def summarize(results, ps=(0, 1, 2, 5), eps=None, noise_floor=NOISE_FLOOR):
    """Build the JSON-serializable suite summary: n, compile_rate, correctness_rate, eps_global and the
    noise-floor-adjusted fast_p curve (same rule as the leaderboard)."""
    fl = _floor(eps, noise_floor)
    return {"n": len(results), "compile_rate": compile_rate(results),
            "correctness_rate": correctness_rate(results), "eps_global": fl[0],
            "fast_p": {p: fast_p(results, p, _floors=fl) for p in ps}}

def geomean_speedup(base_ms, cand_ms):
    """Geometric mean over shapes of baseline_ms / candidate_ms (None if empty, mismatched or non-positive)."""
    if not base_ms or len(base_ms) != len(cand_ms) or min(base_ms) <= 0 or min(cand_ms) <= 0:
        return None
    return math.exp(sum(math.log(b / c) for b, c in zip(base_ms, cand_ms)) / len(base_ms))
