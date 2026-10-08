# SPDX-License-Identifier: Apache-2.0
import json, warnings
from pathlib import Path
from collections import defaultdict
from .score import HARNESS_VERSION

_PS = (1, 2, 5)
NOISE_FLOOR = Path(__file__).resolve().parents[1] / "harness" / "noise_floor.json"

def _frac(items, pred):
    items = list(items)
    return (sum(1 for x in items if pred(x)) / len(items)) if items else 0.0

def _fast_p(rows, p, eps_global, per):
    """Share of rows correct with speedup >= p * (1 + eps of that problem); eps_global if not measured."""
    return _frac(rows, lambda r: r["correct"] and (r.get("speedup") or 0) >= p * (1 + per.get(r.get("name"), eps_global)))

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

def build_leaderboard(runs_dir: Path, eps: float | None = None, noise_floor: Path = NOISE_FLOOR) -> str:
    """Aggregate runs_dir/*/scores.json of the current harness version into a Markdown leaderboard.

    `eps` (a float) overrides the measured per-problem noise floor for every problem."""
    runs_dir = Path(runs_dir)
    eps_global, per, unstable = load_noise_floor(noise_floor) if eps is None else (eps, {}, set())
    eps_max = max([eps_global, *per.values()])
    latest, skipped = {}, 0  # model -> (mtime, scores dict)
    for sj in runs_dir.glob("*/scores.json"):
        data = json.loads(sj.read_text())
        if data.get("harness_version") != HARNESS_VERSION:
            skipped += 1; continue
        m = data["model"]; mt = sj.stat().st_mtime
        if m not in latest or mt > latest[m][0]:
            latest[m] = (mt, data)

    models = []
    for m, (_, data) in latest.items():
        rows = data["per_problem"]
        models.append((m, rows,
                       _frac(rows, lambda r: r["status"] != "compile_error"),
                       _frac(rows, lambda r: r["correct"]),
                       {p: _fast_p(rows, p, eps_global, per) for p in _PS}))
    models.sort(key=lambda t: t[4][1], reverse=True)  # by fast_p@1

    def pct(x): return f"{100*x:.1f}%"
    lines = ["# MiniDwarf Leaderboard", "",
             "| Model | compile% | correct% | fast_p@1 | fast_p@2 | fast_p@5 |",
             "|---|---|---|---|---|---|"]
    for m, rows, comp, corr, fp in models:
        lines.append(f"| {m} | {pct(comp)} | {pct(corr)} | {pct(fp[1])} | {pct(fp[2])} | {pct(fp[5])} |")
    lines += ["", "## Per-dwarf breakdown", "",
              "| Model | Dwarf | correct% | fast_p@1 |", "|---|---|---|---|"]
    for m, rows, *_ in models:
        by_d = defaultdict(list)
        for r in rows: by_d[r["dwarf"]].append(r)
        for d in sorted(by_d):
            dr = by_d[d]; u = sorted({r.get("name") for r in dr} & unstable)
            dl = f"{d} (unstable timing: {', '.join(u)})" if u else d
            lines.append(f"| {m} | {dl} | {pct(_frac(dr, lambda r: r['correct']))} | {pct(_fast_p(dr, 1, eps_global, per))} |")
    lines += ["", f"`fast_p@p`: share of problems solved correctly with speedup >= p * (1 + eps), where eps is "
                  f"that problem's own timing noise floor (eps_global = {eps_global:.3f} for unmeasured problems, "
                  f"eps_max = {eps_max:.3f}; harness v{HARNESS_VERSION})."]
    if unstable:
        lines.append("Unstable timing: " + ", ".join(f"{k} (eps {per.get(k, eps_global):.2f})"
                     for k in sorted(unstable)) + " — fast_p for these is reported but low-confidence.")
    if skipped:
        lines.append(f"_Skipped {skipped} run(s) scored with another harness version._")
    return "\n".join(lines) + "\n"

def write_leaderboard(runs_dir: Path, out: Path = Path("LEADERBOARD.md")) -> None:
    Path(out).write_text(build_leaderboard(runs_dir))
