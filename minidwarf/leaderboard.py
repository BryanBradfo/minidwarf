# SPDX-License-Identifier: Apache-2.0
import json
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

def load_noise_floor(path: Path = NOISE_FLOOR) -> tuple[float, dict]:
    """Return (eps_global, {problem: eps}) from noise_floor.json ((0.0, {}) if not measured yet)."""
    path = Path(path)
    if not path.exists(): return 0.0, {}
    data = json.loads(path.read_text())
    per = {k: float(v["eps"]) for k, v in (data.get("per_problem") or {}).items() if isinstance(v, dict) and "eps" in v}
    return float(data.get("eps_global", data["eps"])), per

def build_leaderboard(runs_dir: Path, eps: float | None = None, noise_floor: Path = NOISE_FLOOR) -> str:
    """Aggregate runs_dir/*/scores.json of the current harness version into a Markdown leaderboard.

    `eps` (a float) overrides the measured per-problem noise floor for every problem."""
    runs_dir = Path(runs_dir)
    eps_global, per = load_noise_floor(noise_floor) if eps is None else (eps, {})
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
            dr = by_d[d]
            lines.append(f"| {m} | {d} | {pct(_frac(dr, lambda r: r['correct']))} | {pct(_fast_p(dr, 1, eps_global, per))} |")
    lines += ["", f"`fast_p@p`: share of problems solved correctly with speedup >= p * (1 + eps), where eps is "
                  f"that problem's own timing noise floor (eps_global = {eps_global:.3f} for unmeasured problems, "
                  f"eps_max = {eps_max:.3f}; harness v{HARNESS_VERSION})."]
    if skipped:
        lines.append(f"_Skipped {skipped} run(s) scored with another harness version._")
    return "\n".join(lines) + "\n"

def write_leaderboard(runs_dir: Path, out: Path = Path("LEADERBOARD.md")) -> None:
    Path(out).write_text(build_leaderboard(runs_dir))
