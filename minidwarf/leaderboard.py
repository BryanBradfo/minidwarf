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

def _fast_p(rows, p, eps):
    return _frac(rows, lambda r: r["correct"] and (r.get("speedup") or 0) >= p * (1 + eps))

def load_noise_floor(path: Path = NOISE_FLOOR) -> float:
    """Return the measured timing noise eps from noise_floor.json (0.0 if not measured yet)."""
    path = Path(path)
    return float(json.loads(path.read_text())["eps"]) if path.exists() else 0.0

def build_leaderboard(runs_dir: Path, eps: float | None = None) -> str:
    """Aggregate runs_dir/*/scores.json of the current harness version into a Markdown leaderboard."""
    runs_dir = Path(runs_dir)
    eps = load_noise_floor() if eps is None else eps
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
                       {p: _fast_p(rows, p, eps) for p in _PS}))
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
            lines.append(f"| {m} | {d} | {pct(_frac(dr, lambda r: r['correct']))} | {pct(_fast_p(dr, 1, eps))} |")
    lines += ["", f"`fast_p@p`: share of problems solved correctly with speedup >= p * (1 + eps), "
                  f"eps = {eps:.3f} (timing noise floor, harness v{HARNESS_VERSION})."]
    if skipped:
        lines.append(f"_Skipped {skipped} run(s) scored with another harness version._")
    return "\n".join(lines) + "\n"

def write_leaderboard(runs_dir: Path, out: Path = Path("LEADERBOARD.md")) -> None:
    Path(out).write_text(build_leaderboard(runs_dir))
