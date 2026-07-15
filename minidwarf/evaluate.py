# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
from collections import defaultdict
from .grade import grade_problem, ProblemResult

_RANK = {"ok": 4, "wrong_output": 3, "timeout": 2, "runtime_error": 1, "compile_error": 0}

def best_result(results: list[ProblemResult]) -> ProblemResult:
    """Best-of-n: highest speedup among correct; else the least-bad status."""
    correct = [r for r in results if r.correct]
    if correct:
        return max(correct, key=lambda r: r.speedup or 0.0)
    return max(results, key=lambda r: _RANK.get(r.status, 0))

def score_run(run_dir: Path, problems_root: Path = Path("problems")) -> Path:
    """Grade every generated kernel in a run and write best-of-n scores.json."""
    run_dir = Path(run_dir)
    rows = [json.loads(l) for l in (run_dir / "results.jsonl").read_text().splitlines()]
    by_problem = defaultdict(list)
    for row in rows:
        by_problem[(row["problem"], row["dwarf"])].append(row)
    model = rows[0]["model"] if rows else "unknown"
    per_problem = []
    import tempfile
    for (name, dwarf), group in sorted(by_problem.items()):
        pdir = Path(problems_root) / dwarf / name
        graded = []
        for row in group:
            kpath = run_dir / row["kernel_path"]
            with tempfile.TemporaryDirectory() as wd:
                graded.append(grade_problem(pdir, kpath, Path(wd)))
        b = best_result(graded)
        per_problem.append({"name": name, "dwarf": dwarf, "status": b.status,
                            "correct": b.correct, "speedup": b.speedup})
    out = run_dir / "scores.json"
    out.write_text(json.dumps({"model": model, "per_problem": per_problem}, indent=2))
    return out
