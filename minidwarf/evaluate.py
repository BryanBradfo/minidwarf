# SPDX-License-Identifier: Apache-2.0
import json, os, tempfile
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from .grade import grade_problem, ProblemResult
from .report import sanitize
from .preflight import ensure_gpu_idle, env_record
from .score import HARNESS_VERSION

_RANK = {"ok": 4, "wrong_output": 3, "timeout": 2, "runtime_error": 1, "forbidden_api": 0, "compile_error": 0}

def best_result(results: list[ProblemResult]) -> ProblemResult:
    """Best-of-n: highest speedup among correct; else the least-bad status."""
    correct = [r for r in results if r.correct]
    if correct:
        return max(correct, key=lambda r: r.speedup or 0.0)
    return max(results, key=lambda r: _RANK.get(r.status, 0))

def score_run(run_dir: Path, problems_root: Path = Path("problems"), allow_busy_gpu: bool = False) -> Path:
    """Grade every generated kernel in a run (harness v3) and write best-of-n scores.json."""
    run_dir = Path(run_dir)
    (run_dir / "scores.json").unlink(missing_ok=True)
    rows = [json.loads(l) for l in (run_dir / "results.jsonl").read_text().splitlines()]
    by_problem = defaultdict(list)
    for row in rows:
        by_problem[(row["problem"], row["dwarf"])].append(row)
    model = rows[0]["model"] if rows else "unknown"
    env_start, per_problem, busy_seen = env_record(), [], {}
    for (name, dwarf), group in sorted(by_problem.items()):
        busy = ensure_gpu_idle(allow_busy_gpu)
        if busy: busy_seen[name] = busy
        pdir = Path(problems_root) / dwarf / name
        graded = []
        for row in group:
            with tempfile.TemporaryDirectory() as wd:
                graded.append(grade_problem(pdir, run_dir / row["kernel_path"], Path(wd)))
        per_problem.append(asdict(best_result(graded)))
    out = run_dir / "scores.json"
    doc = sanitize({"model": model, "harness_version": HARNESS_VERSION, "allow_busy_gpu": allow_busy_gpu,
                   "busy_seen": busy_seen, "env": {"start": env_start, "end": env_record()},
                   "per_problem": per_problem})
    tmp = run_dir / "scores.json.tmp"
    tmp.write_text(json.dumps(doc, indent=2, allow_nan=False)); os.replace(tmp, out)
    return out
