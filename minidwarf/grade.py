# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .spec import load_problem
from .problem_io import load_module_fn
from .compile import compile_binary, CompileError
from .runner import run_binary, RunError, RunResult
from .correctness import check_correct

@dataclass
class ProblemResult:
    name: str
    dwarf: str
    status: str
    correct: bool
    speedup: float | None

def grade_problem(problem_root, candidate_cu, work_dir, seed=12345, reps=30) -> ProblemResult:
    p = load_problem(problem_root)
    gen = load_module_fn(p.root, "inputs.py", "generate")
    ref = load_module_fn(p.root, "reference.py", "run")
    work_dir = Path(work_dir)
    try:
        cand_exe = compile_binary(Path(candidate_cu), work_dir / "cand")
        base_exe = compile_binary(p.root / "baseline.cu", work_dir / "base")
    except CompileError:
        return ProblemResult(p.name, p.dwarf, "compile_error", False, None)

    correct = True
    cand_ms_sum = base_ms_sum = 0.0
    try:
        for i, shape in enumerate(p.eval_shapes):
            ins = gen(shape, seed + i)
            expected = ref(ins, shape)
            output_shapes = [np.asarray(e).shape for e in expected]
            cr: RunResult = run_binary(cand_exe, ins, shape, output_shapes, reps)
            if not check_correct(cr.outputs, expected, p.rtol, p.atol):
                correct = False
            br: RunResult = run_binary(base_exe, ins, shape, output_shapes, reps)
            cand_ms_sum += cr.median_ms
            base_ms_sum += br.median_ms
    except RunError:
        return ProblemResult(p.name, p.dwarf, "runtime_error", False, None)

    status = "ok" if correct else "wrong_output"
    speedup = (base_ms_sum / cand_ms_sum) if cand_ms_sum > 0 else None
    return ProblemResult(p.name, p.dwarf, status, correct, speedup)
