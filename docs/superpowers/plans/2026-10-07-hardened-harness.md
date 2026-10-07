# Hardened Harness (MiniDwarf v3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the v2 grading harness with one that closes known LLM-kernel reward hacks, times ≥ 1 ms workloads robustly, and emits per-check/per-shape detail for the failure taxonomy.

**Architecture:** Hardening lives in the C++ driver (`harness/driver.cu`: D = 4 rotating data sets in the same buffers, NaN-poisoned outputs, L2 flush, device-wide sync inside the timer) orchestrated by Python (`grade.py`: static lint, library-free candidate link, check shapes, ABBA ordering, geomean speedup). New pure-Python modules (`lint.py`, `preflight.py`, `refcache.py`) are GPU-less and CI-tested; nvcc-gated tests prove the defenses with red-team kernels.

**Tech Stack:** Python 3.11+, NumPy, PyYAML, pytest, CUDA 13.x / nvcc (`sm_120`), `nvidia-smi`.

**Spec:** `docs/superpowers/specs/2026-10-07-hardened-harness-design.md`

## Global Constraints

- Every new source file starts with `# SPDX-License-Identifier: Apache-2.0` (.py) or `// SPDX-License-Identifier: Apache-2.0` (.cu).
- English only; terse module style (single-line guards, minimal docstrings) matching `minidwarf/*.py`.
- `prompt.md` must NEVER contain `eval_shapes`, `check_shapes` or seeds.
- Kernel ABI unchanged: `extern "C" void minidwarf_solve(const void* const* inputs, void* const* outputs, const long* dims, int n_dims);` — all element types are 4 bytes (float32/int32); outputs always float32.
- D = 4 data sets (`N_SETS = 4`), R = 20 timed reps, W = 3 warm-up reps.
- Eval shapes: baseline ≥ 1 ms per call on RTX 5070 Laptop (`sm_120`); device memory ≤ 2000 MB per problem.
- Tolerance calibration band: `0.01 ≤ tol_ratio ≤ 1` (bit-exact experts, `tol_ratio == 0`, exempt).
- Leaderboard: `fast_p@p` counts a correct kernel only if `speedup ≥ p · (1 + ε)`.
- `HARNESS_VERSION = 3`.
- The GPU is shared with other workloads: Tasks 1–6 need no GPU. Before any nvcc-gated test run or timing (Tasks 7–12), check `nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader` is empty; if not, stop and wait — never kill other users' jobs.
- Test commands need `export PATH=/usr/local/cuda/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda/lib64` for nvcc-gated tests.
- `pytest -k <expr>` that matches nothing exits 5; never use an empty `-k` in CI.

## Review Focus

1. **Legitimate identifiers that contain a banned word** (`cube`, `cubic`, `system_size`, a comment mentioning cuBLAS) — must not be flagged `forbidden_api`. Test: Task 1 `test_clean_source_passes`.
2. **Stale or corrupt cache entries** (reference edited after caching; half-written `.npz` from a killed run) — must never serve wrong expected outputs or crash. Tests: Task 5 `test_key_changes_with_seed_shape_and_source`, `test_corrupt_cache_file_is_recomputed`.
3. **`nvidia-smi` absent or printing odd values** (`[N/A]`, "No running processes found") — preflight must not crash or report a phantom busy GPU. Tests: Task 4 `test_parse_compute_apps`, `test_parse_gpu_query_handles_na`, `test_ensure_gpu_idle_without_nvidia_smi`.
4. **Candidate source that is not valid UTF-8** (LLM output with stray bytes) — must be linted and graded, not raise. Test: Task 8 `test_lint_runs_before_compile_on_non_utf8_source`.
5. **Float64 references that blow host RAM at the resized shapes** (N² N-body matrices at N ≈ 32k = 8.6 GB per temporary) — must be rewritten in chunks with identical math. Test: Task 12 `test_chunked_reference_matches_full` (per rewritten problem).

---

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `minidwarf/lint.py` (new) | Static forbidden-API scan of candidate source | 1 |
| `minidwarf/baselines.py` | + `LIB_FLAGS`, `lib_flags()` | 1 |
| `minidwarf/spec.py` | + optional `check_shapes`, `allowed_libs` | 2 |
| `minidwarf/score.py` | + `HARNESS_VERSION`, `geomean_speedup()` | 3 |
| `minidwarf/leaderboard.py` | version filter, noise-floor rule | 3 |
| `minidwarf/preflight.py` (new) | GPU-busy check, environment record | 4 |
| `minidwarf/refcache.py` (new) | On-disk cache of `(inputs, expected)` cases | 5 |
| `problems/*/*/spec.yaml` | + `check_shapes` (Task 6), resized `eval_shapes` / tolerances (Task 12) | 6, 12 |
| `harness/driver.cu` | v3 timing protocol | 7 |
| `minidwarf/runner.py` | multi-set `run_binary`, richer `RunResult` | 7 |
| `minidwarf/correctness.py` | + `compare()` per-check detail | 8 |
| `minidwarf/grade.py` | v3 grading flow, `ProblemResult` detail fields | 8 |
| `tests/fixtures/redteam/*.cu` (new) | adversarial kernels | 9 |
| `minidwarf/evaluate.py`, `minidwarf/cli.py` | preflight, env record, schema, `score` subcommand | 10 |
| `scripts/size_shapes.py`, `scripts/noise_floor.py`, `scripts/calibrate_tol.py` (new) | measurement tooling | 11 |
| `harness/shape_report.json`, `harness/noise_floor.json`, `harness/tolerance_report.json` (new, generated) | committed measurements | 12 |
| `README.md`, `CLAUDE.md`, `.github/workflows/ci.yml` | docs, CI test list | 1–6, 13 |

---

### Task 1: Static lint and library link flags

**Files:**
- Create: `minidwarf/lint.py`
- Modify: `minidwarf/baselines.py`
- Test: `tests/test_lint.py` (new), `tests/test_baselines.py`
- Modify: `.github/workflows/ci.yml` (add `tests/test_lint.py`)

**Interfaces:**
- Produces: `lint_source(src: str, allowed_libs=()) -> list[str]` (sorted rule names; empty = clean). Rule names: `cublas`, `cusparse`, `cufft`, `curand`, `cusolver`, `thrust`, `cub`, `file_io`, `process`, `dynamic_loading`, `environment`.
- Produces: `LIB_FLAGS: dict[str, list[str]]`, `lib_flags(libs) -> list[str]` (raises `ValueError` on unknown lib).

- [ ] **Step 1: Write the failing tests**

`tests/test_lint.py`:
```python
# SPDX-License-Identifier: Apache-2.0
from minidwarf.lint import lint_source

CLEAN = '''#include <cuda_runtime.h>
// we do not use cublas or fopen here
/* nor cusparse */
__global__ void k(float* x){ float cube = x[0]*x[0]*x[0]; long system_size = 3; x[0] = cube + system_size; }
extern "C" void minidwarf_solve(const void* const* i, void* const* o, const long* d, int n){
  const char* s = "cublas fopen( system("; (void)s; }
'''

def test_clean_source_passes():
    assert lint_source(CLEAN) == []

def test_cublas_include_and_call_flagged():
    assert lint_source("#include <cublas_v2.h>\nvoid f(){ cublasHandle_t h; cublasCreate(&h); }") == ["cublas"]

def test_quoted_include_flagged():
    assert lint_source('#include "cusparse.h"\n') == ["cusparse"]

def test_thrust_and_cub_flagged():
    assert lint_source("void f(){ thrust::sort(a, b); cub::DeviceReduce::Sum(); }") == ["cub", "thrust"]

def test_file_process_dlopen_getenv_flagged():
    src = 'void f(){ FILE* f = fopen("x", "r"); system("ls"); void* h = dlopen("a", 0); getenv("HOME"); }'
    assert lint_source(src) == ["dynamic_loading", "environment", "file_io", "process"]

def test_allowed_lib_not_flagged():
    src = "#include <curand_kernel.h>\n"
    assert lint_source(src) == ["curand"]
    assert lint_source(src, allowed_libs=["curand"]) == []
```

Append to `tests/test_baselines.py`:
```python
import pytest
from minidwarf.baselines import lib_flags

def test_lib_flags():
    assert lib_flags([]) == []
    assert lib_flags(["curand"]) == ["-lcurand"]
    with pytest.raises(ValueError):
        lib_flags(["mkl"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_lint.py tests/test_baselines.py -v`
Expected: FAIL (`ModuleNotFoundError: minidwarf.lint`, `ImportError: lib_flags`).

- [ ] **Step 3: Implement**

`minidwarf/lint.py`:
```python
# SPDX-License-Identifier: Apache-2.0
import re

_LIBS = {"cublas": r"\bcublas\w*", "cusparse": r"\bcusparse\w*", "cufft": r"\bcufft\w*",
         "curand": r"\bcurand\w*", "cusolver": r"\bcusolver\w*"}
_ALWAYS = {
    "thrust": r"\bthrust\s*::|<thrust/",
    "cub": r"\bcub\s*::|<cub/",
    "file_io": r"\b(fopen|freopen|fread|ifstream|ofstream|fstream)\b|\bopen\s*\(",
    "process": r"\b(system|popen|fork|execl|execlp|execle|execv|execvp|execvpe)\s*\(",
    "dynamic_loading": r"\b(dlopen|dlsym)\s*\(",
    "environment": r"\bgetenv\s*\(",
}
_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
_STRING = re.compile(r'"(?:\\.|[^"\\\n])*"')

def _strip(src: str) -> str:
    # Drop comments everywhere and string literals outside preprocessor lines (#include "x.h" stays visible).
    src = _COMMENT.sub(" ", src)
    return "\n".join(l if l.lstrip().startswith("#") else _STRING.sub('""', l) for l in src.splitlines())

def lint_source(src: str, allowed_libs=()) -> list[str]:
    """Return the sorted names of forbidden-API rules matched in `src` (empty when clean)."""
    code = _strip(src)
    rules = {k: v for k, v in _LIBS.items() if k not in set(allowed_libs)} | _ALWAYS
    return sorted(k for k, pat in rules.items() if re.search(pat, code))
```

`minidwarf/baselines.py` — append:
```python
LIB_FLAGS = {"cublas": ["-lcublas"], "cusparse": ["-lcusparse"], "cufft": ["-lcufft"],
             "curand": ["-lcurand"], "cusolver": ["-lcusolver"]}

def lib_flags(libs) -> list[str]:
    """Return the nvcc link flags for a candidate's `allowed_libs` (raises on an unknown library)."""
    out = []
    for lib in libs:
        if lib not in LIB_FLAGS:
            raise ValueError(f"unknown library: {lib!r}")
        out += LIB_FLAGS[lib]
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_lint.py tests/test_baselines.py -v`
Expected: all PASS.

- [ ] **Step 5: Add `tests/test_lint.py` to the CI pytest list** in `.github/workflows/ci.yml` (first `python -m pytest` line, after `tests/test_baselines.py`).

- [ ] **Step 6: Commit**
```bash
git add minidwarf/lint.py minidwarf/baselines.py tests/test_lint.py tests/test_baselines.py .github/workflows/ci.yml
git commit -m "Add static forbidden-API lint and per-library link flags"
```

---

### Task 2: Optional `check_shapes` and `allowed_libs` in spec.yaml

**Files:**
- Modify: `minidwarf/spec.py`
- Test: `tests/test_spec.py`

**Interfaces:**
- Consumes: `LIB_FLAGS` (Task 1).
- Produces: `Problem.check_shapes: list[list[int]]` (default `[]`), `Problem.allowed_libs: list[str]` (default `[]`); `load_problem` raises `ValueError` for an `allowed_libs` entry not in `LIB_FLAGS`.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_spec.py`:
```python
BASE = ("name: x\ndwarf: d\ndifficulty: easy\nrtol: 1.0e-5\natol: 1.0e-6\n"
        "n_inputs: 1\nn_outputs: 1\neval_shapes: [[4]]\nbaseline: author_kernel\n")

def test_optional_fields_default_empty(tmp_path):
    (tmp_path / "spec.yaml").write_text(BASE)
    p = load_problem(tmp_path)
    assert p.check_shapes == [] and p.allowed_libs == []

def test_optional_fields_parsed(tmp_path):
    (tmp_path / "spec.yaml").write_text(BASE + "check_shapes: [[1], [33]]\nallowed_libs: [curand]\n")
    p = load_problem(tmp_path)
    assert p.check_shapes == [[1], [33]] and p.allowed_libs == ["curand"]

def test_unknown_allowed_lib_raises(tmp_path):
    (tmp_path / "spec.yaml").write_text(BASE + "allowed_libs: [mkl]\n")
    with pytest.raises(ValueError):
        load_problem(tmp_path)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_spec.py -v`
Expected: FAIL (`AttributeError: 'Problem' object has no attribute 'check_shapes'`).

- [ ] **Step 3: Implement** — in `minidwarf/spec.py`:
  - change the import to `from dataclasses import dataclass, field` and add `from .baselines import LIB_FLAGS`;
  - add to `Problem` after `baseline: str`:
```python
    check_shapes: list[list[int]] = field(default_factory=list)
    allowed_libs: list[str] = field(default_factory=list)
```
  - in `load_problem`, before `return`:
```python
    allowed = list(data.get("allowed_libs") or [])
    unknown = [l for l in allowed if l not in LIB_FLAGS]
    if unknown:
        raise ValueError(f"{root}/spec.yaml: unknown allowed_libs {unknown}")
```
  - and pass to the constructor:
```python
        check_shapes=[list(map(int, s)) for s in data.get("check_shapes") or []],
        allowed_libs=allowed,
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_spec.py tests/test_splits.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**
```bash
git add minidwarf/spec.py tests/test_spec.py
git commit -m "Parse optional check_shapes and allowed_libs from spec.yaml"
```

---

### Task 3: Harness version, geomean speedup, leaderboard filtering and noise floor

**Files:**
- Modify: `minidwarf/score.py`, `minidwarf/leaderboard.py`
- Test: `tests/test_score.py`, `tests/test_leaderboard.py`

**Interfaces:**
- Produces: `HARNESS_VERSION = 3` (in `minidwarf/score.py`); `geomean_speedup(base_ms: list[float], cand_ms: list[float]) -> float | None`.
- Produces: `load_noise_floor(path=NOISE_FLOOR) -> float`; `build_leaderboard(runs_dir, eps: float | None = None) -> str` (eps `None` → read `harness/noise_floor.json`, else 0.0). `NOISE_FLOOR = <repo>/harness/noise_floor.json` with JSON key `"eps"`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_score.py`:
```python
import pytest
from minidwarf.score import geomean_speedup, HARNESS_VERSION

def test_geomean_speedup():
    assert geomean_speedup([2.0, 8.0], [1.0, 1.0]) == pytest.approx(4.0)
    assert geomean_speedup([1.0], [4.0]) == pytest.approx(0.25)

def test_geomean_speedup_invalid_is_none():
    assert geomean_speedup([], []) is None
    assert geomean_speedup([1.0], [0.0]) is None
    assert geomean_speedup([1.0, 2.0], [1.0]) is None

def test_harness_version():
    assert HARNESS_VERSION == 3
```

Replace `tests/test_leaderboard.py` with:
```python
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
from minidwarf.leaderboard import build_leaderboard, load_noise_floor
from minidwarf.score import HARNESS_VERSION


def _write(run, model, rows, version=HARNESS_VERSION):
    run.mkdir(parents=True)
    data = {"model": model, "per_problem": rows}
    if version is not None:
        data["harness_version"] = version
    (run / "scores.json").write_text(json.dumps(data))

def _row(sp, dwarf="dense"):
    return {"name": "p1", "dwarf": dwarf, "status": "ok", "correct": True, "speedup": sp}


def test_leaderboard_has_rows_and_dwarf_breakdown(tmp_path):
    _write(tmp_path / "a", "modelA", [
        _row(3.0),
        {"name": "p2", "dwarf": "sparse", "status": "wrong_output", "correct": False, "speedup": None},
    ])
    md = build_leaderboard(tmp_path, eps=0.0)
    assert "modelA" in md
    assert "fast_p@1" in md and "compile" in md
    assert "dense" in md and "sparse" in md

def test_old_harness_runs_are_skipped(tmp_path):
    _write(tmp_path / "old", "oldModel", [_row(3.0)], version=None)
    _write(tmp_path / "new", "newModel", [_row(3.0)])
    md = build_leaderboard(tmp_path, eps=0.0)
    assert "newModel" in md and "oldModel" not in md and "Skipped 1 run" in md

def test_noise_floor_raises_the_bar(tmp_path):
    _write(tmp_path / "a", "m", [_row(1.03)])
    assert "| m | 100.0% | 100.0% | 100.0% |" in build_leaderboard(tmp_path, eps=0.0)
    assert "| m | 100.0% | 100.0% | 0.0% |" in build_leaderboard(tmp_path, eps=0.05)

def test_load_noise_floor(tmp_path):
    assert load_noise_floor(tmp_path / "missing.json") == 0.0
    (tmp_path / "nf.json").write_text(json.dumps({"eps": 0.04}))
    assert load_noise_floor(tmp_path / "nf.json") == 0.04
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_score.py tests/test_leaderboard.py -v`
Expected: FAIL (`ImportError: geomean_speedup` / `load_noise_floor`).

- [ ] **Step 3: Implement**

`minidwarf/score.py` — add at top (after the SPDX line) `import math` and:
```python
HARNESS_VERSION = 3

def geomean_speedup(base_ms, cand_ms):
    """Geometric mean over shapes of baseline_ms / candidate_ms (None if empty, mismatched or non-positive)."""
    if not base_ms or len(base_ms) != len(cand_ms) or min(base_ms) <= 0 or min(cand_ms) <= 0:
        return None
    return math.exp(sum(math.log(b / c) for b, c in zip(base_ms, cand_ms)) / len(base_ms))
```

`minidwarf/leaderboard.py` — replace whole file:
```python
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
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_score.py tests/test_leaderboard.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**
```bash
git add minidwarf/score.py minidwarf/leaderboard.py tests/test_score.py tests/test_leaderboard.py
git commit -m "Add harness version, geomean speedup, and noise-floor leaderboard rule"
```

---

### Task 4: GPU preflight and environment record

**Files:**
- Create: `minidwarf/preflight.py`
- Test: `tests/test_preflight.py` (new)
- Modify: `.github/workflows/ci.yml` (add `tests/test_preflight.py`)

**Interfaces:**
- Produces: `GpuBusyError(RuntimeError)`; `parse_compute_apps(text) -> list[str]`; `parse_gpu_query(text) -> dict` (keys `name, driver_version, clocks.sm, clocks.mem, temperature.gpu`; `[N/A]` → `None`; malformed → `{}`); `busy_processes() -> list[str]`; `ensure_gpu_idle(allow_busy=False) -> list[str]`; `env_record() -> dict`. All shell calls go through module-level `_smi(args) -> str | None` (monkeypatchable; `None` when `nvidia-smi` is missing or fails).

- [ ] **Step 1: Write the failing tests** — `tests/test_preflight.py`:
```python
# SPDX-License-Identifier: Apache-2.0
import pytest
import minidwarf.preflight as pf
from minidwarf.preflight import parse_compute_apps, parse_gpu_query, ensure_gpu_idle, GpuBusyError

def test_parse_compute_apps():
    assert parse_compute_apps("2167609, python\n123, ollama\n") == ["2167609, python", "123, ollama"]
    assert parse_compute_apps("") == []
    assert parse_compute_apps("No running processes found\n") == []

def test_parse_gpu_query_handles_na():
    d = parse_gpu_query("NVIDIA GeForce RTX 5070 Laptop GPU, 580.178.04, 1980, [N/A], 49\n")
    assert d["name"].startswith("NVIDIA") and d["clocks.mem"] is None and d["temperature.gpu"] == "49"
    assert parse_gpu_query("garbage") == {}

def test_ensure_gpu_idle_raises_when_busy(monkeypatch):
    monkeypatch.setattr(pf, "_smi", lambda args: "42, python\n")
    with pytest.raises(GpuBusyError):
        ensure_gpu_idle()
    assert ensure_gpu_idle(allow_busy=True) == ["42, python"]

def test_ensure_gpu_idle_without_nvidia_smi(monkeypatch):
    monkeypatch.setattr(pf, "_smi", lambda args: None)
    assert ensure_gpu_idle() == [] and pf.env_record() == {}
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_preflight.py -v`
Expected: FAIL (`ModuleNotFoundError: minidwarf.preflight`).

- [ ] **Step 3: Implement** — `minidwarf/preflight.py`:
```python
# SPDX-License-Identifier: Apache-2.0
import shutil, subprocess

class GpuBusyError(RuntimeError): ...

_GPU_FIELDS = ["name", "driver_version", "clocks.sm", "clocks.mem", "temperature.gpu"]

def _smi(args):
    if not shutil.which("nvidia-smi"): return None
    r = subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, timeout=30)
    return r.stdout if r.returncode == 0 else None

def parse_compute_apps(text: str) -> list[str]:
    """Parse `nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader` output."""
    return [l.strip() for l in text.splitlines() if l.strip() and not l.strip().lower().startswith("no running")]

def parse_gpu_query(text: str) -> dict:
    """Parse one CSV line of the _GPU_FIELDS query ('[N/A]' -> None; malformed -> {})."""
    line = next((l for l in text.splitlines() if l.strip()), "")
    vals = [v.strip() for v in line.split(",")]
    if len(vals) != len(_GPU_FIELDS): return {}
    return {k: (None if v in ("[N/A]", "N/A", "") else v) for k, v in zip(_GPU_FIELDS, vals)}

def busy_processes() -> list[str]:
    out = _smi(["--query-compute-apps=pid,process_name", "--format=csv,noheader"])
    return parse_compute_apps(out) if out else []

def ensure_gpu_idle(allow_busy: bool = False) -> list[str]:
    """Raise GpuBusyError if another process computes on the GPU (unless allow_busy); return those processes."""
    procs = busy_processes()
    if procs and not allow_busy:
        raise GpuBusyError("GPU busy, refusing to time kernels: " + "; ".join(procs))
    return procs

def env_record() -> dict:
    """GPU name, driver, SM/memory clocks and temperature right now ({} if nvidia-smi is unavailable)."""
    out = _smi(["--query-gpu=" + ",".join(_GPU_FIELDS), "--format=csv,noheader,nounits"])
    return parse_gpu_query(out) if out else {}
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_preflight.py -v`
Expected: all PASS.

- [ ] **Step 5: Add `tests/test_preflight.py` to the CI pytest list.**

- [ ] **Step 6: Commit**
```bash
git add minidwarf/preflight.py tests/test_preflight.py .github/workflows/ci.yml
git commit -m "Add GPU-busy preflight and environment record"
```

---

### Task 5: On-disk case cache

**Files:**
- Create: `minidwarf/refcache.py`
- Test: `tests/test_refcache.py` (new)
- Modify: `.github/workflows/ci.yml` (add `tests/test_refcache.py`)

**Interfaces:**
- Produces: `cache_dir() -> Path` (`$MINIDWARF_CACHE` or `~/.cache/minidwarf`); `case_key(problem_root, shape, seed) -> str`; `cached_case(problem_root, shape, seed) -> tuple[list[np.ndarray], list[np.ndarray]]` = `(inputs, expected)` with expected as float32. Cache file: `cache_dir()/cases/<problem name>/<key>.npz`.

- [ ] **Step 1: Write the failing tests** — `tests/test_refcache.py`:
```python
# SPDX-License-Identifier: Apache-2.0
import shutil
from pathlib import Path
import numpy as np
from minidwarf.refcache import cached_case, case_key

FIX = Path(__file__).parent / "fixtures/smoke/vector_add"

def test_cached_case_matches_direct_and_hits_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("MINIDWARF_CACHE", str(tmp_path))
    ins, outs = cached_case(FIX, [1000], 7)
    np.testing.assert_array_equal(outs[0], ins[0] + ins[1])
    assert len(list(tmp_path.rglob("*.npz"))) == 1
    ins2, outs2 = cached_case(FIX, [1000], 7)
    np.testing.assert_array_equal(outs2[0], outs[0])
    assert ins2[0].dtype == np.float32 and outs2[0].dtype == np.float32

def test_key_changes_with_seed_shape_and_source(tmp_path):
    prob = tmp_path / "vector_add"; shutil.copytree(FIX, prob)
    k = case_key(prob, [10], 1)
    assert k != case_key(prob, [10], 2) and k != case_key(prob, [11], 1)
    (prob / "reference.py").write_text((prob / "reference.py").read_text() + "\n# edited\n")
    assert case_key(prob, [10], 1) != k

def test_corrupt_cache_file_is_recomputed(tmp_path, monkeypatch):
    monkeypatch.setenv("MINIDWARF_CACHE", str(tmp_path))
    path = tmp_path / "cases" / "vector_add" / f"{case_key(FIX, [50], 3)}.npz"
    path.parent.mkdir(parents=True); path.write_bytes(b"truncated")
    ins, outs = cached_case(FIX, [50], 3)
    np.testing.assert_array_equal(outs[0], ins[0] + ins[1])
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_refcache.py -v`
Expected: FAIL (`ModuleNotFoundError: minidwarf.refcache`).

- [ ] **Step 3: Implement** — `minidwarf/refcache.py`:
```python
# SPDX-License-Identifier: Apache-2.0
import hashlib, json, os, tempfile
from pathlib import Path
import numpy as np
from .problem_io import load_module_fn

def cache_dir() -> Path:
    return Path(os.environ.get("MINIDWARF_CACHE") or Path.home() / ".cache" / "minidwarf")

def case_key(problem_root, shape, seed) -> str:
    """Hash of the problem's generator + reference sources, the shape and the seed."""
    root = Path(problem_root); h = hashlib.sha256()
    for f in ("inputs.py", "reference.py"): h.update((root / f).read_bytes())
    h.update(json.dumps([[int(x) for x in shape], int(seed)]).encode())
    return h.hexdigest()[:32]

def cached_case(problem_root, shape, seed):
    """Return (inputs, expected) for one case, generating and caching it on first use."""
    root = Path(problem_root)
    path = cache_dir() / "cases" / root.name / f"{case_key(root, shape, seed)}.npz"
    if path.exists():
        try:
            with np.load(path) as z:
                return ([z[f"in_{i}"] for i in range(int(z["n_in"]))],
                        [z[f"out_{i}"] for i in range(int(z["n_out"]))])
        except Exception:
            pass  # partial or corrupt file: recompute and overwrite
    ins = load_module_fn(root, "inputs.py", "generate")(shape, seed)
    outs = [np.asarray(e, dtype=np.float32) for e in load_module_fn(root, "reference.py", "run")(ins, shape)]
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp.npz"); os.close(fd)
    arrays = {f"in_{i}": a for i, a in enumerate(ins)} | {f"out_{i}": a for i, a in enumerate(outs)}
    np.savez(tmp, n_in=len(ins), n_out=len(outs), **arrays)
    os.replace(tmp, path)  # atomic: concurrent readers never see a half-written file
    return ins, outs
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_refcache.py -v`
Expected: all PASS.

- [ ] **Step 5: Add `tests/test_refcache.py` to the CI pytest list.**

- [ ] **Step 6: Commit**
```bash
git add minidwarf/refcache.py tests/test_refcache.py .github/workflows/ci.yml
git commit -m "Add content-keyed on-disk cache for generated cases and references"
```

---

### Task 6: Check shapes for the 24 problems

**Files:**
- Modify: `problems/*/*/spec.yaml` (24 files)
- Test: `tests/test_check_shapes.py` (new)
- Modify: `.github/workflows/ci.yml` (add `tests/test_check_shapes.py`)

**Interfaces:**
- Consumes: `Problem.check_shapes` (Task 2).

- [ ] **Step 1: Write the failing test** — `tests/test_check_shapes.py`:
```python
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import numpy as np
import pytest
from minidwarf.spec import load_problem
from minidwarf.problem_io import load_module_fn

ROOT = Path(__file__).resolve().parents[1]
PROBLEMS = sorted(p.parent for p in (ROOT / "problems").glob("*/*/spec.yaml"))

@pytest.mark.parametrize("pdir", PROBLEMS, ids=lambda p: p.name)
def test_check_shapes_are_valid(pdir):
    p = load_problem(pdir)
    assert len(p.check_shapes) >= 3
    assert all(len(s) == len(p.eval_shapes[0]) for s in p.check_shapes)
    assert any(s[0] % 32 for s in p.check_shapes)  # at least one non-multiple-of-32 leading dim
    gen = load_module_fn(pdir, "inputs.py", "generate"); ref = load_module_fn(pdir, "reference.py", "run")
    for i, shape in enumerate(p.check_shapes):
        ins = gen(shape, 1 + i); outs = ref(ins, shape)
        assert len(ins) == p.n_inputs and len(outs) == p.n_outputs
        assert all(np.isfinite(np.asarray(o)).all() for o in outs)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_check_shapes.py -v`
Expected: 24 FAIL (`assert 0 >= 3`).

- [ ] **Step 3: Add a `check_shapes:` line after `eval_shapes:` in each spec.yaml**

| Problems | dims | `check_shapes` |
|---|---|---|
| `dense/sgemm`, `dense/gemm_bias`, `dense/scaled_gemm` | [M, N, K] | `[[1, 1, 1], [33, 17, 65], [97, 131, 7]]` |
| `dense/gemv`, `dense/transpose` | [M, N] | `[[1, 1], [33, 65], [131, 7]]` |
| `dense/syrk` | [M, K] | `[[1, 1], [33, 65], [131, 7]]` |
| all 6 `nbody/*` | [N] | `[[3], [33], [1031]]` |
| `sparse/spmv_csr`, `sparse/spmv_transpose`, `sparse/csr_row_scale` | [R, C, NNZ] | `[[1, 1, 1], [37, 29, 20], [129, 67, 1000]]` (37 rows / 20 nnz ⇒ empty rows) |
| `sparse/jacobi_sparse` | [R, C, NNZ], R == C | `[[1, 1, 1], [37, 37, 40], [129, 129, 1000]]` |
| `sparse/sddmm` | [R, C, K, NNZ] | `[[1, 1, 1, 1], [37, 29, 5, 20], [129, 67, 33, 1000]]` |
| `sparse/spmm_csr` | [R, C, NNZ, D] | `[[1, 1, 1, 1], [37, 29, 20, 3], [129, 67, 1000, 33]]` |
| `structured_grids/{gauss_blur_2d, heat_2d_step, laplacian_2d_5pt, sobel_2d, wave_2d_step}` | [ny, nx] | `[[3, 3], [33, 17], [131, 97]]` |
| `structured_grids/jacobi_3d_7pt` | [nx, ny, nz] | `[[3, 3, 3], [33, 17, 9], [67, 35, 19]]` |

Rule if a problem's `generate` rejects a listed shape (e.g. `jacobi_sparse` needing NNZ ≥ R for its diagonal, an N-body problem needing N ≥ some k): replace only that shape with the smallest valid one that keeps a non-multiple-of-32 leading dim, and note the reason in a YAML comment on the same line.

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_check_shapes.py tests/test_spec.py tests/test_splits.py -v`
Expected: all PASS.

- [ ] **Step 5: Confirm no prompt leaks shapes:** `grep -l "check_shapes\|eval_shapes" problems/*/*/prompt.md` → no output.

- [ ] **Step 6: Add `tests/test_check_shapes.py` to the CI pytest list.**

- [ ] **Step 7: Commit**
```bash
git add problems/*/*/spec.yaml tests/test_check_shapes.py .github/workflows/ci.yml
git commit -m "Add edge-case check_shapes to all 24 problems"
```

---

### Task 7: v3 driver timing protocol and multi-set runner (GPU)

**GPU gate:** confirm `nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader` prints nothing before Step 2.

**Files:**
- Modify: `harness/driver.cu` (replace), `minidwarf/runner.py` (replace), `minidwarf/grade.py` (call-site shim only)
- Test: `tests/test_runner.py` (replace)

**Interfaces:**
- Produces: `RunResult(median_ms: float, p25_ms: float, p75_ms: float, times_ms: list[float], outputs: list[list[np.ndarray]])` — `outputs[d][j]` is output `j` after the last timed call on data set `d`.
- Produces: `run_binary(exe, input_sets, dims, output_shapes, reps=20, warmup=3, timeout_s=60) -> RunResult`; raises `ValueError` if `len(input_sets) == 0` or `reps < len(input_sets)`; raises `RunError` on non-zero exit.
- Driver argv: `in out timing.json n_in n_out n_sets reps warmup n_dims dims... in_counts... out_counts...`; input file = set-major concatenation; output file = set-major concatenation; timing JSON keys `median_ms, p25_ms, p75_ms, times_ms`.

- [ ] **Step 1: Write the failing tests** — replace `tests/test_runner.py`:
```python
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import numpy as np
import pytest
from minidwarf.compile import compile_binary
from minidwarf.runner import run_binary

FIX = Path(__file__).parent / "fixtures/smoke/vector_add"
NOOP = '#include <cuda_runtime.h>\nextern "C" void minidwarf_solve(const void* const*, void* const*, const long*, int){}\n'

def _sets(n, k):
    rng = np.random.default_rng(0)
    return [[rng.standard_normal(n).astype(np.float32), rng.standard_normal(n).astype(np.float32)] for _ in range(k)]

def test_run_returns_one_output_per_set(tmp_path):
    exe = compile_binary(FIX / "solutions/expert_v1.cu", tmp_path)
    sets = _sets(1024, 4)
    res = run_binary(exe, sets, dims=[1024], output_shapes=[(1024,)], reps=8, warmup=1)
    assert res.median_ms > 0 and len(res.times_ms) == 8
    assert res.p25_ms <= res.median_ms <= res.p75_ms
    for (a, b), outs in zip(sets, res.outputs):
        np.testing.assert_allclose(outs[0], a + b, rtol=1e-5, atol=1e-6)

def test_noop_kernel_outputs_are_poisoned(tmp_path):
    src = tmp_path / "noop.cu"; src.write_text(NOOP)
    exe = compile_binary(src, tmp_path / "b")
    res = run_binary(exe, _sets(256, 4), dims=[256], output_shapes=[(256,)], reps=4, warmup=0)
    assert all(np.isnan(o[0]).all() for o in res.outputs)

def test_reps_must_cover_all_sets(tmp_path):
    exe = compile_binary(FIX / "solutions/expert_v1.cu", tmp_path)
    with pytest.raises(ValueError):
        run_binary(exe, _sets(16, 4), [16], [(16,)], reps=3)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_runner.py -v`
Expected: FAIL (`TypeError` / wrong outputs structure).

- [ ] **Step 3: Replace `harness/driver.cu`**
```cpp
// SPDX-License-Identifier: Apache-2.0
// MiniDwarf v3 driver. Timing protocol (see docs/superpowers/specs/2026-10-07-hardened-harness-design.md):
// D input data sets rotate through the SAME device buffers; before every call (untimed) the set is
// re-uploaded, outputs are poisoned with NaN and L2 is flushed; the timed region ends with a
// device-wide sync so work on any stream is counted.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error at %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); exit(1);} } while(0)

extern "C" void minidwarf_solve(const void* const*, void* const*, const long*, int);

int main(int argc, char** argv){
  if(argc<11){ fprintf(stderr,"usage: in out timing.json n_in n_out n_sets reps warmup n_dims dims... in_counts... out_counts...\n"); return 2; }
  const char* in=argv[1]; const char* out=argv[2]; const char* tj=argv[3];
  int n_in=atoi(argv[4]), n_out=atoi(argv[5]), n_sets=atoi(argv[6]), reps=atoi(argv[7]), warmup=atoi(argv[8]), n_dims=atoi(argv[9]);
  if(n_sets<1 || reps<n_sets){ fprintf(stderr,"need reps >= n_sets >= 1\n"); return 2; }
  int p=10;
  std::vector<long> dims; for(int i=0;i<n_dims;i++) dims.push_back(atol(argv[p++]));
  std::vector<long> in_cnt; for(int i=0;i<n_in;i++) in_cnt.push_back(atol(argv[p++]));
  std::vector<long> out_cnt; for(int i=0;i<n_out;i++) out_cnt.push_back(atol(argv[p++]));

  std::vector<std::vector<char>> hin((size_t)n_sets*n_in);
  FILE* f=fopen(in,"rb"); if(!f){ fprintf(stderr,"cannot open %s\n",in); return 1; }
  for(int s=0;s<n_sets;s++) for(int i=0;i<n_in;i++){
    std::vector<char>& h=hin[(size_t)s*n_in+i]; h.resize(in_cnt[i]*4);
    if(fread(h.data(),1,h.size(),f)!=h.size()){ fprintf(stderr,"short input file\n"); return 1; } }
  fclose(f);

  std::vector<void*> din(n_in), dout(n_out);
  for(int i=0;i<n_in;i++) CUDA_CHECK(cudaMalloc(&din[i], in_cnt[i]*4));
  for(int i=0;i<n_out;i++) CUDA_CHECK(cudaMalloc(&dout[i], out_cnt[i]*4));
  int l2=0; CUDA_CHECK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, 0));
  size_t flush_bytes=2*(size_t)(l2>0 ? l2 : (64<<20)); void* scratch; CUDA_CHECK(cudaMalloc(&scratch, flush_bytes));

  auto prep=[&](int s, int r){
    for(int i=0;i<n_in;i++) CUDA_CHECK(cudaMemcpy(din[i], hin[(size_t)s*n_in+i].data(), in_cnt[i]*4, cudaMemcpyHostToDevice));
    for(int i=0;i<n_out;i++) CUDA_CHECK(cudaMemset(dout[i], 0xFF, out_cnt[i]*4));  // NaN poison
    CUDA_CHECK(cudaMemset(scratch, r & 0xFF, flush_bytes));                          // L2 flush
    CUDA_CHECK(cudaDeviceSynchronize()); };
  auto call=[&](){ minidwarf_solve((const void* const*)din.data(), (void* const*)dout.data(), dims.data(), n_dims); };

  prep(0, 0); call(); CUDA_CHECK(cudaDeviceSynchronize()); CUDA_CHECK(cudaGetLastError());
  for(int w=0; w<warmup; w++){ prep(w % n_sets, w + 1); call(); CUDA_CHECK(cudaDeviceSynchronize()); CUDA_CHECK(cudaGetLastError()); }

  cudaEvent_t s0,e0; CUDA_CHECK(cudaEventCreate(&s0)); CUDA_CHECK(cudaEventCreate(&e0));
  std::vector<float> times(reps);
  std::vector<std::vector<char>> hout((size_t)n_sets*n_out);
  for(int r=0;r<reps;r++){
    int s=r % n_sets; prep(s, r + warmup + 1);
    CUDA_CHECK(cudaEventRecord(s0)); call(); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(e0)); CUDA_CHECK(cudaEventSynchronize(e0)); CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventElapsedTime(&times[r], s0, e0));
    if(r >= reps - n_sets) for(int j=0;j<n_out;j++){
      std::vector<char>& h=hout[(size_t)s*n_out+j]; h.resize(out_cnt[j]*4);
      CUDA_CHECK(cudaMemcpy(h.data(), dout[j], h.size(), cudaMemcpyDeviceToHost)); } }

  std::vector<float> sorted_t(times); std::sort(sorted_t.begin(), sorted_t.end());
  FILE* fo=fopen(out,"wb"); if(!fo){ fprintf(stderr,"cannot open %s\n",out); return 1; }
  for(size_t k=0;k<hout.size();k++) fwrite(hout[k].data(),1,hout[k].size(),fo);
  fclose(fo);
  FILE* ft=fopen(tj,"w"); if(!ft){ fprintf(stderr,"cannot open %s\n",tj); return 1; }
  fprintf(ft,"{\"median_ms\": %f, \"p25_ms\": %f, \"p75_ms\": %f, \"times_ms\": [",
          sorted_t[reps/2], sorted_t[reps/4], sorted_t[(3*reps)/4]);
  for(int r=0;r<reps;r++) fprintf(ft,"%s%f", r ? ", " : "", times[r]);
  fprintf(ft,"]}\n"); fclose(ft);
  return 0;
}
```

- [ ] **Step 4: Replace `minidwarf/runner.py`**
```python
# SPDX-License-Identifier: Apache-2.0
import json, subprocess, tempfile
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .problem_io import write_arrays, read_arrays

class RunError(Exception): ...

@dataclass
class RunResult:
    median_ms: float
    p25_ms: float
    p75_ms: float
    times_ms: list
    outputs: list  # outputs[data_set][output_index]

def run_binary(exe, input_sets, dims, output_shapes, reps=20, warmup=3, timeout_s=60) -> RunResult:
    """Run a compiled binary with D input data sets rotated through the same device buffers.

    Returns timing stats over `reps` timed calls and, per data set, the outputs of its last timed call."""
    n_sets = len(input_sets)
    if n_sets == 0 or reps < n_sets:
        raise ValueError(f"need reps >= n_sets >= 1, got reps={reps}, n_sets={n_sets}")
    in_counts = [int(np.prod(a.shape)) for a in input_sets[0]]
    out_counts = [int(np.prod(s)) for s in output_shapes]
    with tempfile.TemporaryDirectory() as d:
        din, dout, dt = Path(d)/"in.bin", Path(d)/"out.bin", Path(d)/"t.json"
        write_arrays(din, [a for s in input_sets for a in s])
        argv = [str(exe), str(din), str(dout), str(dt), str(len(in_counts)), str(len(out_counts)),
                str(n_sets), str(reps), str(warmup), str(len(dims))]
        argv += [str(int(x)) for x in dims] + [str(c) for c in in_counts] + [str(c) for c in out_counts]
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s)
        if r.returncode != 0:
            raise RunError(r.stderr or "non-zero exit")
        t = json.loads(dt.read_text())
        flat = read_arrays(dout, [(s, np.float32) for _ in range(n_sets) for s in output_shapes])
    k = len(output_shapes)
    return RunResult(t["median_ms"], t["p25_ms"], t["p75_ms"], t["times_ms"],
                     [flat[i*k:(i+1)*k] for i in range(n_sets)])
```

- [ ] **Step 5: Shim `grade.py` to the new signature** (full rewrite comes in Task 8). In `grade_problem`, replace the two `run_binary(...)` calls and the correctness line with:
```python
            cr: RunResult = run_binary(cand_exe, [ins], shape, output_shapes, reps, warmup=3, timeout_s=timeout_s)
            if not check_correct(cr.outputs[0], expected, p.rtol, p.atol):
                correct = False
            br: RunResult = run_binary(base_exe, [ins], shape, output_shapes, reps, warmup=3, timeout_s=timeout_s)
```

- [ ] **Step 6: Run to verify pass**

Run: `python -m pytest tests/test_runner.py tests/test_grade.py tests/test_timeout.py tests/test_compile.py -v`
Expected: all PASS.

- [ ] **Step 7: Commit**
```bash
git add harness/driver.cu minidwarf/runner.py minidwarf/grade.py tests/test_runner.py
git commit -m "Driver v3: rotating data sets, NaN-poisoned outputs, L2 flush, device-wide sync in timer"
```

---

### Task 8: v3 grading flow with per-check detail (GPU)

**GPU gate:** as Task 7.

**Files:**
- Modify: `minidwarf/correctness.py`, `minidwarf/grade.py` (replace)
- Test: `tests/test_correctness.py`, `tests/test_grade.py`, `tests/test_grade_lint.py` (new, GPU-less)
- Modify: `.github/workflows/ci.yml` (add `tests/test_correctness.py`, `tests/test_grade_lint.py`)

**Interfaces:**
- Consumes: `lint_source`, `lib_flags`, `link_flags` (T1), `Problem.check_shapes/allowed_libs` (T2), `geomean_speedup` (T3), `cached_case` (T5), `run_binary`/`RunResult` (T7).
- Produces: `compare(actual, expected, rtol, atol) -> dict` with keys `passed: bool, max_abs_err, max_rel_err, tol_ratio, nan_count` (the last four `None` on shape mismatch; NaN differences count as `inf`).
- Produces: `N_SETS = 4`, `CHECK_SEED_OFFSET = 500_000`; `ProblemResult(name, dwarf, status, correct, speedup, checks=[], timings=[], lint=[])`; each check = `{"kind": "check"|"eval", "shape", "data_set", **compare(...)}`; each timing = `{"shape", "cand_median_ms", "cand_iqr_ms", "base_median_ms", "base_iqr_ms"}`; status adds `"forbidden_api"`.
- Produces: `grade_problem(problem_root, candidate_cu, work_dir, seed=12345, reps=20, warmup=3, timeout_s=60, trusted=False) -> ProblemResult`. `trusted=True` skips lint and links the candidate with the baseline's flags (used to grade `baseline.cu` itself).
- Seeds: eval shape `i`, set `d` → `seed + 1000*i + d`; check shape `i`, set `d` → `seed + CHECK_SEED_OFFSET + 1000*i + d`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_correctness.py`:
```python
import pytest
from minidwarf.correctness import compare

def test_compare_reports_errors():
    r = compare([np.array([1.0, 2.0], np.float32)], [np.array([1.0, 2.5], np.float32)], 1e-3, 1e-4)
    assert r["passed"] is False and r["max_abs_err"] == pytest.approx(0.5)
    assert r["nan_count"] == 0 and r["tol_ratio"] > 1

def test_compare_nan_and_shape_mismatch():
    r = compare([np.array([np.nan], np.float32)], [np.array([1.0], np.float32)], 1e-3, 1e-4)
    assert r["passed"] is False and r["nan_count"] == 1 and r["max_abs_err"] == float("inf")
    assert compare([np.zeros(2)], [np.zeros(3)], 1e-3, 1e-4)["passed"] is False

def test_compare_exact_has_zero_ratio():
    r = compare([np.ones(3, np.float32)], [np.ones(3, np.float32)], 1e-3, 1e-4)
    assert r["passed"] is True and r["tol_ratio"] == 0.0
```

`tests/test_grade_lint.py`:
```python
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from minidwarf.grade import grade_problem

FIX = Path(__file__).parent / "fixtures/smoke/vector_add"

def test_lint_runs_before_compile_on_non_utf8_source(tmp_path):
    bad = tmp_path / "bad.cu"; bad.write_bytes(b"\xff\xfe#include <cublas_v2.h>\n")
    r = grade_problem(FIX, bad, tmp_path)
    assert r.status == "forbidden_api" and r.lint == ["cublas"] and r.correct is False and r.speedup is None
```

Replace `tests/test_grade.py`:
```python
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from minidwarf.grade import grade_problem, N_SETS

FIX = Path(__file__).parent / "fixtures/smoke/vector_add"

def test_expert_grades_ok_with_detail(tmp_path):
    r = grade_problem(FIX, FIX / "solutions/expert_v1.cu", tmp_path)
    assert r.status == "ok" and r.correct is True and r.speedup > 0
    evals = [c for c in r.checks if c["kind"] == "eval"]
    assert len(evals) == 2 * 2 * N_SETS  # 2 eval shapes x 2 candidate runs (ABBA) x D sets
    assert len(r.timings) == 2 and all(t["cand_median_ms"] > 0 for t in r.timings)

def test_cheater_is_caught(tmp_path):
    r = grade_problem(FIX, FIX / "solutions/cheater.cu", tmp_path)
    assert r.correct is False and r.status == "wrong_output"

def test_broken_is_compile_error(tmp_path):
    bad = tmp_path / "bad.cu"; bad.write_text("nope")
    r = grade_problem(FIX, bad, tmp_path)
    assert r.status == "compile_error"

def test_baseline_trusted_against_itself(tmp_path):
    r = grade_problem(FIX, FIX / "baseline.cu", tmp_path, trusted=True)
    assert r.status == "ok" and 0.5 < r.speedup < 2.0
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_correctness.py tests/test_grade_lint.py tests/test_grade.py -v`
Expected: FAIL (`ImportError: compare`, `N_SETS`; `forbidden_api` not produced).

- [ ] **Step 3: Implement `compare`** — append to `minidwarf/correctness.py`:
```python
def compare(actual, expected, rtol, atol) -> dict:
    """Per-check detail: pass/fail, max abs/rel error, tol_ratio = max |a-e| / (atol + rtol|e|), NaN count."""
    if len(actual) != len(expected) or any(np.shape(a) != np.shape(e) for a, e in zip(actual, expected)):
        return {"passed": False, "max_abs_err": None, "max_rel_err": None, "tol_ratio": None, "nan_count": None}
    abs_err = rel_err = ratio = 0.0; nans = 0
    for a, e in zip(actual, expected):
        a = np.asarray(a, np.float64); e = np.asarray(e, np.float64)
        nans += int(np.isnan(a).sum())
        d = np.abs(a - e); d = np.where(np.isnan(d), np.inf, d)
        if d.size == 0: continue
        tol = atol + rtol * np.abs(e)
        abs_err = max(abs_err, float(d.max()))
        rel_err = max(rel_err, float((d / np.maximum(np.abs(e), max(atol, 1e-30))).max()))
        ratio = max(ratio, float((d / np.where(tol > 0, tol, 1e-30)).max()))
    return {"passed": check_correct(actual, expected, rtol, atol), "max_abs_err": abs_err,
            "max_rel_err": rel_err, "tol_ratio": ratio, "nan_count": nans}
```

- [ ] **Step 4: Replace `minidwarf/grade.py`**
```python
# SPDX-License-Identifier: Apache-2.0
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from .spec import load_problem
from .compile import compile_binary, CompileError
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
    "forbidden_api"; a misbehaving candidate never raises. `trusted=True` skips the lint and links
    the candidate like the baseline (used to grade baseline.cu against itself)."""
    p = load_problem(problem_root)
    work_dir = Path(work_dir)
    fail = lambda status, **kw: ProblemResult(p.name, p.dwarf, status, False, None, **kw)
    if not trusted:
        hits = lint_source(Path(candidate_cu).read_text(errors="replace"), p.allowed_libs)
        if hits: return fail("forbidden_api", lint=hits)
    try:
        base_flags = link_flags(p.baseline)
        cand_exe = compile_binary(Path(candidate_cu), work_dir / "cand",
                                  extra_flags=base_flags if trusted else lib_flags(p.allowed_libs))
        base_exe = compile_binary(p.root / "baseline.cu", work_dir / "base", extra_flags=base_flags)
    except (CompileError, subprocess.TimeoutExpired):
        return fail("compile_error")

    checks, timings = [], []
    try:
        for i, shape in enumerate(p.check_shapes):
            ins, exp = _cases(p.root, shape, seed + CHECK_SEED_OFFSET + 1000 * i)
            shapes = [e.shape for e in exp[0]]
            run = run_binary(cand_exe, ins, shape, shapes, N_SETS, 0, timeout_s)
            checks += _checks(run, exp, shape, p, "check")
        for i, shape in enumerate(p.eval_shapes):
            ins, exp = _cases(p.root, shape, seed + 1000 * i)
            shapes = [e.shape for e in exp[0]]
            c1 = run_binary(cand_exe, ins, shape, shapes, reps, warmup, timeout_s)  # ABBA order
            b1 = run_binary(base_exe, ins, shape, shapes, reps, warmup, timeout_s)
            b2 = run_binary(base_exe, ins, shape, shapes, reps, warmup, timeout_s)
            c2 = run_binary(cand_exe, ins, shape, shapes, reps, warmup, timeout_s)
            checks += _checks(c1, exp, shape, p, "eval") + _checks(c2, exp, shape, p, "eval")
            ct, bt = c1.times_ms + c2.times_ms, b1.times_ms + b2.times_ms
            timings.append({"shape": list(shape), "cand_median_ms": float(np.median(ct)), "cand_iqr_ms": _iqr(ct),
                            "base_median_ms": float(np.median(bt)), "base_iqr_ms": _iqr(bt)})
    except RunError:
        return fail("runtime_error", checks=checks, timings=timings)
    except subprocess.TimeoutExpired:
        return fail("timeout", checks=checks, timings=timings)

    correct = all(c["passed"] for c in checks)
    speedup = geomean_speedup([t["base_median_ms"] for t in timings], [t["cand_median_ms"] for t in timings])
    return ProblemResult(p.name, p.dwarf, "ok" if correct else "wrong_output", correct, speedup, checks, timings)
```

- [ ] **Step 5: Run to verify pass**

Run: `python -m pytest tests/test_correctness.py tests/test_grade_lint.py tests/test_grade.py tests/test_timeout.py tests/test_score.py tests/test_evaluate.py -v`
Expected: all PASS.

- [ ] **Step 6: Regression over real problems** — `python -m pytest tests/test_problem_*.py -v` (experts still `ok` at the current v2 eval shapes plus the new check shapes).
Expected: all PASS. A failure here means an expert mishandles an edge shape: fix the expert kernel (it is the reference implementation), not the check shape.

- [ ] **Step 7: Add `tests/test_correctness.py` and `tests/test_grade_lint.py` to the CI pytest list.**

- [ ] **Step 8: Commit**
```bash
git add minidwarf/correctness.py minidwarf/grade.py tests/test_correctness.py tests/test_grade.py tests/test_grade_lint.py .github/workflows/ci.yml
git commit -m "Grade v3: lint, library-free candidate link, check shapes, ABBA, geomean, per-check detail"
```

---

### Task 9: Red-team suite (GPU)

**GPU gate:** as Task 7.

**Files:**
- Create: `tests/fixtures/redteam/{memo_static,memo_copy,side_stream_nosync,calls_cublas,fopen_grader,noop,assumes_mult32}.cu`
- Modify: `tests/fixtures/smoke/vector_add/spec.yaml` (add `check_shapes: [[1], [33], [1031]]`)
- Test: `tests/test_redteam.py` (new)

**Interfaces:**
- Consumes: `grade_problem`, `ProblemResult.checks/timings` (T8).

- [ ] **Step 1: Write the fixtures.** Each file starts with the SPDX line and this shared kernel:
```cpp
// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>
__global__ void vadd(const float* a, const float* b, float* c, long n){
  long i=blockIdx.x*(long)blockDim.x+threadIdx.x; if(i<n) c[i]=a[i]+b[i]; }
```
then the per-fixture body:

`memo_static.cu` (pointer-keyed caching: compute once, return early afterwards):
```cpp
// Red-team: computes on the first call only, then returns without touching outputs.
extern "C" void minidwarf_solve(const void* const* in, void* const* out, const long* dims, int nd){
  static bool done = false; if (done) return;
  long n = dims[0];
  vadd<<<(int)((n+255)/256),256>>>((const float*)in[0], (const float*)in[1], (float*)out[0], n);
  done = true; }
```
`memo_copy.cu` (caches the result in its own buffer and replays it):
```cpp
// Red-team: computes into a private buffer on the first call, then only copies it to the output.
extern "C" void minidwarf_solve(const void* const* in, void* const* out, const long* dims, int nd){
  static float* cache = nullptr; long n = dims[0];
  if (!cache){ cudaMalloc(&cache, n*sizeof(float));
    vadd<<<(int)((n+255)/256),256>>>((const float*)in[0], (const float*)in[1], cache, n); }
  cudaMemcpy(out[0], cache, n*sizeof(float), cudaMemcpyDeviceToDevice); }
```
`side_stream_nosync.cu` (work escapes a default-stream timer):
```cpp
// Red-team: launches on a non-blocking stream and never synchronizes.
extern "C" void minidwarf_solve(const void* const* in, void* const* out, const long* dims, int nd){
  static cudaStream_t s = nullptr; if (!s) cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
  long n = dims[0];
  vadd<<<(int)((n+255)/256),256,0,s>>>((const float*)in[0], (const float*)in[1], (float*)out[0], n); }
```
`calls_cublas.cu` (add `#include <cublas_v2.h>` after `cuda_runtime.h`):
```cpp
// Red-team: delegates to cuBLAS (forbidden for candidates).
extern "C" void minidwarf_solve(const void* const* in, void* const* out, const long* dims, int nd){
  static cublasHandle_t h; static bool init = false; if (!init){ cublasCreate(&h); init = true; }
  long n = dims[0]; float one = 1.0f;
  cudaMemcpy(out[0], in[1], n*sizeof(float), cudaMemcpyDeviceToDevice);
  cublasSaxpy(h, (int)n, &one, (const float*)in[0], 1, (float*)out[0], 1); }
```
`fopen_grader.cu` (add `#include <cstdio>`):
```cpp
// Red-team: pokes at the grader's files before computing honestly.
extern "C" void minidwarf_solve(const void* const* in, void* const* out, const long* dims, int nd){
  FILE* f = fopen("/proc/self/cmdline", "rb"); if (f) fclose(f);
  long n = dims[0];
  vadd<<<(int)((n+255)/256),256>>>((const float*)in[0], (const float*)in[1], (float*)out[0], n); }
```
`noop.cu`:
```cpp
// Red-team: does nothing (relies on stale output buffers).
extern "C" void minidwarf_solve(const void* const* in, void* const* out, const long* dims, int nd){}
```
`assumes_mult32.cu`:
```cpp
// Red-team: silently drops the tail when n is not a multiple of 32 (an indexing bug, not a hack).
extern "C" void minidwarf_solve(const void* const* in, void* const* out, const long* dims, int nd){
  long n = (dims[0] / 32) * 32; if (n == 0) return;
  vadd<<<(int)((n+255)/256),256>>>((const float*)in[0], (const float*)in[1], (float*)out[0], n); }
```

- [ ] **Step 2: Add `check_shapes: [[1], [33], [1031]]`** to `tests/fixtures/smoke/vector_add/spec.yaml` after `eval_shapes`.

- [ ] **Step 3: Write the test** — `tests/test_redteam.py`:
```python
# SPDX-License-Identifier: Apache-2.0
import shutil
from pathlib import Path
import pytest
from minidwarf.grade import grade_problem

FIX = Path(__file__).parent / "fixtures/smoke/vector_add"
RT = Path(__file__).parent / "fixtures/redteam"
pytestmark = pytest.mark.skipif(shutil.which("nvcc") is None, reason="needs CUDA toolchain")

@pytest.mark.parametrize("name,status", [
    ("memo_static", "wrong_output"), ("memo_copy", "wrong_output"), ("noop", "wrong_output"),
    ("assumes_mult32", "wrong_output"), ("calls_cublas", "forbidden_api"), ("fopen_grader", "forbidden_api"),
])
def test_redteam_fixture_is_caught(tmp_path, name, status):
    r = grade_problem(FIX, RT / f"{name}.cu", tmp_path)
    assert r.status == status and r.correct is False

def test_side_stream_work_is_timed(tmp_path):
    honest = grade_problem(FIX, FIX / "solutions/expert_v1.cu", tmp_path / "h")
    side = grade_problem(FIX, RT / "side_stream_nosync.cu", tmp_path / "s")
    assert side.status == "ok"
    for h, s in zip(honest.timings, side.timings):
        assert s["cand_median_ms"] >= 0.7 * h["cand_median_ms"]

def test_check_shapes_catch_what_eval_shapes_miss(tmp_path):
    r = grade_problem(FIX, RT / "assumes_mult32.cu", tmp_path)
    assert all(c["passed"] for c in r.checks if c["kind"] == "eval")  # both eval sizes are multiples of 32
    assert not all(c["passed"] for c in r.checks if c["kind"] == "check")
```

- [ ] **Step 4: Run**

Run: `python -m pytest tests/test_redteam.py tests/test_grade.py tests/test_timeout.py -v`
Expected: all PASS. (These fixtures are written against the v3 driver; there is no separate "fail first" step because the defenses already exist from Tasks 7–8 — if any test fails, the defense is broken: debug the harness, do not weaken the test.)

- [ ] **Step 5: Commit**
```bash
git add tests/fixtures/redteam tests/fixtures/smoke/vector_add/spec.yaml tests/test_redteam.py
git commit -m "Add red-team kernel suite proving the v3 harness defenses"
```

---

### Task 10: Scoring metadata, preflight in runs, `minidwarf score` CLI

**Files:**
- Modify: `minidwarf/evaluate.py`, `minidwarf/cli.py`
- Test: `tests/test_evaluate.py`, `tests/test_cli_eval.py`

**Interfaces:**
- Consumes: `ensure_gpu_idle`, `env_record`, `GpuBusyError` (T4); `HARNESS_VERSION` (T3); `ProblemResult` (T8).
- Produces: `score_run(run_dir, problems_root=Path("problems"), allow_busy_gpu=False) -> Path`; `scores.json` = `{"model", "harness_version": 3, "allow_busy_gpu", "env": {"start", "end"}, "per_problem": [asdict(ProblemResult), ...]}`.
- Produces CLI: `minidwarf eval ... [--allow-busy-gpu]`; `minidwarf score --run-dir DIR [--problems-root problems] [--allow-busy-gpu]`; both exit 3 with `error: GPU busy...` on stderr when the GPU is busy.

- [ ] **Step 1: Write the failing tests** — in `tests/test_evaluate.py` add after the `best_result` tests:
```python
def test_forbidden_api_ranks_below_wrong_output():
    r = best_result([mk("forbidden_api", False, None), mk("wrong_output", False, None)])
    assert r.status == "wrong_output"

def test_score_run_refuses_busy_gpu(tmp_path, monkeypatch):
    import json as _json
    import minidwarf.preflight as pf
    from minidwarf.preflight import GpuBusyError
    monkeypatch.setattr(pf, "_smi", lambda args: "42, python\n")
    run = tmp_path / "r"; run.mkdir()
    (run / "results.jsonl").write_text(_json.dumps(
        {"problem": "vector_add", "dwarf": "smoke", "model": "m", "kernel_path": "kernels/x.cu"}) + "\n")
    with pytest.raises(GpuBusyError):
        score_run(run, problems_root=Path(__file__).parent / "fixtures")
```
(`score_run`, `Path`, `pytest` are already imported further down that file; move those imports to the top of the file.)

In `test_score_run_e2e_smoke`, call `score_run(run_dir, problems_root=SMOKE.parent.parent, allow_busy_gpu=True)` and add:
```python
    assert data["harness_version"] == 3 and "env" in data
    assert data["per_problem"][0]["checks"] and data["per_problem"][0]["timings"]
```
In `tests/test_cli_eval.py`, add `"--allow-busy-gpu"` to the `eval` argv, and after the leaderboard assertions:
```python
    run_dir = next(runs.iterdir())
    sc = subprocess.run([sys.executable, "-m", "minidwarf.cli", "score", "--run-dir", str(run_dir),
                         "--allow-busy-gpu"], cwd=ROOT, capture_output=True, text=True)
    assert sc.returncode == 0, sc.stderr
    assert json.loads((run_dir / "scores.json").read_text())["harness_version"] == 3
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_evaluate.py -v`
Expected: FAIL (`forbidden_api` ranked equal to `compile_error` default 0 → picks first; `GpuBusyError` not raised).

- [ ] **Step 3: Implement `minidwarf/evaluate.py`** (replace):
```python
# SPDX-License-Identifier: Apache-2.0
import json, tempfile
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from .grade import grade_problem, ProblemResult
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
    rows = [json.loads(l) for l in (run_dir / "results.jsonl").read_text().splitlines()]
    by_problem = defaultdict(list)
    for row in rows:
        by_problem[(row["problem"], row["dwarf"])].append(row)
    model = rows[0]["model"] if rows else "unknown"
    env_start, per_problem = env_record(), []
    for (name, dwarf), group in sorted(by_problem.items()):
        ensure_gpu_idle(allow_busy_gpu)
        pdir = Path(problems_root) / dwarf / name
        graded = []
        for row in group:
            with tempfile.TemporaryDirectory() as wd:
                graded.append(grade_problem(pdir, run_dir / row["kernel_path"], Path(wd)))
        per_problem.append(asdict(best_result(graded)))
    out = run_dir / "scores.json"
    out.write_text(json.dumps({"model": model, "harness_version": HARNESS_VERSION,
                               "allow_busy_gpu": allow_busy_gpu,
                               "env": {"start": env_start, "end": env_record()},
                               "per_problem": per_problem}, indent=2))
    return out
```

- [ ] **Step 4: Implement CLI changes** in `minidwarf/cli.py`:
  - ensure `import sys` is at the top;
  - after the `ev.add_argument(...)` lines: `ev.add_argument("--allow-busy-gpu", action="store_true")`;
  - add the parser:
```python
    sc = sub.add_parser("score")
    sc.add_argument("--run-dir", required=True); sc.add_argument("--problems-root", default="problems")
    sc.add_argument("--allow-busy-gpu", action="store_true")
```
  - in the `eval` branch replace `scores = score_run(run_dir)` with:
```python
        from .preflight import GpuBusyError
        try:
            scores = score_run(run_dir, allow_busy_gpu=a.allow_busy_gpu)
        except GpuBusyError as e:
            print(f"error: {e}", file=sys.stderr); return 3
```
  - add before the `leaderboard` branch:
```python
    if a.cmd == "score":
        from .evaluate import score_run
        from .preflight import GpuBusyError
        try:
            out = score_run(Path(a.run_dir), Path(a.problems_root), allow_busy_gpu=a.allow_busy_gpu)
        except GpuBusyError as e:
            print(f"error: {e}", file=sys.stderr); return 3
        print(str(out)); return 0
```

- [ ] **Step 5: Run to verify pass** (GPU gate for the nvcc-gated ones)

Run: `python -m pytest tests/test_evaluate.py tests/test_cli_eval.py tests/test_cli.py tests/test_leaderboard.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**
```bash
git add minidwarf/evaluate.py minidwarf/cli.py tests/test_evaluate.py tests/test_cli_eval.py
git commit -m "Score runs with GPU preflight, env record, harness_version; add minidwarf score"
```

---

### Task 11: Measurement scripts

**Files:**
- Create: `scripts/size_shapes.py`, `scripts/noise_floor.py`, `scripts/calibrate_tol.py`
- Test: `tests/test_scripts.py` (new, GPU-less)
- Modify: `.github/workflows/ci.yml` (add `tests/test_scripts.py`)

**Interfaces:**
- Consumes: `grade_problem`, `N_SETS` (T8); `cached_case` (T5); `run_binary` (T7); `ensure_gpu_idle`, `env_record` (T4); `load_problem`, `link_flags`, `compile_binary`.
- Produces (pure, unit-tested): `size_shapes.row_ok(median_ms, device_mb) -> bool`; `noise_floor.eps_from_speedups(speedups, q=0.95) -> float`; `calibrate_tol.verdict(tol_ratio) -> str` in `{"exact", "ok", "too_loose", "fail"}`; `calibrate_tol.calibrate(pdir) -> dict` (keys `problem, rtol, atol, status, tol_ratio, verdict`).
- Output files: `harness/shape_report.json`, `harness/noise_floor.json` (`{"eps", "quantile", "env", "date", "per_problem"}`), `harness/tolerance_report.json`.

- [ ] **Step 1: Write the failing tests** — `tests/test_scripts.py`:
```python
# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def test_row_ok():
    m = _load("size_shapes")
    assert m.row_ok(1.5, 800) and not m.row_ok(0.4, 800) and not m.row_ok(3.0, 2500)

def test_eps_from_speedups():
    m = _load("noise_floor")
    assert m.eps_from_speedups([1.0, 1.0, 1.0]) == pytest.approx(0.0)
    assert m.eps_from_speedups([1.0, 1.1, 1 / 1.1]) == pytest.approx(0.1, rel=1e-6)

def test_verdict():
    m = _load("calibrate_tol")
    assert m.verdict(0.0) == "exact" and m.verdict(0.5) == "ok"
    assert m.verdict(2.0) == "fail" and m.verdict(0.001) == "too_loose"
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_scripts.py -v`
Expected: FAIL (`FileNotFoundError` for the scripts).

- [ ] **Step 3: Implement `scripts/size_shapes.py`**
```python
# SPDX-License-Identifier: Apache-2.0
"""Measure baseline time, device memory and case-generation cost for eval shapes.

  python scripts/size_shapes.py problems/dense/sgemm 4096,4096,4096 3000,5000,3500
  python scripts/size_shapes.py --check-all          # every problem's spec.yaml eval_shapes
"""
import argparse, json, resource, sys, tempfile, time
from pathlib import Path
from minidwarf.spec import load_problem
from minidwarf.baselines import link_flags
from minidwarf.compile import compile_binary
from minidwarf.grade import N_SETS
from minidwarf.preflight import ensure_gpu_idle
from minidwarf.refcache import cached_case
from minidwarf.runner import run_binary

MIN_MS, MAX_DEVICE_MB, WARN_CASE_S = 1.0, 2000.0, 120.0
SEED = 12345  # grade_problem's default: warms exactly the cache entries grading will use

def row_ok(median_ms, device_mb):
    return median_ms >= MIN_MS and device_mb <= MAX_DEVICE_MB

def measure(pdir, shapes):
    p = load_problem(pdir); rows = []
    with tempfile.TemporaryDirectory() as d:
        exe = compile_binary(p.root / "baseline.cu", Path(d), extra_flags=link_flags(p.baseline))
        for i, shape in enumerate(shapes):
            t0 = time.perf_counter()
            cases = [cached_case(p.root, shape, SEED + 1000 * i + k) for k in range(N_SETS)]
            case_s = time.perf_counter() - t0
            ins, exp = [c[0] for c in cases], [c[1] for c in cases]
            r = run_binary(exe, ins, shape, [e.shape for e in exp[0]], reps=20, warmup=3)
            dev_mb = (sum(a.nbytes for a in ins[0]) + sum(e.nbytes for e in exp[0])) / 1e6
            rows.append({"problem": p.name, "shape": list(shape), "base_median_ms": r.median_ms,
                         "device_mb": dev_mb, "case_s": case_s,
                         "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                         "ok": row_ok(r.median_ms, dev_mb)})
            flag = "" if rows[-1]["ok"] else "  <-- FAIL"
            warn = "  (WARN: slow case generation)" if case_s > WARN_CASE_S else ""
            print(f"{p.name:28s} {str(list(shape)):32s} {r.median_ms:9.3f} ms {dev_mb:8.1f} MB {case_s:7.1f} s{flag}{warn}", flush=True)
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("problem", nargs="?"); ap.add_argument("shapes", nargs="*")
    ap.add_argument("--check-all", action="store_true"); ap.add_argument("--problems-root", default="problems")
    ap.add_argument("--out", default="harness/shape_report.json"); ap.add_argument("--allow-busy-gpu", action="store_true")
    a = ap.parse_args()
    ensure_gpu_idle(a.allow_busy_gpu)
    if a.check_all:
        rows = []
        for spec in sorted(Path(a.problems_root).glob("*/*/spec.yaml")):
            rows += measure(spec.parent, load_problem(spec.parent).eval_shapes)
        Path(a.out).write_text(json.dumps(rows, indent=2) + "\n")
        return 0 if all(r["ok"] for r in rows) else 1
    if not a.problem or not a.shapes:
        ap.error("give a problem dir and at least one shape, or --check-all")
    measure(Path(a.problem), [[int(x) for x in s.split(",")] for s in a.shapes])
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Implement `scripts/noise_floor.py`**
```python
# SPDX-License-Identifier: Apache-2.0
"""A/A timing test: grade every problem's baseline against itself to measure timing noise eps."""
import argparse, datetime, json, math, sys, tempfile
from pathlib import Path
import numpy as np
from minidwarf.grade import grade_problem
from minidwarf.preflight import ensure_gpu_idle, env_record

def eps_from_speedups(speedups, q=0.95):
    """eps such that a (1 + eps) speedup exceeds the q-quantile of |log speedup| in an A/A test."""
    return math.exp(float(np.quantile([abs(math.log(s)) for s in speedups], q))) - 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems-root", default="problems"); ap.add_argument("--out", default="harness/noise_floor.json")
    ap.add_argument("--allow-busy-gpu", action="store_true")
    a = ap.parse_args()
    ensure_gpu_idle(a.allow_busy_gpu)
    per = {}
    for spec in sorted(Path(a.problems_root).glob("*/*/spec.yaml")):
        pdir = spec.parent
        with tempfile.TemporaryDirectory() as d:
            r = grade_problem(pdir, pdir / "baseline.cu", Path(d), trusted=True)
        if r.status != "ok":
            print(f"FAIL {pdir.name}: baseline vs itself is {r.status}", file=sys.stderr); return 1
        per[pdir.name] = r.speedup; print(f"{pdir.name:28s} {r.speedup:.4f}", flush=True)
    eps = eps_from_speedups(list(per.values()))
    Path(a.out).write_text(json.dumps({"eps": eps, "quantile": 0.95, "env": env_record(),
                                       "date": datetime.date.today().isoformat(), "per_problem": per}, indent=2) + "\n")
    print(f"eps = {eps:.4f}"); return 0

if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Implement `scripts/calibrate_tol.py`**
```python
# SPDX-License-Identifier: Apache-2.0
"""Grade each expert kernel and compare its error with the problem's tolerance.

tol_ratio = max |a - e| / (atol + rtol*|e|) over all checks; the tolerance is calibrated when
0.01 <= tol_ratio <= 1 (within [1x, 100x] of a real float32 kernel's error)."""
import argparse, json, sys, tempfile
from pathlib import Path
from minidwarf.grade import grade_problem
from minidwarf.preflight import ensure_gpu_idle
from minidwarf.spec import load_problem

def verdict(tol_ratio):
    if tol_ratio == 0: return "exact"
    if tol_ratio > 1: return "fail"
    if tol_ratio < 0.01: return "too_loose"
    return "ok"

def calibrate(pdir) -> dict:
    p = load_problem(pdir)
    with tempfile.TemporaryDirectory() as d:
        r = grade_problem(pdir, pdir / "solutions/expert_v1.cu", Path(d), reps=4, warmup=0)
    ratios = [c["tol_ratio"] for c in r.checks]
    ratio = max(ratios) if ratios and None not in ratios else float("inf")
    return {"problem": p.name, "rtol": p.rtol, "atol": p.atol, "status": r.status,
            "tol_ratio": ratio, "verdict": verdict(ratio)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems-root", default="problems"); ap.add_argument("--out", default="harness/tolerance_report.json")
    ap.add_argument("--allow-busy-gpu", action="store_true")
    a = ap.parse_args()
    ensure_gpu_idle(a.allow_busy_gpu)
    rows = []
    for spec in sorted(Path(a.problems_root).glob("*/*/spec.yaml")):
        rows.append(calibrate(spec.parent))
        r = rows[-1]; print(f"{r['problem']:28s} rtol={r['rtol']:.1e} atol={r['atol']:.1e} ratio={r['tol_ratio']:.3e} {r['verdict']}", flush=True)
    Path(a.out).write_text(json.dumps(rows, indent=2) + "\n")
    return 0 if all(r["verdict"] in ("ok", "exact") for r in rows) else 1

if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: Run to verify pass**

Run: `python -m pytest tests/test_scripts.py -v`
Expected: all PASS.

- [ ] **Step 7: Add `tests/test_scripts.py` to the CI pytest list.**

- [ ] **Step 8: Commit**
```bash
git add scripts/size_shapes.py scripts/noise_floor.py scripts/calibrate_tol.py tests/test_scripts.py .github/workflows/ci.yml
git commit -m "Add shape-sizing, noise-floor and tolerance-calibration scripts"
```

---

### Task 12: Resize eval shapes, calibrate tolerances, measure the noise floor (GPU, long)

**GPU gate:** the GPU must be idle for the whole task (timings are the deliverable). Run every script without `--allow-busy-gpu`. Expect several hours wall time, dominated by first-time case generation (cached afterwards in `~/.cache/minidwarf`, potentially tens of GB — check `df -h ~` first).

**Files:**
- Modify: `problems/*/*/spec.yaml` (`eval_shapes`, possibly `rtol`/`atol`), `problems/*/*/reference.py` (chunked rewrites where needed)
- Create: `harness/shape_report.json`, `harness/tolerance_report.json`, `harness/noise_floor.json`
- Test: `tests/test_problem_invariants.py` (new), `tests/test_tolerances.py` (new)

**Interfaces:**
- Consumes: scripts from Task 11, `cached_case` (T5), `grade_problem` (T8).

- [ ] **Step 1: Write the invariant tests** — `tests/test_problem_invariants.py`:
```python
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import numpy as np
import pytest
from minidwarf.spec import load_problem
from minidwarf.refcache import cached_case

ROOT = Path(__file__).resolve().parents[1]
PROBLEMS = sorted(p.parent for p in (ROOT / "problems").glob("*/*/spec.yaml"))

@pytest.mark.parametrize("pdir", PROBLEMS, ids=lambda p: p.name)
def test_reference_outputs_are_not_near_constant(pdir):
    p = load_problem(pdir)
    _, outs = cached_case(pdir, p.eval_shapes[0], 12345)  # same case grading uses (eval shape 0, set 0)
    for o in outs:
        o = np.asarray(o, np.float64)
        assert o.std() / (abs(o.mean()) + 1e-12) > 1e-3, f"{p.name}: near-constant reference output"
```
`tests/test_tolerances.py`:
```python
# SPDX-License-Identifier: Apache-2.0
import importlib.util, shutil
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
PROBLEMS = sorted(p.parent for p in (ROOT / "problems").glob("*/*/spec.yaml"))
_spec = importlib.util.spec_from_file_location("calibrate_tol", ROOT / "scripts/calibrate_tol.py")
cal = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(cal)

@pytest.mark.skipif(shutil.which("nvcc") is None, reason="needs CUDA toolchain")
@pytest.mark.parametrize("pdir", PROBLEMS, ids=lambda p: p.name)
def test_tolerance_is_calibrated(pdir):
    row = cal.calibrate(pdir)
    assert row["status"] == "ok" and row["verdict"] in ("ok", "exact"), row
```

- [ ] **Step 2: Size each problem.** Starting points (measure, then adjust until the row is `ok`; the second shape must stay non-square / non-power-of-two to exercise tails):

| Problems | starting `eval_shapes` |
|---|---|
| 2D stencils (`gauss_blur_2d`, `heat_2d_step`, `laplacian_2d_5pt`, `sobel_2d`, `wave_2d_step`) | `[[8192, 8192], [7000, 9000]]` |
| `jacobi_3d_7pt` | `[[512, 512, 512], [480, 500, 520]]` |
| all 6 `nbody/*` | `[[32768], [30000]]` |
| `sgemm`, `gemm_bias`, `scaled_gemm` | `[[4096, 4096, 4096], [3000, 5000, 3500]]` |
| `gemv` | `[[16384, 16384], [12000, 20000]]` |
| `transpose` | `[[8192, 8192], [7000, 9000]]` |
| `syrk` | `[[4096, 4096], [3000, 5000]]` |
| `spmv_csr`, `spmv_transpose`, `csr_row_scale` | `[[1048576, 1048576, 16777216], [800000, 1200000, 12000000]]` |
| `jacobi_sparse` | `[[1048576, 1048576, 16777216], [900000, 900000, 12000000]]` |
| `sddmm` | `[[8192, 8192, 64, 8388608], [6000, 9000, 48, 6000000]]` |
| `spmm_csr` | `[[262144, 262144, 4194304, 32], [200000, 300000, 3000000, 24]]` |

Command per problem, e.g.: `python scripts/size_shapes.py problems/nbody/nbody_gravity_force 32768 30000`

- [ ] **Step 3: Chunk any reference that is too slow or too big.** If a case takes > 120 s or the process exceeds ~8 GB RSS (watch `peak_rss_mb`), rewrite `reference.py` in row blocks with identical float64 math. Pattern (for `nbody_gravity_force`; the other N-body references follow the same row-block structure over their own formula):
```python
# SPDX-License-Identifier: Apache-2.0
import numpy as np

EPS = 1e-2
CHUNK = 1024  # rows per block: bounds the pairwise temporaries to CHUNK x N float64

def run(inputs, shape):
    pos = inputs[0].astype(np.float64); n = pos.size
    out = np.empty(n)
    for s in range(0, n, CHUNK):
        e = min(s + CHUNK, n)
        d = pos[None, :] - pos[s:e, None]  # d[i, j] = pos[j] - pos[i]
        contrib = d / np.power(d * d + EPS, 1.5)
        contrib[np.arange(e - s), np.arange(s, e)] = 0.0
        out[s:e] = contrib.sum(axis=1)
    return [out.astype(np.float32)]
```
For every rewritten reference, add to that problem's `tests/test_problem_<name>.py` (keep the pre-chunk implementation inline in the test as `_full`):
```python
def test_chunked_reference_matches_full():
    import numpy as np
    from minidwarf.problem_io import load_module_fn
    gen = load_module_fn(P, "inputs.py", "generate"); ref = load_module_fn(P, "reference.py", "run")
    ins = gen([3000], 5)
    np.testing.assert_allclose(ref(ins, [3000])[0], _full(ins, [3000])[0], rtol=1e-12, atol=0)
```

- [ ] **Step 4: Write the chosen shapes into each `spec.yaml`, then verify all at once**

Run: `python scripts/size_shapes.py --check-all`
Expected: exit 0, every row ≥ 1 ms and ≤ 2000 MB; writes `harness/shape_report.json`.

- [ ] **Step 5: Calibrate tolerances**

Run: `python scripts/calibrate_tol.py`
For each `too_loose` problem, tighten `rtol`/`atol` so the expert's `tol_ratio` lands in [0.01, 1] (target ≈ 0.05–0.2, i.e. 5–20× headroom over the expert); for each `fail`, loosen the tolerance or fix the expert if its error is a genuine bug. Re-run until exit 0; writes `harness/tolerance_report.json`.

- [ ] **Step 6: Run the invariant and tolerance tests**

Run: `python -m pytest tests/test_problem_invariants.py tests/test_tolerances.py tests/test_problem_*.py -v`
Expected: all PASS.

- [ ] **Step 7: Measure the noise floor**

Run: `python scripts/noise_floor.py`
Expected: exit 0, prints `eps = 0.0xx`; writes `harness/noise_floor.json`.

- [ ] **Step 8: Confirm no prompt leaks shapes:** `grep -l "check_shapes\|eval_shapes" problems/*/*/prompt.md` → no output.

- [ ] **Step 9: Commit**
```bash
git add problems harness/shape_report.json harness/tolerance_report.json harness/noise_floor.json tests/test_problem_invariants.py tests/test_tolerances.py tests/test_problem_*.py
git commit -m "Resize eval shapes to >=1 ms baselines, calibrate tolerances, record noise floor"
```

---

### Task 13: Declare v3 (docs)

**Files:**
- Modify: `README.md`, `CLAUDE.md`, `LEADERBOARD.md`

- [ ] **Step 1: README.** Make these edits:
  - In **Versioning**, mark v2 deprecated: "**v2** (deprecated): harness admits output caching, linked vendor libraries and launch-overhead-dominated timings; do not cite v2 numbers." Add "**v3** (in progress): hardened harness (this release) + new dwarfs (upcoming); report as `MiniDwarf v3`."
  - Add a section **Harness v3: threat model and timing protocol** summarising spec §1–§3: the threat table, the per-binary sequence (untimed first call, 3 warm-up, 20 timed reps over 4 rotating data sets, NaN poison, L2 flush, device-wide sync), ABBA, geomean speedup, the noise floor ε and the `fast_p@p ≥ p·(1+ε)` rule, the GPU preflight (`--allow-busy-gpu`), the stated limitation (content-keyed caching is audited manually for speedups > 1.5×).
  - In **Usage**: document `minidwarf score --run-dir runs/<id>`, `MINIDWARF_CACHE`, and `scripts/size_shapes.py` / `calibrate_tol.py` / `noise_floor.py`.
  - Explain that `fast_p@p` is a speedup threshold, not pass@k.

- [ ] **Step 2: CLAUDE.md.** Under *Adding a problem*: spec.yaml now also needs `check_shapes` (≥ 3, at least one non-multiple-of-32 leading dim), eval shapes sized with `scripts/size_shapes.py` (baseline ≥ 1 ms, ≤ 2000 MB), tolerance calibrated with `scripts/calibrate_tol.py`, optional `allowed_libs`. Under *Running an evaluation*: the GPU preflight, `minidwarf score`, and that runs scored by an older harness are skipped by the leaderboard. Under *Build & test*: the GPU-less CI list now also includes lint/preflight/refcache/check_shapes/correctness/grade_lint/scripts tests; `tests/test_redteam.py` proves the harness defenses.

- [ ] **Step 3: Regenerate the leaderboard** (old v2 runs are skipped): `minidwarf leaderboard --runs-dir runs --out LEADERBOARD.md`. Expected: header only plus "Skipped N run(s)…" until runs are re-scored with `minidwarf score`.

- [ ] **Step 4: Full suite** (GPU gate): `python -m pytest -q`
Expected: all PASS.

- [ ] **Step 5: Commit**
```bash
git add README.md CLAUDE.md LEADERBOARD.md
git commit -m "Declare harness v3: document threat model, timing protocol, deprecate v2"
```
