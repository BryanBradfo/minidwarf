# Hardened grading harness (MiniDwarf v3, sub-project 1) — design

Status: approved in conversation 2026-10-07, pending written-spec review.

## Context

MiniDwarf is being turned into a workshop paper: *"Which HPC computational
patterns do LLMs fail at, and why?"* (target: ICLR/ICML 2027 workshops). The
work is split into five sub-projects, each with its own spec:

1. **Hardened harness** (this document)
2. Failure instrumentation (sanitizers, targeted checks → failure taxonomy)
3. Roofline efficiency metric
4. MiniDwarf v3 problem set (4 → 6 dwarfs, canonical/perturbed pairs)
5. Evaluation campaign and paper

Sub-project 1 is a prerequisite for all others: the v2 harness admits known
reward hacks and times microsecond-scale runs, so no number it produces is
publishable.

### Decisions already taken

- The hardened harness and resized eval shapes ship as **MiniDwarf v3**
  (together with the new dwarfs of sub-project 4). v2 is declared
  **deprecated** ("vulnerable harness, do not cite"); no external v2 results
  exist, so nothing is lost.
- Hardening lives in `harness/driver.cu`, orchestrated by the existing Python
  modules. Rejected alternatives: in-process ctypes/CuPy timing (candidate
  shares the grader's process, new dependency, full rewrite) and one process
  per rep (100–300 ms CUDA context start-up per rep).

## Goals / non-goals

Goals: close every known reward hack listed below; make timings meaningful on
the reference laptop GPU; emit per-check / per-shape detail that sub-project 2
can consume; prove the defenses with an adversarial test suite.

Non-goals: sanitizer-based failure classification (SP2), roofline (SP3), new
problems or perturbed variants (SP4), model runs (SP5).

## 1. Threat model

| Hack | Defense | Location |
|---|---|---|
| Output caching across calls, keyed by pointer (`static bool done`) | **D = 4 distinct input data sets** rotated *through the same device buffers*; contents rewritten (untimed) before every rep; the output of the last timed rep of *each* set is verified | driver |
| No-op / partial writes reusing stale output | Output buffers poisoned with NaN (`0xFF` bytes) before every rep, including the first untimed call | driver |
| Call counting: skipping the unverified timed reps (e.g. `if (++calls > 4) return;`), or stashing outputs by call index | **Every call is verified** (untimed, warm-up, timed) against the expected outputs inside the driver (allclose semantics, NaN fails; any failure → `wrong_output`); the data-set order is a **random permutation** seeded from `std::random_device` | driver |
| Rewriting the pointer/dims arrays passed by the driver | Driver keeps master copies and re-copies them into the arrays handed to the candidate before every call | driver |
| Work on a non-blocking side stream escaping the timer | `cudaDeviceSynchronize()` before recording the end event | driver |
| Calling cuBLAS / cuSPARSE / Thrust / CUB / cuFFT / cuRAND | Candidate compiled **without** vendor link flags (baseline keeps them); static lint → new status `forbidden_api`. Per-problem allowlist `allowed_libs` in `spec.yaml` (default empty) for future problems (e.g. cuRAND for Monte Carlo) | `grade.py`, new `lint.py` |
| Same-process escapes the regex can miss (macros, `##`, raw strings, POSIX `open`/`read`/`mmap`, `exit`, host threads, `cudaSetDevice`) | **Symbol check**: the candidate is also compiled alone (`nvcc -c`) and rejected (`forbidden_api`) if `nm -u` lists a banned libc/CUDA/vendor symbol; regex lint also flags `__attribute__((constructor))` | `grade.py`, `lint.py` |
| Reading files, spawning processes, dynamic loading | Lint bans `fopen`, `fread`, `ifstream`, `system`, `popen`, `exec*`, `dlopen`, `dlsym`, `getenv` → `forbidden_api` | `lint.py` |
| Hardcoding shapes / test values | Already mitigated by hidden `eval_shapes`; strengthened by `check_shapes` (§3) and D = 4 data sets | spec / driver |
| Content-keyed caching (hash inputs, replay) | **Out of scope, stated in the paper.** Mitigation by protocol: every kernel with speedup > 1.5× is manually audited before results are published | protocol |

The lint is a regex scan over the candidate source after stripping comments
and string literals; it is a tripwire, not a sandbox, and the paper says so.

## 2. Timing protocol

- **Shapes:** eval shapes are re-chosen so that the baseline runs **≥ 1 ms**
  per call on the reference GPU (RTX 5070 Laptop, sm_120), with total device
  memory ≤ 2 GB per problem. `scripts/size_shapes.py <problem> <shape>...`
  compiles the baseline, times it with the v3 driver, and prints median ms and
  device bytes, so shapes are picked by measurement, not guesswork.
- **Per-binary sequence (driver):**
  1. untimed first call on data set 0 (outputs poisoned first);
  2. W = 3 warm-up reps;
  3. R = 20 timed reps. Before each rep, untimed: copy data set `r % D` into
     the input buffers, poison outputs, flush L2 by `cudaMemsetAsync` over a
     scratch buffer of 2 × `cudaDevAttrL2CacheSize` bytes, synchronize.
     Timed region: `eventRecord(start)` → `minidwarf_solve` →
     `cudaDeviceSynchronize()` → `eventRecord(end)`.
  4. After the last timed rep of each data set, copy its outputs to host;
     the driver writes all D output sets.
  The driver reports median, p25, p75 and all raw rep times as JSON.
- **ABBA ordering:** for each eval shape the grader runs candidate, baseline,
  baseline, candidate, pooling the two runs of each binary, to cancel thermal
  drift. Correctness is checked on every candidate run.
- **Speedup:** geometric mean over eval shapes of
  `baseline_median / candidate_median` (v2 used ratio of sums, which lets the
  largest shape dominate).
- **Noise floor:** `scripts/noise_floor.py` runs an A/A test (baseline vs
  itself, ABBA) over all problems and writes ε (95th percentile of
  |log speedup|) to `harness/noise_floor.json`. Reported speedups keep their
  raw value; the leaderboard's `fast_p@p` counts a problem only if
  `speedup ≥ p · (1 + ε)`.
- **GPU preflight:** before any timing, the grader queries
  `nvidia-smi --query-compute-apps` and refuses to time (status
  `gpu_busy`, raised as an error for the whole run, not scored as a candidate
  failure) if another compute process is present. `--allow-busy-gpu` overrides
  for development, and the run metadata records that it was used.
- **Environment record:** SM clock, memory clock, temperature and driver
  version are captured at the start and end of each scored run into the run
  metadata. Clocks cannot be locked on the laptop without root; transparency
  replaces locking.

## 3. Correctness

- **`check_shapes`** (new optional `spec.yaml` field): small and edge shapes
  checked for correctness only (not timed): sizes not multiple of 32, primes,
  size 1, and problem-specific degenerate cases (e.g. empty CSR rows). A
  candidate is `correct` only if it passes all eval shapes × all D data sets
  and all check shapes. For the existing 24 problems, check shapes are added
  in this sub-project.
- **Calibrated tolerances:** the expert kernel (`solutions/expert_v1.cu`) is
  the problem's canonical float32 implementation. `scripts/calibrate_tol.py`
  grades it and records its normalized error
  `tol_ratio = max |a - e| / (atol + rtol·|e|)` over all checks. A test
  asserts `0.01 ≤ tol_ratio ≤ 1` for every problem, i.e. the tolerance is
  within [1×, 100×] of the error of a real float32 kernel (problems whose
  expert is bit-exact, `tol_ratio == 0`, are exempt).
- **Non-trivial references:** a test asserts no reference output is
  near-constant (std / (|mean| + 1e-12) > 1e-3) on eval shapes, so all-zero or
  constant-output kernels cannot pass.

## 3b. Case cache

Generating inputs and float64 references at the resized shapes can take
seconds to minutes (e.g. O(N²) N-body references, per-row CSR generators), and
grading uses D = 4 data sets per shape. `minidwarf/refcache.py` caches each
`(inputs, expected)` case as `.npz` under `$MINIDWARF_CACHE` (default
`~/.cache/minidwarf`), keyed by the SHA-256 of `inputs.py`, `reference.py`,
the shape and the seed, so editing a problem invalidates its cache. Writes are
atomic (temp file + rename); an unreadable cache file is recomputed.
References that do not fit in 8 GB of host RAM at the new shapes are rewritten
in chunks (same float64 math).

## 4. Result schema and versioning

- `ProblemResult` gains:
  - `checks`: list of `{shape, data_set, passed, max_abs_err, max_rel_err, nan_count}`;
  - `timings`: list of `{shape, cand_median_ms, cand_iqr_ms, base_median_ms, base_iqr_ms}`;
  - `lint`: list of forbidden matches (empty when clean).
  - New statuses: `forbidden_api`. (`gpu_busy` aborts the run instead.)
- `scores.json` gains `harness_version: 3` and the environment record. The
  leaderboard ignores runs whose `harness_version` differs from the current
  one and prints how many it skipped.
- New CLI subcommand `minidwarf score --run-dir runs/<id>` re-scores an
  existing run's kernels with the current harness (needed to re-grade
  generations made under v2); `eval` and `score` take `--allow-busy-gpu`.
- README: v2 marked deprecated; a v3 section documents the threat model,
  timing protocol, and noise floor. CLAUDE.md updated accordingly.

## 5. Testing (TDD, red-team suite first)

Adversarial fixtures under `tests/fixtures/redteam/`, each on a small,
existing problem, with the expected status:

| Fixture | Expected |
|---|---|
| `memo_static.cu` — computes once, returns early afterwards | `wrong_output` |
| `memo_copy.cu` — caches result in its own buffer, copies it back | `wrong_output` |
| `side_stream_nosync.cu` — same kernel as an honest twin, launched on a non-blocking stream without sync | `ok`, with median time ≥ 0.7 × the honest twin's (the side-stream work is inside the timer; the smoke problem runs in tens of µs, so 0.7 leaves room for noise while the v2 hack measured ≈ 0.2×) |
| `calls_cublas.cu` | `forbidden_api` |
| `fopen_grader.cu` | `forbidden_api` |
| `noop.cu` | `wrong_output` |
| `assumes_mult32.cu` — drops the tail when N % 32 != 0 | `wrong_output` (via `check_shapes`) |

Plus: GPU-less unit tests for `lint.py`, geomean speedup, the noise-floor
leaderboard rule, the schema/versioning logic and the preflight parser (these
run in CI); all `solutions/expert_v1.cu` remain `ok` and all `baseline.cu`
remain `ok` against themselves (nvcc-gated).

The red-team table is reused as the paper's "harness validation" table.

## Operational constraint

The reference GPU is shared with other workloads. Implementation proceeds with
GPU-less work first (lint, schema, scoring, leaderboard, scripts); nvcc-gated
tests and any timing run only when the preflight shows the GPU idle.

## Out of scope / follow-ups

- Sanitizer passes and failure taxonomy labels → SP2 spec.
- Bytes/FLOPs per problem and roofline efficiency → SP3 spec.
- New dwarfs, perturbed variants, ≥ 6 problems per dwarf → SP4 spec.
- Models, n samples per problem, bootstrap/mixed-effects analysis, manual
  validation of taxonomy labels → SP5 spec.
