# Defense matrix: evaluation-harness integrity in LLM GPU-kernel benchmarks

Static, read-only review of the default evaluation path (no code from these repositories was executed).
Purpose: compare integrity mechanisms, not to publish attacks. To be shared with the maintainers before
any publication.

Commits reviewed: KernelBench `423217d9fda91e0c2d67e4a43bf62f96f6d104f1` · TritonBench
`603e28a5050e8c268f6883a69709d477a272d49a` · robust-kbench `078f5bab29934a822268d59a4e707d449abf9b4e` ·
MiniDwarf v3 `760acf4`.

| # | Mechanism | KernelBench | TritonBench | robust-kbench | MiniDwarf v3 |
|---|---|---|---|---|---|
| 1 | Outputs of the timed runs are verified | NO — `timing.py:263` discards the timed output; correctness is a separate phase (`eval.py:584-606`) | NO — `performance_utils.py:124-125` keeps only the time | NO — `sandbox/eval_forward_fn.py:30-32` | YES — the driver verifies every call (untimed, warm-up, timed) against expected outputs (`harness/driver.cu`) |
| 2 | Inputs vary between calls / phases | PARTIAL — 5 correctness seeds from fixed 42; one fixed timing input (`eval.py:615-618,750-764`) | PARTIAL — distinct test/perf sets, mostly unseeded, same tensor every timed call | PARTIAL — fixed seed 42; single timing input | YES — 4 data sets rotated in random order through the same buffers, contents rewritten before every call (hidden fixed seeds) |
| 3 | Timer covers all device work | PARTIAL — events on the current stream (`timing.py:215,266-268`); opt-in `host_time` covers all streams | PARTIAL — `triton.testing.do_bench` on the current stream | PARTIAL — current-stream events | YES — `cudaDeviceSynchronize()` inside the timed region |
| 4 | Output buffers reset/poisoned between calls | N/A — candidate returns its own outputs | N/A | N/A | YES — NaN poison before every call |
| 5 | Candidate isolated from grader state | PARTIAL — per-sample worker, but candidate, reference, comparator and timer share one process | PARTIAL — correctness in a subprocess; timing in the candidate's process | PARTIAL — subprocess per phase, but reference, comparison, timing and result JSON in the same process | PARTIAL — Python grader and comparator are in a separate process; expected outputs live in the driver process (documented residual risk, manual audit) |
| 6 | Library/fallback use detected and enforced | PARTIAL — try/except fallback strict; torch ops warnings only | NO | NO | YES — no vendor link flags for candidates; regex lint + object-level symbol/definition/syscall checks → `forbidden_api` |
| 7 | Static checks enforced in the main eval path | PARTIAL — enforced at generation, not in `eval_from_generations.py` | NO | NO | YES — every candidate, before linking |
| 8 | Tolerance documented and per-problem calibrated | PARTIAL — documented, same for all problems | NO — exact stdout match | NO — global atol/rtol | YES — per problem, expert `tol_ratio` within [0.01, 1] (`scripts/calibrate_tol.py`) |
| 9 | L2 flush between timed runs | YES (256 MB) | YES (`do_bench`) | PARTIAL — default mode does not flush | YES (2 × L2 size) |
| 10 | Timing noise quantified and used in the metric | NO — std recorded, unused | NO | NO — IQR recorded, unused | YES — per-problem A/A noise floor; `fast_p@p` uses `speedup ≥ p·(1+ε_p)`; unstable problems flagged |
| 11 | Implausible speedups acted upon | NO — flag set (`eval.py:691-695`), not read by `score.py` | YES — ≥10× (or ≤0.1×) dropped from the average | NO | PARTIAL — protocol: every kernel > 1.5× is audited manually (not automated) |

Defaults — KernelBench: atol = rtol = 1e-4 (fp32), 3 warm-up + 100 timed trials, value-based correctness.
TritonBench: no numeric tolerance (stdout string equality), `do_bench` 100 ms warm-up / 1000 ms reps, median.
robust-kbench: 1e-3 (CLI) / 1e-5 (library), 25 warm-up + 1000–10000 timed calls, value-based, `--use_fast_math`.
MiniDwarf v3: per-problem calibrated tolerances, 3 warm-up + 20 timed calls × BAAB, 4 data sets, geomean speedup.

Notes for maintainers (factual observations from the review):
- TritonBench: at this commit only 7/184 G and 1/166 T test sections print anything after the separator, so
  the stdout comparison in `1_exe_acc.py` compares outputs without printed results for most problems.
- robust-kbench: `run_kernel.py:21-22` declares `--multi_*_settings` with `type=bool`, so any non-empty
  string parses as `True`.
- KernelBench: the adversarial streams unit test passes when the excessive-speedup flag is set; it does not
  check that the result is rejected.

Caveat: verdicts are from reading code at the commits above; empirical confirmation is pending and should be
coordinated with the maintainers.
