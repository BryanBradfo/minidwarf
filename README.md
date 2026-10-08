# MiniDwarf

**MiniDwarf** is an HPC-first GPU kernel benchmark for LLMs. Instead of
covering ML-op-shaped kernels (attention, matmul, softmax, ...), MiniDwarf
organizes its problems by the [Berkeley "Dwarfs"](https://view.eecs.berkeley.edu/wiki/Dwarfs)
taxonomy of computational patterns that recur across scientific and
high-performance computing: structured grids (stencils on a mesh), N-body
(all-pairs / particle interactions), dense linear algebra (GEMM-family
kernels graded against a real vendor BLAS), and sparse linear algebra
(CSR-format kernels graded against a real vendor sparse library). The goal
is to measure whether a model can write a correct, *fast* CUDA kernel for
the kind of numerical code that shows up in simulation and scientific
computing, not just the kernels that happen to be popular in deep learning
frameworks.

Every problem is graded fully automatically: a model's kernel is compiled,
checked for correctness against a NumPy reference on held-out shapes, and
timed against an honest baseline kernel. There is no human or LLM judge in
the loop.

## Why not another ML-kernel benchmark?

Existing GPU-kernel-for-LLMs benchmarks (KernelBench and similar) are
built around the operator vocabulary of ML training/inference: GEMM,
convolution, normalization, attention. That is a useful but narrow slice of
GPU programming. MiniDwarf instead draws its problems from the dwarfs that
dominate classical HPC and scientific simulation, where the sources of
difficulty are different: irregular memory access patterns (stencil halos,
all-pairs interactions), numerically sensitive reductions, and problems
where the "obvious" parallel decomposition is not the fast one. A model
that has memorized flash-attention-shaped kernels gets no special help
here.

## Problem set: 4 dwarfs, 24 problems

| Dwarf             | # problems | valid | test |
|--------------------|-----------:|------:|-----:|
| Structured Grids   |          6 |     3 |    3 |
| N-Body             |          6 |     3 |    3 |
| Dense Linear Algebra |        6 |     3 |    3 |
| Sparse Linear Algebra |       6 |     3 |    3 |
| **Total**          |     **24** | **12** |**12** |

The exact split is pinned in [`SPLITS.yaml`](SPLITS.yaml). As with
MiniF2F, `valid` is meant for iterating on a method (prompting strategy,
agent scaffold, fine-tuning, ...) and `test` is meant for reporting a
final, held-out number. Do not tune against `test`.

Problems currently in the set (eval shapes resized for v3):

- **structured_grids**: `jacobi_3d_7pt`, `laplacian_2d_5pt`, `gauss_blur_2d`,
  `sobel_2d`, `heat_2d_step`, `wave_2d_step`
- **nbody**: `nbody_gravity_force`, `nbody_potential`, `coulomb_1d`,
  `knn_distance_1d`, `pairwise_mean_dist`, `nbody_count_within_radius`
- **dense**: `sgemm`, `gemv`, `transpose`, `gemm_bias`, `syrk`, `scaled_gemm`
- **sparse**: `spmv_csr`, `spmm_csr`, `spmv_transpose`, `csr_row_scale`,
  `sddmm`, `jacobi_sparse`

## Versioning

MiniDwarf reports are only comparable when tied to a named, frozen
version:

- **v1** (frozen): 2 dwarfs (Structured Grids, N-Body), 12 problems, 6/6
  valid/test.
- **v2** (deprecated): harness admits output caching, linked vendor
  libraries and launch-overhead-dominated timings; do not cite v2 numbers.
  4 dwarfs, 24 problems, 12/12 valid/test.
- **v3** (in progress): hardened harness (this release, `harness_version` 3)
  + new dwarfs (upcoming); report as `MiniDwarf v3`.

Report results as "MiniDwarf v3" (or `MiniDwarf-v3-valid` /
`MiniDwarf-v3-test`) so numbers stay comparable across papers and runs.
Problems are never silently mutated within a named version -- a score against
"MiniDwarf" without a version is not reproducible. The leaderboard skips runs
scored by a different harness version; re-score old generations with
`minidwarf score` (below).

## Reference hardware

Timings (and therefore `fast_p`, see below) are hardware-dependent.
Numbers reported without a hardware note are not comparable. The reference
hardware for MiniDwarf is:

- GPU: NVIDIA RTX 5070 (Blackwell, `sm_120`), 8 GB VRAM
- CUDA Toolkit >= 12.8, with `nvcc` on `PATH`
- cuBLAS and cuSPARSE, which ship with the CUDA Toolkit (no separate
  install): the Dense and Sparse dwarfs link against them to build the
  `baseline.cu` and `solutions/` binaries for every `dense`/`sparse`
  problem whose `spec.yaml` declares `baseline: cublas` or
  `baseline: cusparse` (see "Kernel ABI contract" below).

The reference laptop GPU switches memory clocks (12001 <-> 9001 MHz) under its
power cap, and clocks cannot be locked without root; see the noise floor below.

Run `python scripts/check_env.py` to verify your toolchain can compile and
run a trivial `sm_120` kernel before grading anything.

## Install

MiniDwarf is meant to be run from a source checkout, not installed as a
self-contained wheel: the CUDA driver (`harness/driver.cu`) and the
`problems/` tree live in the repo and are resolved by the CLI relative to
the checkout at runtime. Clone the repo and install it editable so the
`minidwarf` package points back at that checkout:

```bash
git clone https://github.com/BryanBradfo/minidwarf.git
cd minidwarf
pip install -e .
python scripts/check_env.py
```

## Usage

Grade a single kernel against a single problem:

```bash
minidwarf run --problem problems/structured_grids/jacobi_3d_7pt --kernel my_kernel.cu
```

Grade a directory of kernels (one `.cu` file per problem, named
`<problem_name>.cu`) against every problem under a root:

```bash
minidwarf suite --root problems --kernels my_kernels_dir
```

`minidwarf suite` prints a JSON summary (see `fast_p` below); a problem
whose kernel file is missing is scored as a compile error rather than
skipped, so incomplete submissions are penalized rather than silently
excluded.

To get the exact prompt text a model should see for a given problem
(never including eval shapes -- see [CONTRIBUTING.md](CONTRIBUTING.md)):

```bash
python scripts/gen_prompt.py problems/structured_grids/jacobi_3d_7pt
```

### Evaluating an LLM end-to-end

To run a model against a whole split and score it automatically, describe the
model with a small YAML config (one per model) and run `minidwarf eval`. The
model backend is any OpenAI-compatible endpoint, so **Ollama**, **vLLM**, and
hosted APIs (e.g. Gemini's OpenAI-compatible endpoint) all work by swapping
`base_url` / `provider_model_name` / `api_key_env_var` -- see the ready-made
examples in [`configs/eval/`](configs/eval) (`qwen-coder-ollama.yaml`,
`qwen-coder-vllm.yaml`, `gemini-flash.yaml`). Local servers need no API key.

```bash
# generate a kernel per problem, then grade every one (writes runs/<id>/)
minidwarf eval --config configs/eval/qwen-coder-ollama.yaml --split valid

# aggregate every run under runs/ into the leaderboard
minidwarf leaderboard --runs-dir runs --out LEADERBOARD.md
```

```bash
# re-score an existing run's kernels with the current harness (no model calls)
minidwarf score --run-dir runs/<id>
```

`eval` and `score` take `--allow-busy-gpu` (development only; see the GPU
preflight below). Reference inputs/outputs are cached as `.npz` under
`$MINIDWARF_CACHE` (default `~/.cache/minidwarf`); the cache can reach ~70 GB,
accumulates stale entries when problems change, and is safe to delete (it is
rebuilt on demand).

Maintainer scripts: `scripts/size_shapes.py <problem> <shape>...` times a
baseline with the v3 driver and prints median ms and device bytes (used to
size eval shapes); `scripts/calibrate_tol.py` grades the expert kernels and
records normalized errors to `harness/tolerance_report.json`;
`scripts/noise_floor.py` runs the A/A noise-floor measurement
(`harness/noise_floor.json`). `harness/shape_report.json` records the shape
sizing.

`eval` splits into generation (prompt the model, extract its ``` ```cuda ``` block,
save the kernel + raw response under `runs/<run_id>/`) and scoring (grade each
kernel with the same pipeline as `minidwarf run`), so re-scoring never re-runs
the model. Set `n_samples` in the config for pass@k (the leaderboard reports
best-of-n). `runs/` is gitignored; commit only the generated `LEADERBOARD.md`.

## Harness v3: threat model and timing protocol

Full design: [`docs/superpowers/specs/2026-10-07-hardened-harness-design.md`](docs/superpowers/specs/2026-10-07-hardened-harness-design.md).
Each submitted kernel goes through compile (with static checks), correctness,
and timing, with no human or LLM judge.

**Threat model.** Known reward hacks and their defenses:

| Hack | Defense |
|---|---|
| Output caching / call counting / skipping timed reps | 4 data sets rotated through the same buffers in a random (seeded from `std::random_device`) order; outputs NaN-poisoned before every call; **every** call (untimed, warm-up, timed; candidate and baseline) verified in the driver against expected outputs |
| Rewriting the pointer/dims arrays | Driver re-copies them from master copies before each call |
| Side-stream work escaping the timer | `cudaDeviceSynchronize()` before the end event |
| Vendor libraries (cuBLAS, cuSPARSE, Thrust, CUB, ...) | Candidate compiled without vendor link flags; regex lint (also flags raw strings and constructor attributes); optional per-problem `allowed_libs`; status `forbidden_api` |
| Escapes the regex can miss (file/process/dlopen/syscalls, interposing driver symbols) | Candidate compiled once to an `-O3` object and statically checked with `nm`/`objdump`: banned undefined symbols (`symbol:`), interposing definitions (`defines:`, with a W/V allowlist for driver template names), inline syscalls (`asm:syscall`); then linked with the driver first |
| Hardcoded or edge-case-blind kernels | Hidden `eval_shapes`; plus `check_shapes` (small/odd sizes, correctness only) and all 4 data sets |

**Stated residual risks.** Content-keyed caching (hash the inputs, replay),
in-process memory scanning, and obfuscated syscalls are not detected: the
static checks are a tripwire, not a sandbox. Every kernel with speedup > 1.5x
is audited manually before results are published. A process sandbox
(bubblewrap/nsjail) is future work.

**Timing protocol.** Eval shapes are sized so the baseline >= 1 ms and <= 2000 MB device memory (measured on the reference GPU: >= 1.4 ms, <= 1.1 GB). Per binary, the driver does an untimed first call,
3 warm-up reps, then 20 timed reps over the 4 rotating data sets. Before each
call (untimed): upload the data set, NaN-poison outputs, flush L2, run a ~10 ms
device spin to stabilize clocks. The timed region is
`eventRecord` -> `minidwarf_solve` -> `cudaDeviceSynchronize` -> `eventRecord`.
For each shape the grader runs candidate, baseline, baseline, candidate
(ABBA) and pools the two runs of each binary. The reported speedup is the
geometric mean over shapes of `baseline_median / candidate_median`.

**GPU preflight.** Before timing, the grader refuses to run if another compute
process is on the GPU: the run aborts with GpuBusyError (exit code 3 from the CLI); it is never scored as a candidate failure.
`--allow-busy-gpu` overrides for development; the run
metadata records it (`allow_busy_gpu`, `busy_seen`). `scores.json` also holds
`harness_version: 3` and the environment (SM/memory clocks, temperature,
driver) at start and end (the end record is a post-run idle snapshot), plus
provenance: `git_commit`, `problems_digest` (sha256 over every problem's
`spec.yaml`, `inputs.py`, `reference.py`, `baseline.cu`) and `numpy_version`.

**Noise floor.** `scripts/noise_floor.py` runs A/A tests (baseline vs itself)
per problem: 5 repeats, floored at `eps_global` (0.006, pooled over the
stable problems); problems whose A/A max/min exceeds 1.05 are flagged
unstable and re-measured with 20 repeats. Currently unstable: `sddmm`
(eps 0.21) and five compute-bound N-body problems (eps 0.09–0.25), whose
float32 baselines push the laptop GPU into its power cap. Each problem's
epsilon `eps_p` is in `harness/noise_floor.json`; the leaderboard lists
unstable problems as low-confidence.

**Result detail.** `ProblemResult` carries `checks` (per shape/data set error
stats), `timings` (per-shape medians and IQRs), `lint`, `bad_calls` and
`baseline_bad_calls`; statuses are `ok`, `compile_error`, `forbidden_api`,
`runtime_error`, `timeout`, `wrong_output`. JSON reports are strict (non-finite
values become `null`). `tests/test_redteam.py` runs adversarial kernels
(memoization, no-ops, cuBLAS calls, file access, shape assumptions, side
streams) to prove the defenses; calibrated tolerances are checked against the
expert kernels (`harness/tolerance_report.json`).

## The `fast_p` metric

`fast_p@p` is a **speedup threshold**, not pass@k: it is the fraction of
problems a submission gets both **correct** *and* **at least `p`x faster
than the honest baseline** kernel shipped with the problem (the baseline is
a straightforward, unoptimized implementation -- not a strawman, but not
tuned either). A problem counts toward `fast_p@p` only if
`speedup >= p * (1 + eps_p)`, where `eps_p` is that problem's noise floor
(`eps_global` for unmeasured problems); the leaderboard and `minidwarf suite`
apply the same rule. `fast_p@0` is
just the correctness rate (any non-negative speedup counts), and
`fast_p@1`, `fast_p@2`, `fast_p@5`, ... report the fraction that clears
increasingly demanding speed bars. `minidwarf suite` reports the full
noise-floor-adjusted curve (and the `eps_global` it used) alongside the raw
`compile_rate` (`forbidden_api` counts as not compiled) and
`correctness_rate`, since a low
`fast_p` can come from either failing to compile/pass correctness or
simply being too slow -- those are different failure modes and should be
reported separately, not collapsed into one number.

## Kernel ABI contract

Every problem requires a single C-linkage entry point:

```c++
extern "C" void minidwarf_solve(const void* const* inputs,
                                 void* const* outputs,
                                 const long* dims, int n_dims);
```

- `inputs` and `outputs` are arrays of **untyped device pointers**
  (`void*`), already allocated by the harness; a kernel must not allocate
  or free them. Cast each input to its actual element type before use --
  every problem's `prompt.md` documents, per input, which of `float32`
  (`const float*`) or `int32` (`const int*`) it is (sparse-format arrays
  like CSR `row_ptr`/`col_idx` are `int32`; everything else is
  `float32`). **Outputs are always `float32`** (`float*`), regardless of
  what the inputs are. Typical cast pattern:

  ```c++
  const float* A       = (const float*)inputs[0];
  const int*   row_ptr = (const int*)inputs[1];
  float*       out      = (float*)outputs[0];
  ```

- `dims` gives the problem's logical shape (e.g. `[nx, ny, nz]` for a 3D
  grid, or `[R, C, NNZ]` for a sparse matrix), `n_dims` its length.
  **Per-array sizes are decoupled from `dims`**: a kernel must not assume
  every input/output array has `prod(dims)` elements. For example, in
  `sgemm` (`dims = [M, N, K]`) `A` has length `M*K`, `B` has length `K*N`,
  and `C` has length `M*N` -- three different lengths from one `dims`
  triple. In `spmv_csr` (`dims = [R, C, NNZ]`) the CSR arrays have length
  `NNZ` or `R+1`, not `R*C*NNZ`. Each problem's `prompt.md` states the
  exact length of every input and output array explicitly; don't infer it
  from `dims` alone.
- `minidwarf_solve` must ensure all outputs are fully written and the
  device is synchronized before returning to the caller.
- No external libraries beyond `<cuda_runtime.h>` are permitted in a
  *candidate* kernel (no cuBLAS, cuDNN, Thrust, etc.) unless a specific
  problem's `prompt.md` says otherwise.

### Vendor baselines (Dense / Sparse)

For the Dense and Sparse dwarfs, the honest baseline a candidate is
graded against is frequently a real vendor library call rather than a
hand-rolled kernel: `spec.yaml`'s `baseline` field is one of
`author_kernel`, `cublas`, or `cusparse`. When it is `cublas` or
`cusparse`, `baseline.cu` calls the corresponding vendor API (e.g.
`cublasSgemm`, `cusparseSpMV`) and the harness links the extra
`-lcublas`/`-lcusparse` flag for both the candidate and baseline builds
(`minidwarf/baselines.py`). This means `fast_p` for those problems
measures speedup over a real, non-strawman vendor implementation, not
over another naive kernel. Vendor baselines hoist one-time setup (handle
creation, `cusparseCreateCsr`/`cusparseCreateDnVec` descriptors, SpMV
work-buffer allocation) out of the timed path via function-local
`static` state initialized on first call, so the comparison is
apples-to-apples against a candidate kernel that has no such per-call
setup cost.

Unlike the Structured Grids dwarf (whose shipped `solutions/expert_v1.cu`
files are byte-identical to `baseline.cu`, see "Known limitations" below),
**the N-Body, Dense and Sparse dwarfs ship real, distinct hand-written expert
kernels**: e.g. `dense/sgemm`'s `expert_v1.cu` is a shared-memory tiled
SGEMM kernel, and `sparse/spmv_csr`'s `expert_v1.cu` is a one-thread-
per-row CSR kernel -- both are different code from their cuBLAS/cuSPARSE
`baseline.cu`, and both are graded to demonstrate the problem is
solvable with a real (not vendor-library) kernel. A handful of sparse
problems (`csr_row_scale`, `sddmm`, `jacobi_sparse`) have no clean
single cuSPARSE call for their exact operation on the current CUDA
Toolkit and so use `baseline: author_kernel` (a straightforward
hand-written baseline) instead, same as v1.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full per-problem file
layout and how to add new problems.

## Known limitations

- **Shipped experts equal the baselines -- but only for the 6
  `structured_grids/` problems.** Their `solutions/expert_v1.cu` is
  byte-identical to `baseline.cu`: it proves the problem is solvable within
  its `rtol`/`atol`, not achievable speedup (expect `speedup ~= 1.0`). The
  `nbody/` experts are shared-memory tiled float32 kernels measured against
  naive float32 baselines (both sides float32, so speedups reflect kernel
  design, not an FP64-to-FP32 switch), and the `dense/`/`sparse/` experts
  are distinct hand-written kernels against (often vendor-library)
  baselines -- see "Vendor baselines" above. Optimized stencil experts
  (`expert_v2.cu`, etc.) are welcome contributions; see
  [CONTRIBUTING.md](CONTRIBUTING.md).
- **Best-of-n selection bias.** `scores.json` keeps the fastest correct of
  n samples, timed once; with n > 1 this max is biased upward by timing
  noise, so a best-of-n speedup should be re-timed in a fresh grade (or
  eps scaled with n) before it is reported (planned for SP5).
- **Static checks are not a sandbox.** v3 defends against output caching,
  call counting and vendor-library use (see the threat model), but
  content-keyed caching, in-process memory scanning and obfuscated syscalls
  are not detected; kernels with speedup > 1.5x are audited manually.
- **Laptop-GPU timing noise.** Clock switching under the power cap cannot be
  locked without root; per-problem noise floors and the unstable-problem list
  quantify it.
- **The anti-gaming property is against the prompt, not against repo
  access.** `eval_shapes` (in each problem's `spec.yaml`) and the grading
  `seed` (in `grade.py`) are committed in this repository. The guarantee
  described above ("a kernel can't hardcode a shape or memorize an
  output") holds for a model that sees only `prompt.md` -- it does not
  hold against someone who reads the repository itself, since the eval
  shapes and seeds are not secret from a repo-level adversary.

## Citation

If you use MiniDwarf, please cite it -- see [`CITATION.cff`](CITATION.cff) for
the full, machine-readable citation metadata (Citation File Format v1.2.0).
Plain-text equivalent:

```
Chen, Bryan. "MiniDwarf: an HPC-first GPU kernel benchmark for LLMs." 2026.
https://github.com/BryanBradfo/minidwarf
```

## License

Apache-2.0. See [LICENSE](LICENSE).
