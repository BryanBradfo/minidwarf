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

## v2: 4 dwarfs, 24 problems

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

Problems currently in v2:

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
- **v2** (frozen, current): 4 dwarfs (adds Dense Linear Algebra and Sparse
  Linear Algebra), 24 problems, 12/12 valid/test.

**MiniDwarf v2 is frozen at this 24-problem set.** Report results as
"MiniDwarf v2" (or `MiniDwarf-v2-valid` / `MiniDwarf-v2-test` if you need
to distinguish the split) so numbers stay comparable across papers and
runs. v1 numbers remain valid as "MiniDwarf v1" and are not superseded by
v2 -- they cover a different (smaller) problem set and are not directly
comparable to a v2 score. If problems are added or changed later, that
will ship as a new named version (v2.1, v3, ...) rather than silently
mutating v2 -- a score against "MiniDwarf" without a version is not
reproducible.

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

Run `python scripts/check_env.py` to verify your toolchain can compile and
run a trivial `sm_120` kernel before grading anything.

## Install

MiniDwarf is meant to be run from a source checkout, not installed as a
self-contained wheel: the CUDA driver (`harness/driver.cu`) and the
`problems/` tree live in the repo and are resolved by the CLI relative to
the checkout at runtime. Clone the repo and install it editable so the
`minidwarf` package points back at that checkout:

```bash
git clone <this repo>
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

## Grading pipeline

Each submitted kernel goes through the same four stages:

1. **Compile.** The candidate `.cu` file and the problem's `baseline.cu`
   are both compiled with `nvcc`. A compile failure on the candidate is
   scored `compile_error` immediately.
2. **Correctness.** The candidate is run on the problem's `eval_shapes`
   (input shapes not disclosed in `prompt.md`) with seeded random inputs,
   and its outputs are compared against a NumPy reference (`reference.py`)
   with the problem's `rtol`/`atol`.
3. **Anti-gaming.** Because `eval_shapes` are not fixed at prompt time
   (and are shape-varied, not just re-seeded), a kernel that special-cases
   a hardcoded shape or memorizes an expected output will fail correctness
   on at least one shape.
4. **Timing.** Both the candidate and the baseline are run for multiple
   reps; each rep's device-side GPU time is measured with `cudaEvent`
   (not host wall-clock time). For each `eval_shapes` entry the median of
   its reps is taken, and the reported speedup is
   `sum(baseline medians) / sum(candidate medians)` across all shapes (see
   `grade.py`) -- not a single wall-clock median.

## The `fast_p` metric

`fast_p` is the fraction of problems a submission gets both **correct**
*and* **at least `p`x faster than the honest baseline** kernel shipped
with the problem (the baseline is a straightforward, unoptimized
implementation -- not a strawman, but not tuned either). `fast_p(0)` is
just the correctness rate (any non-negative speedup counts), and
`fast_p(1)`, `fast_p(2)`, `fast_p(5)`, ... report the fraction that clears
increasingly demanding speed bars. `minidwarf suite` reports the full
curve alongside the raw `compile_rate` and `correctness_rate`, since a low
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

Unlike v1 (where every shipped `solutions/expert_v1.cu` is
byte-identical to its `baseline.cu`, see "Known limitations" below),
**the Dense and Sparse dwarfs ship real, distinct hand-written expert
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

## Known limitations (v2)

- **Shipped experts equal the baselines -- but only for the 12 v1
  problems.** The `solutions/expert_v1.cu` file for every problem under
  `structured_grids/` and `nbody/` (the original v1 set) is byte-identical
  to that problem's `baseline.cu`. They exist to prove the problem is
  solvable within its `rtol`/`atol` with a real kernel, not as optimized
  reference implementations -- expect `speedup ~= 1.0` when grading them,
  not a demonstration of achievable speedup. This caveat does **not**
  apply to the 12 `dense/` and `sparse/` problems added in v2: their
  shipped experts are real, distinct hand-written kernels that differ
  from the (often vendor-library) `baseline.cu` -- see "Vendor baselines"
  above. Optimized expert solutions (`expert_v2.cu`, etc.) for the v1
  problems are welcome contributions; see [CONTRIBUTING.md](CONTRIBUTING.md).
- **Timing trust model assumes a non-adversarial candidate.** The driver
  (`harness/driver.cu`) reuses the same input/output device buffers across
  the warmup call and all timed reps, and correctness is checked once from
  the final buffer contents after timing completes. This means the timing
  path trusts that the candidate actually does the same work on every
  rep -- a deliberately adversarial kernel could, in principle, do its real
  work during the untimed warmup call and no-op (or memoize) during the
  timed reps to inflate its reported speedup. v2 does not defend against
  this; the intended mitigation for a future version is to re-randomize
  inputs before each timed rep (a naive per-rep memset was deliberately
  not added in this pass, since it would distort the speedup ratio for
  honest kernels without a properly designed fix).
- **The anti-gaming property is against the prompt, not against repo
  access.** `eval_shapes` (in each problem's `spec.yaml`) and the grading
  `seed` (in `grade.py`) are committed in this repository. The guarantee
  described above ("a kernel can't hardcode a shape or memorize an
  output") holds for a model that sees only `prompt.md` -- it does not
  hold against someone who reads the repository itself, since the eval
  shapes and seeds are not secret from a repo-level adversary.

## License

Apache-2.0. See [LICENSE](LICENSE).
