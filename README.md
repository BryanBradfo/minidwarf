# MiniDwarf

**MiniDwarf** is an HPC-first GPU kernel benchmark for LLMs. Instead of
covering ML-op-shaped kernels (attention, matmul, softmax, ...), MiniDwarf
organizes its problems by the [Berkeley "Dwarfs"](https://view.eecs.berkeley.edu/wiki/Dwarfs)
taxonomy of computational patterns that recur across scientific and
high-performance computing: structured grids (stencils on a mesh) and
N-body (all-pairs / particle interactions). The goal is to measure whether
a model can write a correct, *fast* CUDA kernel for the kind of numerical
code that shows up in simulation and scientific computing, not just the
kernels that happen to be popular in deep learning frameworks.

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

## v1: 2 dwarfs, 12 problems

| Dwarf             | # problems | valid | test |
|--------------------|-----------:|------:|-----:|
| Structured Grids   |          6 |     3 |    3 |
| N-Body             |          6 |     3 |    3 |
| **Total**          |     **12** | **6** |**6** |

The exact split is pinned in [`SPLITS.yaml`](SPLITS.yaml). As with
MiniF2F, `valid` is meant for iterating on a method (prompting strategy,
agent scaffold, fine-tuning, ...) and `test` is meant for reporting a
final, held-out number. Do not tune against `test`.

Problems currently in v1:

- **structured_grids**: `jacobi_3d_7pt`, `laplacian_2d_5pt`, `gauss_blur_2d`,
  `sobel_2d`, `heat_2d_step`, `wave_2d_step`
- **nbody**: `nbody_gravity_force`, `nbody_potential`, `coulomb_1d`,
  `knn_distance_1d`, `pairwise_mean_dist`, `nbody_count_within_radius`

## Versioning: v1 is frozen

**MiniDwarf v1 is frozen at these 12 problems.** Report results as
"MiniDwarf v1" (or `MiniDwarf-v1-valid` / `MiniDwarf-v1-test` if you need
to distinguish the split) so numbers stay comparable across papers and
runs. If problems are added or changed later, that will ship as a new
named version (v1.1, v2, ...) rather than silently mutating v1 -- a score
against "MiniDwarf" without a version is not reproducible.

## Reference hardware

Timings (and therefore `fast_p`, see below) are hardware-dependent.
Numbers reported without a hardware note are not comparable. The reference
hardware for MiniDwarf v1 is:

- GPU: NVIDIA RTX 5070 (Blackwell, `sm_120`), 8 GB VRAM
- CUDA Toolkit >= 12.8, with `nvcc` on `PATH`

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
extern "C" void minidwarf_solve(const float* const* inputs,
                                 float* const* outputs,
                                 const long* dims, int n_dims);
```

- `inputs` and `outputs` are arrays of device pointers, already allocated
  by the harness; a kernel must not allocate or free them.
- `dims` gives the problem's shape (e.g. `[nx, ny, nz]` for a 3D grid),
  `n_dims` its length.
- `minidwarf_solve` must ensure all outputs are fully written and the
  device is synchronized before returning to the caller.
- No external libraries beyond `<cuda_runtime.h>` (no cuBLAS, cuDNN,
  Thrust, etc.) unless a specific problem's `prompt.md` says otherwise.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full per-problem file
layout and how to add new problems.

## Known limitations (v1)

- **Shipped experts equal the baselines.** The `solutions/expert_v1.cu`
  file for every v1 problem is byte-identical to that problem's
  `baseline.cu`. They exist to prove the problem is solvable within its
  `rtol`/`atol` with a real kernel, not as optimized reference
  implementations -- expect `speedup ~= 1.0` when grading them, not a
  demonstration of achievable speedup. Optimized expert solutions
  (`expert_v2.cu`, etc.) are welcome contributions; see
  [CONTRIBUTING.md](CONTRIBUTING.md).
- **Timing trust model assumes a non-adversarial candidate.** The driver
  (`harness/driver.cu`) reuses the same input/output device buffers across
  the warmup call and all timed reps, and correctness is checked once from
  the final buffer contents after timing completes. This means the timing
  path trusts that the candidate actually does the same work on every
  rep -- a deliberately adversarial kernel could, in principle, do its real
  work during the untimed warmup call and no-op (or memoize) during the
  timed reps to inflate its reported speedup. v1 does not defend against
  this; the intended mitigation for v1.1 is to re-randomize inputs before
  each timed rep (a naive per-rep memset was deliberately not added in
  this pass, since it would distort the speedup ratio for honest kernels
  without a properly designed fix).
- **The anti-gaming property is against the prompt, not against repo
  access.** `eval_shapes` (in each problem's `spec.yaml`) and the grading
  `seed` (in `grade.py`) are committed in this repository. The guarantee
  described above ("a kernel can't hardcode a shape or memorize an
  output") holds for a model that sees only `prompt.md` -- it does not
  hold against someone who reads the repository itself, since the eval
  shapes and seeds are not secret from a repo-level adversary.

## License

Apache-2.0. See [LICENSE](LICENSE).
