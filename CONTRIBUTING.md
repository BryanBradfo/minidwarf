# Contributing to MiniDwarf

Thanks for considering a contribution. This document covers how to add a
new problem, how to add an expert solution to an existing problem, and the
licensing requirements for any code you submit.

Note on commit conventions: the maintainer commits without trailers (no
`Co-Authored-By`, no attribution footers). That is a maintainer-side
convention for this repository's history, not something contributors need
to replicate -- follow normal PR etiquette (clear commit messages,
sign-off if your fork requires it, etc.) and the maintainer will handle
merge/squash conventions.

## Adding a new problem

Every problem lives at `problems/<dwarf>/<problem_name>/` and consists of
exactly six files:

```
problems/<dwarf>/<problem_name>/
  spec.yaml       # metadata: dwarf, difficulty, tolerances, shapes, baseline
  inputs.py       # generate(shape, seed) -> list[np.ndarray]
  reference.py    # run(inputs, shape) -> list[np.ndarray]  (ground truth)
  baseline.cu     # honest, unoptimized minidwarf_solve implementation
  prompt.md       # the problem statement shown to the model being graded
  solutions/      # expert reference solution(s), e.g. expert_v1.cu
```

`<dwarf>` must be one of the existing Berkeley Dwarf categories
(`structured_grids`, `nbody`, `dense`, `sparse`, ...); propose a new dwarf
category in an issue before adding problems under it.

### `spec.yaml`

Required fields (see `minidwarf/spec.py` for the loader): `name`, `dwarf`,
`difficulty`, `rtol`, `atol`, `n_inputs`, `n_outputs`, `eval_shapes`,
`baseline`. `eval_shapes` is a list of shapes (e.g. `[[128,128,128],
[160,160,96]]`) the grader will actually run the candidate and baseline
kernels against -- **these are not disclosed in `prompt.md`** (see the
anti-contamination rule below).

`baseline` must be one of `author_kernel`, `cublas`, or `cusparse`
(`minidwarf/baselines.py`'s `link_flags`); it selects both what
`baseline.cu` is expected to contain and which extra linker flags
(`-lcublas` / `-lcusparse`) are passed to `nvcc` when compiling *both* the
candidate and the baseline for this problem:

- `author_kernel`: `baseline.cu` is a straightforward, hand-written
  `minidwarf_solve` using only `<cuda_runtime.h>` -- no extra link flags.
  Use this for `structured_grids`/`nbody` problems, and for any `dense`/
  `sparse` problem that has no clean single vendor-library call for its
  exact operation (e.g. `sparse/csr_row_scale`, `sparse/sddmm`,
  `sparse/jacobi_sparse` on the current CUDA Toolkit).
- `cublas`: `baseline.cu` calls a cuBLAS routine (e.g. `cublasSgemm`) via
  `<cublas_v2.h>`; the harness links `-lcublas` automatically. Use for
  `dense` problems with a direct cuBLAS equivalent.
- `cusparse`: `baseline.cu` calls a cuSPARSE generic API routine (e.g.
  `cusparseSpMV` via `cusparseCreateCsr`/`cusparseCreateDnVec`) through
  `<cusparse.h>`; the harness links `-lcusparse` automatically. Use for
  `sparse` problems with a direct cuSPARSE equivalent.

When writing a `cublas`/`cusparse` `baseline.cu`, hoist one-time setup
(handle creation, matrix/vector descriptors, `*_bufferSize` + the
resulting work buffer) out of the timed `minidwarf_solve` path using
function-local `static` variables initialized on first call (see
`problems/sparse/spmv_csr/baseline.cu` for the pattern). This is safe
because the harness calls `minidwarf_solve` repeatedly with the same
device buffers and `dims` for a given eval shape, and it keeps the timing
comparison fair -- a candidate kernel pays no equivalent per-call setup
cost, so the baseline shouldn't either.

### `inputs.py` and `reference.py`

`inputs.py` must define `generate(shape, seed) -> list[np.ndarray]`,
producing seeded, reproducible random inputs (`np.random.default_rng(seed)`,
never the global RNG). `reference.py` must define
`run(inputs, shape) -> list[np.ndarray]` and is the ground truth every
candidate kernel is checked against.

For reductions or anything where float32 accumulation error is a concern
(sums over many particles/pixels, pairwise distances, potentials, ...),
**compute in `float64` internally and cast to `float32` only at the very
end**. This keeps the reference numerically honest so tolerances reflect
real kernel error, not reference rounding error.

**Per-array sizes are decoupled from `dims`.** In v1, every array's length
was `prod(dims)`. As of v2 this is no longer a hard invariant: `dims` is
the problem's *logical* shape (e.g. `[M, N, K]` for a GEMM, `[R, C, NNZ]`
for a sparse matrix), and each input/output array declares its own length
independently -- the harness (`minidwarf/runner.py`) computes each
array's byte count from its actual NumPy shape (`inputs.py`/
`reference.py` output), not by deriving it from `dims`. For example
`dense/sgemm` (`dims = [M, N, K]`) has `A` of length `M*K`, `B` of length
`K*N`, and `C` of length `M*N` -- three different lengths from one
`dims`; `sparse/spmv_csr` (`dims = [R, C, NNZ]`) has CSR arrays of length
`NNZ` or `R+1`, not `R*C*NNZ`. When adding a problem, pick whatever
per-array lengths are natural for the math and state each one explicitly
in `prompt.md` (see `problems/dense/sgemm/prompt.md` and
`problems/sparse/spmv_csr/prompt.md` for worked examples) -- don't assume
a reader can infer array length from `dims` alone. The old
`prod(dims)`-for-every-array convention still holds for every
`structured_grids`/`nbody` problem and remains a reasonable default when
it fits your problem.

**Input dtypes**: every input array must be `float32` or `int32` --
`minidwarf/problem_io.py`'s `write_arrays` raises `ValueError` on any
other dtype. Use `int32` for arrays that are inherently integer (e.g. CSR
`row_ptr`/`col_idx`); use `float32` for everything else. **Outputs are
always `float32`** regardless of input dtypes -- the driver
(`harness/driver.cu`) and `runner.py`'s `run_binary` always read output
buffers back as `float32`, so a `minidwarf_solve` must write `float32`
data to every output pointer even if some inputs were `int32`.

### Kernel ABI (`void**`)

Every `minidwarf_solve` implementation (`baseline.cu`, everything under
`solutions/`, and every candidate kernel graded against this problem)
must match this exact signature:

```c++
extern "C" void minidwarf_solve(const void* const* inputs,
                                 void* const* outputs,
                                 const long* dims, int n_dims);
```

`inputs`/`outputs` are arrays of untyped device pointers; cast each one
to its real element type (`const float*` or `const int*` for inputs,
always `float*` for outputs) before dereferencing it, and document each
input's dtype and length explicitly in `prompt.md` -- see the "Kernel ABI
contract" section of the [README](README.md) for the full contract and a
cast-pattern example.

### Tolerance guidance (`rtol`/`atol`)

- Pure elementwise or `min`/`max`-style operations (stencils, per-point
  transforms, nearest-neighbor lookups): tight tolerances are fine, e.g.
  `rtol=1e-4`, `atol=1e-5`, since there's no unbounded accumulation.
- fp32 all-pairs reductions (gravity/potential sums, mean pairwise
  distance, anything summing O(n^2) or O(n) terms in float32 on the
  candidate side): loosen tolerances to account for legitimate
  reduction-order differences between the candidate's kernel and the
  float64-computed reference (e.g. `rtol=1e-3` or looser, tuned per
  problem's `n` and term magnitude). Don't loosen further than necessary
  to pass a numerically reasonable float32 implementation -- the point is
  to tolerate reduction-order variance, not sloppy kernels.

### `baseline.cu`

A correct, straightforward, but *not* deliberately crippled
`minidwarf_solve` implementation. It's the "honest baseline" the `fast_p`
metric measures speedup against -- if it's artificially slow, `fast_p`
becomes meaningless.

### `prompt.md`

The problem statement given to a model attempting the kernel. It must
describe the math, the data layout, and the `minidwarf_solve` ABI in
enough detail to implement the kernel, but:

**NEVER put eval shapes or seeds in `prompt.md`.** The `eval_shapes` in
`spec.yaml` and the seeds used by the grader are held out specifically so
a submitted kernel can't hardcode a shape, special-case an expected
output, or otherwise game correctness without actually solving the
general problem. If you need to illustrate the ABI with an example, use a
shape that is clearly a toy example, distinct from anything in
`eval_shapes`, and say so.

### `solutions/`

At least one working `minidwarf_solve` implementation (e.g.
`solutions/expert_v1.cu`) that a maintainer can point to as evidence the
problem is solvable and the tolerances are achievable by a real kernel.
For the v1 problems (`structured_grids`/`nbody`), the maintainer's
`expert_v1.cu` files are byte-identical to `baseline.cu` -- they
establish solvability, not achievable speedup (see "Known limitations" in
the [README](README.md)). For the v2 `dense`/`sparse` problems, ship a
real, distinct hand-written `expert_v1.cu` even when `baseline.cu` calls
a vendor library (`cublas`/`cusparse`) -- see `problems/dense/sgemm/` and
`problems/sparse/spmv_csr/` for examples of a hand-written expert kernel
paired with a cuBLAS/cuSPARSE baseline. Optimized expert solutions that
actually improve on the baseline (e.g. alternative tiling strategies,
named `expert_v2.cu`, etc.) are welcome and encouraged contributions.

## Adding an expert solution to an existing problem

Add a new file under that problem's `solutions/` directory (e.g.
`solutions/expert_v2.cu`), following the same ABI as `baseline.cu`. Run
`minidwarf run --problem <problem_dir> --kernel <your_solution.cu>` to
confirm it compiles and passes correctness before opening a PR.

## Updating `SPLITS.yaml`

If you add a problem, add its name to exactly one of `valid`/`test` in
`SPLITS.yaml`, keeping the split balanced across dwarfs and across
valid/test. `tests/test_splits.py` will fail CI if a problem is missing
from, or duplicated across, the splits.

## Versioning

MiniDwarf v2 is frozen at the 4-dwarf / 24-problem set described in the
[README](README.md#v2-4-dwarfs-24-problems) (v1 was 2 dwarfs / 12
problems, also frozen). New problems or dwarfs added after this point
should ship as a new named version (v2.1, v3, ...) rather than silently
growing "v2" -- update the version note in the README, not just
`SPLITS.yaml`, when you do. Always report benchmark results against a
named version (e.g. "MiniDwarf v2"), never bare "MiniDwarf".

## License and headers

MiniDwarf is Apache-2.0 licensed (see [LICENSE](LICENSE)). Every `.py`
file you add or modify must start with:

```python
# SPDX-License-Identifier: Apache-2.0
```

(Markdown and YAML files don't need this header.) By submitting a PR you
agree your contribution is licensed under Apache-2.0.

## Running the test suite

```bash
export PATH=/usr/local/cuda/bin:$PATH   # nvcc must be on PATH
python -m pytest -q
```

The full suite compiles CUDA kernels -- including `dense`/`sparse`
problems that link against cuBLAS and cuSPARSE -- so it requires a
working `nvcc` and a CUDA Toolkit (>= 12.8) that ships cuBLAS/cuSPARSE
(both are included in a standard CUDA Toolkit install, no separate
package). Run `python scripts/check_env.py` first if you're unsure your
toolchain is set up correctly.
