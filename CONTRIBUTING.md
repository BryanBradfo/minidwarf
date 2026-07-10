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
(`structured_grids`, `nbody`, ...); propose a new dwarf category in an
issue before adding problems under it.

### `spec.yaml`

Required fields (see `minidwarf/spec.py` for the loader): `name`, `dwarf`,
`difficulty`, `rtol`, `atol`, `n_inputs`, `n_outputs`, `eval_shapes`,
`baseline`. `eval_shapes` is a list of shapes (e.g. `[[128,128,128],
[160,160,96]]`) the grader will actually run the candidate and baseline
kernels against -- **these are not disclosed in `prompt.md`** (see the
anti-contamination rule below).

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

**Output-size invariant**: every array in the `outputs` list returned by a
`minidwarf_solve` implementation (and by `reference.py`) must have length
`prod(dims)` for the corresponding output, where `dims` is the shape
passed to the kernel. This holds for every v1 problem, including
reduction-shaped ones like `nbody_potential` (which returns the
per-body potential, i.e. one value per particle, not a single scalar).
Don't return a reduced/summary shape smaller than `prod(dims)` unless a
future problem is explicitly a reduction to a scalar or fixed-size
output -- state that clearly in `spec.yaml`/`prompt.md` if so, since the
driver assumes every output has `prod(dims)` elements.

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
For v1, the maintainer's `expert_v1.cu` files are byte-identical to
`baseline.cu` -- they establish solvability, not achievable speedup (see
"Known limitations" in the [README](README.md)). Optimized expert
solutions that actually improve on the baseline (e.g. alternative tiling
strategies, named `expert_v2.cu`, etc.) are welcome and encouraged
contributions.

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

The full suite compiles CUDA kernels, so it requires a working `nvcc`
(CUDA Toolkit >= 12.8); run `python scripts/check_env.py` first if you're
unsure your toolchain is set up correctly.
