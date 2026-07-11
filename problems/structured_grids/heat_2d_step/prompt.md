# heat_2d_step (Structured Grids)

Implement one explicit forward-Euler time step of the 2D heat equation on a
regular `ny * nx` grid stored in row-major (C-order) layout, so that the
flat index of grid point `(i, j)` (with `i` the slowest-varying axis and
`j` the fastest-varying axis) is:

```
idx(i, j) = i * nx + j
```

The diffusion coefficient is `alpha = 0.2`.

For every **interior** point (`0 < i < ny-1`, `0 < j < nx-1`), the output
is:

```
out[i,j] = u[i,j] + alpha * ( u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4*u[i,j] )
```

In flat-index terms, the neighbor offsets from `idx` are `± nx` (the `i`
axis) and `± 1` (the `j` axis).

For every **boundary** point (any of `i == 0`, `j == 0`, `i == ny-1`,
`j == nx-1`), the output is simply a copy of the input value at that point
(no computation, no out-of-bounds access).

All data is single-precision (`float32`).

## Kernel ABI

You must implement a C-linkage entry point with exactly this signature:

```c++
extern "C" void minidwarf_solve(const void* const* inputs,
                                 void* const* outputs,
                                 const long* dims, int n_dims);
```

- Buffers are passed as untyped `void*`; cast each `inputs[k]` you use to
  `const float*` and `outputs[k]` to `float*` before use (all data here is
  `float32`).

- `n_dims == 2`, and `dims = [ny, nx]`.
- `inputs[0]` is the flattened input grid `u`, of length `ny * nx`, stored
  in the C-order layout described above.
- `outputs[0]` is the flattened output grid, of the same length `ny * nx`;
  your kernel must fully populate it (interior updates and boundary copies)
  before returning.
- Both buffers already reside in device memory; do not allocate or free
  them, and do not perform any host/device transfers of your own beyond what
  is needed for any auxiliary buffers you introduce.

## Constraints

- Use only CUDA runtime APIs available via `<cuda_runtime.h>`. No external
  libraries (cuBLAS, cuDNN, Thrust, etc.).
- The kernel must synchronize (or otherwise ensure completion) such that
  `outputs[0]` is fully written and valid by the time `minidwarf_solve`
  returns to the caller.
- Correctness is checked against a NumPy reference implementation with
  `rtol = 1e-4` and `atol = 1e-5`. Any grid shape (not fixed in advance) may
  be used to evaluate your kernel, so do not hardcode grid dimensions,
  assume specific tile sizes divide the grid evenly, or otherwise depend on
  a particular shape.
