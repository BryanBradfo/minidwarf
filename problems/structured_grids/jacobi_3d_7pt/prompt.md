# jacobi_3d_7pt (Structured Grids)

Implement one Jacobi iteration of the 3D 7-point heat stencil on a regular
`nx * ny * nz` grid stored in row-major (C-order) layout, so that the flat
index of grid point `(i, j, k)` (with `i` the slowest-varying axis and `k`
the fastest-varying axis) is:

```
idx(i, j, k) = (i * ny + j) * nz + k
```

For every **interior** point (`0 < i < nx-1`, `0 < j < ny-1`, `0 < k < nz-1`),
the output is the average of the point's 6 face-neighbors plus itself:

```
u_new[i,j,k] = ( u[i-1,j,k] + u[i+1,j,k]
               + u[i,j-1,k] + u[i,j+1,k]
               + u[i,j,k-1] + u[i,j,k+1]
               + u[i,j,k] ) / 7
```

In flat-index terms, the neighbor offsets from `idx` are `± ny*nz` (the `i`
axis), `± nz` (the `j` axis), and `± 1` (the `k` axis).

For every **boundary** point (any of `i == 0`, `j == 0`, `k == 0`,
`i == nx-1`, `j == ny-1`, `k == nz-1`), the output is simply a copy of the
input value at that point (no averaging, no out-of-bounds access).

All data is single-precision (`float32`).

## Kernel ABI

You must implement a C-linkage entry point with exactly this signature:

```c++
extern "C" void minidwarf_solve(const float* const* inputs,
                                 float* const* outputs,
                                 const long* dims, int n_dims);
```

- `n_dims == 3`, and `dims = [nx, ny, nz]`.
- `inputs[0]` is the flattened input grid `u`, of length `nx * ny * nz`,
  stored in the C-order layout described above.
- `outputs[0]` is the flattened output grid `u_new`, of the same length
  `nx * ny * nz`; your kernel must fully populate it (interior averages and
  boundary copies) before returning.
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
