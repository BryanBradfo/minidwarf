# sobel_2d (Structured Grids)

Implement the Sobel edge-detection operator on a regular `ny * nx` grid
stored in row-major (C-order) layout, so that the flat index of grid point
`(i, j)` (with `i` the slowest-varying axis and `j` the fastest-varying
axis) is:

```
idx(i, j) = i * nx + j
```

The Sobel gradient kernels are:

```
Gx = [[-1, 0, 1],       Gy = [[-1, -2, -1],
      [-2, 0, 2],             [ 0,  0,  0],
      [-1, 0, 1]]             [ 1,  2,  1]]
```

(`Gy` is the transpose of `Gx`.)

For every **interior** point (`0 < i < ny-1`, `0 < j < nx-1`), compute the
two directional gradients by correlating the 3x3 neighborhood centered at
`(i, j)` with `Gx` and `Gy`, then the output is the gradient magnitude:

```
gx = -u[i-1,j-1] + u[i-1,j+1] - 2*u[i,j-1] + 2*u[i,j+1] - u[i+1,j-1] + u[i+1,j+1]
gy = -u[i-1,j-1] - 2*u[i-1,j] - u[i-1,j+1] + u[i+1,j-1] + 2*u[i+1,j] + u[i+1,j+1]
out[i,j] = sqrt(gx*gx + gy*gy)
```

For every **boundary** point (any of `i == 0`, `j == 0`, `i == ny-1`,
`j == nx-1`), the output is `0.0` (no computation, no out-of-bounds
access).

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
  your kernel must fully populate it (interior gradient magnitudes and
  boundary zeros) before returning.
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
  `rtol = 1e-3` and `atol = 1e-4`. Any grid shape (not fixed in advance) may
  be used to evaluate your kernel, so do not hardcode grid dimensions,
  assume specific tile sizes divide the grid evenly, or otherwise depend on
  a particular shape.
