# nbody_count_within_radius (N-Body)

Implement a radius-based neighbor count for `N` bodies laid out on a line
(1D positions), stored as a flat `float32` array `pos` of length `N`.

For every body `i`, count how many other bodies `j` (`j != i`) lie strictly
within a fixed radius `R = 3.0`:

```
out[i] = (float) count of { j != i : |pos[i] - pos[j]| < R }
```

The comparison is a **strict** `<` (a body exactly `R` away does not count).
The count for body `i` is taken over **all** other bodies `j` in `[0, N)`
(`j != i`); this is an all-pairs (`O(N^2)`) computation, not a spatial-index
/ cell-list scheme. The result is written as a `float`, but its value is
always a non-negative integer.

## Kernel ABI

```c++
extern "C" void minidwarf_solve(const float* const* inputs,
                                 float* const* outputs,
                                 const long* dims, int n_dims);
```

- `n_dims == 1`, and `dims = [N]`.
- `inputs[0]` is the flattened position array `pos`, of length `N`.
- `outputs[0]` is the flattened output array, of the same length `N`;
  `outputs[0][i]` must hold the neighbor count (as a float) for body `i`.
- Both buffers already reside in device memory; do not allocate or free
  them, and do not perform any host/device transfers of your own beyond what
  is needed for any auxiliary buffers you introduce.

## Constraints

- Use only CUDA runtime APIs available via `<cuda_runtime.h>`. No external
  libraries (cuBLAS, cuDNN, Thrust, etc.).
- The kernel must synchronize (or otherwise ensure completion) such that
  `outputs[0]` is fully written and valid by the time `minidwarf_solve`
  returns to the caller.
- Correctness is checked against a NumPy reference implementation. Because
  the result is integer-valued, the tolerance is `rtol = 0.0`,
  `atol = 0.5` (i.e. your count must match exactly). Any number of bodies
  `N` (not fixed in advance) may be used to evaluate your kernel, so do not
  hardcode `N` or otherwise depend on a particular size.
