# knn_distance_1d (N-Body)

Implement nearest-neighbor distance computation for `N` bodies laid out on a
line (1D positions), stored as a flat `float32` array `pos` of length `N`.

For every body `i`, find the distance to its nearest other body `j`
(`j != i`):

```
out[i] = min_{j != i}  |pos[i] - pos[j]|
```

The minimum for body `i` is taken over **all** other bodies `j` in
`[0, N)` (`j != i`); this is an all-pairs (`O(N^2)`) computation, not a
spatial-index / tree-based nearest-neighbor search.

## Kernel ABI

```c++
extern "C" void minidwarf_solve(const float* const* inputs,
                                 float* const* outputs,
                                 const long* dims, int n_dims);
```

- `n_dims == 1`, and `dims = [N]`.
- `inputs[0]` is the flattened position array `pos`, of length `N`.
- `outputs[0]` is the flattened output array, of the same length `N`;
  `outputs[0][i]` must hold the nearest-neighbor distance for body `i`.
- Both buffers already reside in device memory; do not allocate or free
  them, and do not perform any host/device transfers of your own beyond what
  is needed for any auxiliary buffers you introduce.

## Constraints

- Use only CUDA runtime APIs available via `<cuda_runtime.h>`. No external
  libraries (cuBLAS, cuDNN, Thrust, etc.).
- The kernel must synchronize (or otherwise ensure completion) such that
  `outputs[0]` is fully written and valid by the time `minidwarf_solve`
  returns to the caller.
- Correctness is checked against a NumPy (float64 internally) reference
  implementation with `rtol = 1e-5` and `atol = 1e-6` (this problem has no
  summation, so precision requirements are tight). Any number of bodies `N`
  (not fixed in advance) may be used to evaluate your kernel, so do not
  hardcode `N` or otherwise depend on a particular size.
