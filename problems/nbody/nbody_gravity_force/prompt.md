# nbody_gravity_force (N-Body)

Implement the pairwise gravitational-force accumulation for `N` bodies laid
out on a line (1D positions), stored as a flat `float32` array `pos` of
length `N`.

For every body `i`, sum the contribution of every other body `j` (`j != i`):

```
d   = pos[j] - pos[i]
r2  = d * d
out[i] = sum_{j != i}  d / (r2 + eps)^1.5
```

with softening constant `eps = 1e-2` (added to `r2` before raising to the
`1.5` power, to avoid a singularity when two bodies coincide).

The sum for body `i` runs over **all** other bodies `j` in `[0, N)`
(`j != i`); this is an all-pairs (`O(N^2)`) computation, not a
nearest-neighbor or spatial-cutoff scheme.

## Kernel ABI

```c++
extern "C" void minidwarf_solve(const void* const* inputs,
                                 void* const* outputs,
                                 const long* dims, int n_dims);
```

- Buffers are passed as untyped `void*`; cast each `inputs[k]` you use to
  `const float*` and `outputs[k]` to `float*` before use (all data here is
  `float32`).

- `n_dims == 1`, and `dims = [N]`.
- `inputs[0]` is the flattened position array `pos`, of length `N`.
- `outputs[0]` is the flattened output array, of the same length `N`;
  `outputs[0][i]` must hold the accumulated force for body `i`.
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
  implementation with `rtol = 1e-3` and `atol = 1e-4`. Any number of bodies
  `N` (not fixed in advance) may be used to evaluate your kernel, so do not
  hardcode `N` or otherwise depend on a particular size.
