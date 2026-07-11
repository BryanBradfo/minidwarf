# syrk (Dense)

Implement a single-precision symmetric rank-k update: `C = A · A^T`, all
matrices stored row-major as flat `float32` arrays.

- `A` has shape `M x K` (row-major, length `M*K`).
- `C` has shape `M x M` (row-major, length `M*M`), and must hold the FULL
  matrix (both triangles, not just one):

```
C[i, j] = sum_{p in [0, K)} A[i, p] * A[j, p]
```

for every `i in [0, M)`, `j in [0, M)`. Note `C` is symmetric
(`C[i, j] == C[j, i]`), but every entry of the `M x M` output must be
written, not just one triangle.

## Kernel ABI

```c++
extern "C" void minidwarf_solve(const void* const* inputs,
                                 void* const* outputs,
                                 const long* dims, int n_dims);
```

- Buffers are passed as untyped `void*`; cast them before use:
  - `inputs[0]` -> `const float*` -> `A`, length `M*K`.
  - `outputs[0]` -> `float*` -> `C`, length `M*M`.
- `n_dims == 2`, and `dims = [M, K]`.
- All buffers already reside in device memory; do not allocate or free them,
  and do not perform any host/device transfers of your own beyond what is
  needed for any auxiliary buffers you introduce.

## Constraints

- Use only CUDA runtime APIs available via `<cuda_runtime.h>` (no cuBLAS,
  cuDNN, Thrust, or other external libraries).
- The kernel must synchronize (or otherwise ensure completion) such that
  `outputs[0]` is fully written and valid by the time `minidwarf_solve`
  returns to the caller.
- Correctness is checked against a NumPy (float64 internally) reference
  implementation. `M`, `K` are not fixed in advance; do not hardcode them or
  otherwise depend on particular sizes.
