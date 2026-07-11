# transpose (Dense)

Implement a single-precision matrix transpose: `B = A^T`, both matrices
stored row-major as flat `float32` arrays.

- `A` has shape `M x N` (row-major, length `M*N`).
- `B` has shape `N x M` (row-major, length `N*M`), and must hold:

```
B[j, i] = A[i, j]
```

for every `i in [0, M)`, `j in [0, N)`.

## Kernel ABI

```c++
extern "C" void minidwarf_solve(const void* const* inputs,
                                 void* const* outputs,
                                 const long* dims, int n_dims);
```

- Buffers are passed as untyped `void*`; cast them before use:
  - `inputs[0]` -> `const float*` -> `A`, length `M*N`.
  - `outputs[0]` -> `float*` -> `B`, length `N*M`.
- `n_dims == 2`, and `dims = [M, N]`.
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
  implementation. `M`, `N` are not fixed in advance; do not hardcode them or
  otherwise depend on particular sizes. `M` and `N` are not necessarily
  equal.
