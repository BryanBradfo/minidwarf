# scaled_gemm (Dense)

Implement a scaled single-precision matrix multiplication:
`C = alpha * (A · B)`, with `alpha = 0.5`, all matrices stored row-major as
flat `float32` arrays.

- `A` has shape `M x K` (row-major, length `M*K`).
- `B` has shape `K x N` (row-major, length `K*N`).
- `C` has shape `M x N` (row-major, length `M*N`), and must hold:

```
C[i, j] = 0.5 * sum_{p in [0, K)} A[i, p] * B[p, j]
```

for every `i in [0, M)`, `j in [0, N)`.

Note that `A` and `B` are generally different sizes (`M*K` vs `K*N`); do not
assume they are the same length.

## Kernel ABI

```c++
extern "C" void minidwarf_solve(const void* const* inputs,
                                 void* const* outputs,
                                 const long* dims, int n_dims);
```

- Buffers are passed as untyped `void*`; cast them before use:
  - `inputs[0]` -> `const float*` -> `A`, length `M*K`.
  - `inputs[1]` -> `const float*` -> `B`, length `K*N`.
  - `outputs[0]` -> `float*` -> `C`, length `M*N`.
- `n_dims == 3`, and `dims = [M, N, K]`.
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
  implementation. `M`, `N`, `K` are not fixed in advance; do not hardcode
  them or otherwise depend on particular sizes.
