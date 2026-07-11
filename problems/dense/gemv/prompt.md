# gemv (Dense)

Implement single-precision general matrix-vector multiplication: `y = A · x`,
with `A` a row-major `float32` matrix and `x`, `y` `float32` vectors.

- `A` has shape `M x N` (row-major, length `M*N`).
- `x` has shape `N` (length `N`).
- `y` has shape `M` (length `M`), and must hold:

```
y[i] = sum_{j in [0, N)} A[i, j] * x[j]
```

for every `i in [0, M)`.

## Kernel ABI

```c++
extern "C" void minidwarf_solve(const void* const* inputs,
                                 void* const* outputs,
                                 const long* dims, int n_dims);
```

- Buffers are passed as untyped `void*`; cast them before use:
  - `inputs[0]` -> `const float*` -> `A`, length `M*N`.
  - `inputs[1]` -> `const float*` -> `x`, length `N`.
  - `outputs[0]` -> `float*` -> `y`, length `M`.
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
  otherwise depend on particular sizes.
