# vector_add (Structured Grids, smoke test)

Implement element-wise vector addition of two single-precision (`float32`)
vectors `a` and `b`, each of length `n`:

```
out[i] = a[i] + b[i]
```

for every `i in [0, n)`.

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

- `n_dims == 1`, and `dims = [n]`.
- `inputs[0]` is vector `a`, of length `n`.
- `inputs[1]` is vector `b`, of length `n`.
- `outputs[0]` is the flattened output vector, of length `n`; your kernel
  must fully populate it before returning.
- All buffers already reside in device memory; do not allocate or free
  them, and do not perform any host/device transfers of your own beyond what
  is needed for any auxiliary buffers you introduce.

## Constraints

- Use only CUDA runtime APIs available via `<cuda_runtime.h>`. No external
  libraries (cuBLAS, cuDNN, Thrust, etc.).
- The kernel must synchronize (or otherwise ensure completion) such that
  `outputs[0]` is fully written and valid by the time `minidwarf_solve`
  returns to the caller.
- Correctness is checked against a NumPy reference implementation. Any
  vector length (not fixed in advance) may be used to evaluate your kernel,
  so do not hardcode the length.
