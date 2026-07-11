# csr_row_scale (Sparse)

Implement per-row scaling of the nonzeros of a sparse `R x C` matrix `A`
stored in CSR (Compressed Sparse Row) format: each nonzero is multiplied by
a scale factor associated with its row.

CSR represents `A` with (here, two of) three parallel arrays:

- `values`: the `NNZ` nonzero entries of `A`, `float32`, stored row by row.
- `row_ptr`: length `R + 1`, `int32`. Row `i`'s nonzeros occupy the index
  range `[row_ptr[i], row_ptr[i + 1])` in `values`. `row_ptr[0] == 0` and
  `row_ptr[R] == NNZ`.

(Column indices are irrelevant to this problem and are not provided.)

Let `row(k)` denote the row containing nonzero `k`, i.e. the unique `i` such
that `row_ptr[i] <= k < row_ptr[i + 1]`. The result must satisfy, for every
nonzero index `k in [0, NNZ)`:

```
out[k] = values[k] * scale[row(k)]
```

## Kernel ABI

```c++
extern "C" void minidwarf_solve(const void* const* inputs,
                                 void* const* outputs,
                                 const long* dims, int n_dims);
```

- Buffers are passed as untyped `void*`; cast them before use, in this exact
  order:
  - `inputs[0]` -> `const float*` -> `values`, length `NNZ`.
  - `inputs[1]` -> `const int*` -> `row_ptr`, length `R + 1`.
  - `inputs[2]` -> `const float*` -> `scale`, length `R`.
  - `outputs[0]` -> `float*` -> `out`, length `NNZ`.
- `n_dims == 3`, and `dims = [R, C, NNZ]` (`C` is not needed by this
  problem's computation but is provided for context/shape bookkeeping).
- All buffers already reside in device memory; do not allocate or free them,
  and do not perform any host/device transfers of your own beyond what is
  needed for any auxiliary buffers you introduce.

## Constraints

- Use only CUDA runtime APIs available via `<cuda_runtime.h>` (no cuSPARSE,
  cuBLAS, Thrust, or other external libraries).
- The kernel must synchronize (or otherwise ensure completion) such that
  `outputs[0]` is fully written and valid by the time `minidwarf_solve`
  returns to the caller.
- Correctness is checked against a NumPy (float64 internally) reference
  implementation. `R`, `C`, `NNZ` are not fixed in advance; do not hardcode
  them or otherwise depend on particular sizes.
