# spmv_transpose (Sparse)

Implement transposed sparse matrix-vector multiplication: `y = A^T · x`,
where `A` is a sparse `R x C` matrix stored in CSR (Compressed Sparse Row)
format, `x` is a dense vector of length `R`, and `y` is a dense vector of
length `C`.

CSR represents `A` with three parallel arrays:

- `values`: the `NNZ` nonzero entries of `A`, `float32`, stored row by row.
- `row_ptr`: length `R + 1`, `int32`. Row `i`'s nonzeros occupy the index
  range `[row_ptr[i], row_ptr[i + 1])` in `values` / `col_idx`.
  `row_ptr[0] == 0` and `row_ptr[R] == NNZ`.
- `col_idx`: length `NNZ`, `int32`. `col_idx[k]` is the column of the
  nonzero at `values[k]`. Within each row, column indices are sorted in
  increasing order and unique.

The result must satisfy, for every column `j in [0, C)`:

```
y[j] = sum over all nonzeros k with col_idx[k] == j of values[k] * x[row(k)]
```

where `row(k)` is the row containing nonzero `k`. Equivalently: for each row
`i` and each nonzero `k` in that row, `y[col_idx[k]]` accumulates
`values[k] * x[i]`. Columns with no nonzeros produce `y[j] == 0`.

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
  - `inputs[2]` -> `const int*` -> `col_idx`, length `NNZ`.
  - `inputs[3]` -> `const float*` -> `x`, length `R`.
  - `outputs[0]` -> `float*` -> `y`, length `C`.
- `n_dims == 3`, and `dims = [R, C, NNZ]`.
- All buffers already reside in device memory; do not allocate or free them,
  and do not perform any host/device transfers of your own beyond what is
  needed for any auxiliary buffers you introduce.
- `outputs[0]` is not pre-zeroed by the caller; your kernel is responsible
  for producing the correct final values, including for columns with no
  nonzeros.

## Constraints

- Use only CUDA runtime APIs available via `<cuda_runtime.h>` (no cuSPARSE,
  cuBLAS, Thrust, or other external libraries).
- The kernel must synchronize (or otherwise ensure completion) such that
  `outputs[0]` is fully written and valid by the time `minidwarf_solve`
  returns to the caller.
- Correctness is checked against a NumPy (float64 internally) reference
  implementation. `R`, `C`, `NNZ` are not fixed in advance; do not hardcode
  them or otherwise depend on particular sizes.
