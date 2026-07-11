# jacobi_sparse (Sparse)

Implement one iteration of the Jacobi method for solving `A x = b`, where
`A` is a square (`R == C`) sparse `R x R` matrix stored in CSR (Compressed
Sparse Row) format that is guaranteed to have a stored diagonal entry in
every row, `b` is a dense right-hand-side vector of length `R`, and `x` is
the current dense iterate of length `R`.

CSR represents `A` with three parallel arrays:

- `values`: the `NNZ` nonzero entries of `A`, `float32`, stored row by row.
- `row_ptr`: length `R + 1`, `int32`. Row `i`'s nonzeros occupy the index
  range `[row_ptr[i], row_ptr[i + 1])` in `values` / `col_idx`.
  `row_ptr[0] == 0` and `row_ptr[R] == NNZ`.
- `col_idx`: length `NNZ`, `int32`. `col_idx[k]` is the column of the
  nonzero at `values[k]`. Within each row, column indices are sorted in
  increasing order and unique, and every row `i` contains column `i` (the
  diagonal entry `A[i, i]`), which is nonzero.

The result must satisfy, for every row `i in [0, R)`:

```
x_new[i] = (b[i] - sum_{j != i, A[i,j] != 0} A[i,j] * x[j]) / A[i,i]
```

Equivalently: sum `values[k] * x[col_idx[k]]` over all nonzeros `k` in row
`i` where `col_idx[k] != i`, subtract that from `b[i]`, then divide by the
diagonal value `A[i, i]`.

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
  - `inputs[3]` -> `const float*` -> `b`, length `R`.
  - `inputs[4]` -> `const float*` -> `x`, length `R`.
  - `outputs[0]` -> `float*` -> `x_new`, length `R`.
- `n_dims == 3`, and `dims = [R, C, NNZ]` (with `R == C`, since `A` is
  square).
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
  implementation. `R`, `NNZ` are not fixed in advance; do not hardcode them
  or otherwise depend on particular sizes.
