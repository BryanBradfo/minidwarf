# spmm_csr (Sparse)

Implement sparse-dense matrix multiplication: `Y = A · B`, where `A` is a
sparse `R x C` matrix stored in CSR (Compressed Sparse Row) format, `B` is a
dense `C x D` matrix, and `Y` is a dense `R x D` matrix (all dense matrices
stored row-major).

CSR represents `A` with three parallel arrays:

- `values`: the `NNZ` nonzero entries of `A`, `float32`, stored row by row.
- `row_ptr`: length `R + 1`, `int32`. Row `i`'s nonzeros occupy the index
  range `[row_ptr[i], row_ptr[i + 1])` in `values` / `col_idx`.
  `row_ptr[0] == 0` and `row_ptr[R] == NNZ`.
- `col_idx`: length `NNZ`, `int32`. `col_idx[k]` is the column of the
  nonzero at `values[k]`. Within each row, column indices are sorted in
  increasing order and unique.

The result must satisfy, for every row `i in [0, R)` and column `d in [0, D)`:

```
Y[i, d] = sum over k in [row_ptr[i], row_ptr[i + 1]) of values[k] * B[col_idx[k], d]
```

Rows with no nonzeros produce `Y[i, d] == 0` for all `d`.

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
  - `inputs[3]` -> `const float*` -> `B`, length `C * D`, row-major
    (`B[c * D + d]` is element `(c, d)`).
  - `outputs[0]` -> `float*` -> `Y`, length `R * D`, row-major
    (`Y[i * D + d]` is element `(i, d)`).
- `n_dims == 4`, and `dims = [R, C, NNZ, D]`.
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
  implementation. `R`, `C`, `NNZ`, `D` are not fixed in advance; do not
  hardcode them or otherwise depend on particular sizes.
