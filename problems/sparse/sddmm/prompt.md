# sddmm (Sparse)

Implement Sampled Dense-Dense Matrix Multiplication (SDDMM): given a dense
`R x K` matrix `A`, a dense `K x C` matrix `B`, and a sparsity pattern over
an `R x C` matrix stored in CSR (Compressed Sparse Row) format, compute the
dot product of `A`'s row and `B`'s column only at the positions where the
sparsity pattern has a nonzero.

CSR represents the sparsity pattern with two of the usual three parallel
arrays (no `values` array is needed as input; the problem produces one):

- `row_ptr`: length `R + 1`, `int32`. Row `i`'s nonzeros occupy the index
  range `[row_ptr[i], row_ptr[i + 1])` in `col_idx` / the output.
  `row_ptr[0] == 0` and `row_ptr[R] == NNZ`.
- `col_idx`: length `NNZ`, `int32`. `col_idx[e]` is the column of nonzero
  `e`. Within each row, column indices are sorted in increasing order and
  unique.

`A` and `B` are dense, row-major: `A[i * K + k]` is element `(i, k)` of the
`R x K` matrix; `B[k * C + j]` is element `(k, j)` of the `K x C` matrix.

Let `row(e)` denote the row containing nonzero `e` (the unique `i` such that
`row_ptr[i] <= e < row_ptr[i + 1]`), and `col_idx[e]` its column. The result
must satisfy, for every nonzero index `e in [0, NNZ)`:

```
out_values[e] = sum over k in [0, K) of A[row(e) * K + k] * B[k * C + col_idx[e]]
```

## Kernel ABI

```c++
extern "C" void minidwarf_solve(const void* const* inputs,
                                 void* const* outputs,
                                 const long* dims, int n_dims);
```

- Buffers are passed as untyped `void*`; cast them before use, in this exact
  order:
  - `inputs[0]` -> `const float*` -> `A`, length `R * K`, row-major.
  - `inputs[1]` -> `const float*` -> `B`, length `K * C`, row-major.
  - `inputs[2]` -> `const int*` -> `row_ptr`, length `R + 1`.
  - `inputs[3]` -> `const int*` -> `col_idx`, length `NNZ`.
  - `outputs[0]` -> `float*` -> `out_values`, length `NNZ`.
- `n_dims == 4`, and `dims = [R, C, K, NNZ]`.
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
  implementation. `R`, `C`, `K`, `NNZ` are not fixed in advance; do not
  hardcode them or otherwise depend on particular sizes.
