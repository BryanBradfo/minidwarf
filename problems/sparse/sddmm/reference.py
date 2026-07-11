# SPDX-License-Identifier: Apache-2.0
import numpy as np

def run(inputs, shape):
    R, C, K, NNZ = shape
    A = inputs[0].astype(np.float64).reshape(R, K)
    B = inputs[1].astype(np.float64).reshape(K, C)
    row_ptr = inputs[2].astype(np.int64)
    col_idx = inputs[3].astype(np.int64)

    out = np.zeros(NNZ, dtype=np.float64)
    for i in range(R):
        start, end = row_ptr[i], row_ptr[i + 1]
        if end > start:
            cols = col_idx[start:end]
            # dot(A[i, :], B[:, c]) for each nonzero column c in this row
            out[start:end] = A[i, :] @ B[:, cols]

    return [out.astype(np.float32)]
