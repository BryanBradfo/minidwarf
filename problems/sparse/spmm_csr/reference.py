# SPDX-License-Identifier: Apache-2.0
import numpy as np

def run(inputs, shape):
    R, C, NNZ, D = shape
    values = inputs[0].astype(np.float64)
    row_ptr = inputs[1].astype(np.int64)
    col_idx = inputs[2].astype(np.int64)
    B = inputs[3].astype(np.float64).reshape(C, D)

    Y = np.zeros((R, D), dtype=np.float64)
    for i in range(R):
        start, end = row_ptr[i], row_ptr[i + 1]
        if end > start:
            Y[i, :] = values[start:end] @ B[col_idx[start:end], :]

    return [Y.reshape(-1).astype(np.float32)]
