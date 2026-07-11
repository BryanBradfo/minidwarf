# SPDX-License-Identifier: Apache-2.0
import numpy as np

def run(inputs, shape):
    R, C, NNZ = shape
    values = inputs[0].astype(np.float64)
    row_ptr = inputs[1].astype(np.int64)
    col_idx = inputs[2].astype(np.int64)
    x = inputs[3].astype(np.float64)

    y = np.zeros(R, dtype=np.float64)
    for i in range(R):
        start, end = row_ptr[i], row_ptr[i + 1]
        if end > start:
            y[i] = np.dot(values[start:end], x[col_idx[start:end]])

    return [y.astype(np.float32)]
