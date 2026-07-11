# SPDX-License-Identifier: Apache-2.0
import numpy as np

def run(inputs, shape):
    R, C, NNZ = shape
    values = inputs[0].astype(np.float64)
    row_ptr = inputs[1].astype(np.int64)
    col_idx = inputs[2].astype(np.int64)
    x = inputs[3].astype(np.float64)

    y = np.zeros(C, dtype=np.float64)
    for i in range(R):
        start, end = row_ptr[i], row_ptr[i + 1]
        if end > start:
            np.add.at(y, col_idx[start:end], values[start:end] * x[i])

    return [y.astype(np.float32)]
