# SPDX-License-Identifier: Apache-2.0
import numpy as np

def run(inputs, shape):
    R, C, NNZ = shape
    values = inputs[0].astype(np.float64)
    row_ptr = inputs[1].astype(np.int64)
    col_idx = inputs[2].astype(np.int64)
    b = inputs[3].astype(np.float64)
    x = inputs[4].astype(np.float64)

    x_new = np.zeros(R, dtype=np.float64)
    for i in range(R):
        start, end = row_ptr[i], row_ptr[i + 1]
        cols = col_idx[start:end]
        vals = values[start:end]
        diag_pos = np.searchsorted(cols, i)
        diag = vals[diag_pos]
        off_sum = np.dot(vals, x[cols]) - diag * x[i]
        x_new[i] = (b[i] - off_sum) / diag

    return [x_new.astype(np.float32)]
