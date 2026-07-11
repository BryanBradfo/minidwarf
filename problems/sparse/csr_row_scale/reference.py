# SPDX-License-Identifier: Apache-2.0
import numpy as np

def run(inputs, shape):
    R, C, NNZ = shape
    values = inputs[0].astype(np.float64)
    row_ptr = inputs[1].astype(np.int64)
    scale = inputs[2].astype(np.float64)

    out = np.zeros(NNZ, dtype=np.float64)
    for i in range(R):
        start, end = row_ptr[i], row_ptr[i + 1]
        if end > start:
            out[start:end] = values[start:end] * scale[i]

    return [out.astype(np.float32)]
