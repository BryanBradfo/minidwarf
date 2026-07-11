# SPDX-License-Identifier: Apache-2.0
import numpy as np

def run(inputs, shape):
    m, n, k = shape
    a = inputs[0].reshape(m, k).astype(np.float64)
    b = inputs[1].reshape(k, n).astype(np.float64)
    bias = inputs[2].reshape(m, n).astype(np.float64)
    c = (a @ b + bias).astype(np.float32)
    return [c.reshape(-1)]
