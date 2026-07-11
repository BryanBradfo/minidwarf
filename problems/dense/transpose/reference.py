# SPDX-License-Identifier: Apache-2.0
import numpy as np

def run(inputs, shape):
    m, n = shape
    a = inputs[0].reshape(m, n).astype(np.float64)
    b = a.T.astype(np.float32)
    return [b.reshape(-1)]
