# SPDX-License-Identifier: Apache-2.0
import numpy as np

def run(inputs, shape):
    m, n = shape
    a = inputs[0].reshape(m, n).astype(np.float64)
    x = inputs[1].astype(np.float64)
    y = (a @ x).astype(np.float32)
    return [y.reshape(-1)]
