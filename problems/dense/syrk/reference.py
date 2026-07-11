# SPDX-License-Identifier: Apache-2.0
import numpy as np

def run(inputs, shape):
    m, k = shape
    a = inputs[0].reshape(m, k).astype(np.float64)
    c = (a @ a.T).astype(np.float32)
    return [c.reshape(-1)]
