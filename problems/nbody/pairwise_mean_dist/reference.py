# SPDX-License-Identifier: Apache-2.0
import numpy as np

def run(inputs, shape):
    n = shape[0]
    pos = inputs[0].astype(np.float64)
    d = np.abs(pos[:, None] - pos[None, :])  # includes j == i, which is 0
    out = d.sum(axis=1) / n
    return [out.astype(np.float32)]
