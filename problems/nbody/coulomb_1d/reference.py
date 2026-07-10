# SPDX-License-Identifier: Apache-2.0
import numpy as np

EPS = 1e-2

def run(inputs, shape):
    pos = inputs[0].astype(np.float64)
    d = pos[:, None] - pos[None, :]  # d[i, j] = pos[i] - pos[j]
    r2 = d * d
    denom = np.power(r2 + EPS, 1.5)
    contrib = d / denom
    np.fill_diagonal(contrib, 0.0)
    out = contrib.sum(axis=1)
    return [out.astype(np.float32)]
