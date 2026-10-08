# SPDX-License-Identifier: Apache-2.0
import numpy as np

CHUNK = 1024  # rows per block: bounds the pairwise temporaries to CHUNK x N float64

def run(inputs, shape):
    n = shape[0]
    pos = inputs[0].astype(np.float64)
    out = np.empty(n)
    for s in range(0, n, CHUNK):
        e = min(s + CHUNK, n)
        d = np.abs(pos[s:e, None] - pos[None, :])  # includes j == i, which is 0
        out[s:e] = d.sum(axis=1) / n
    return [out.astype(np.float32)]
