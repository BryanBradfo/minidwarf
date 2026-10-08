# SPDX-License-Identifier: Apache-2.0
import numpy as np

EPS = 1e-2
CHUNK = 1024  # rows per block: bounds the pairwise temporaries to CHUNK x N float64

def run(inputs, shape):
    pos = inputs[0].astype(np.float64); n = pos.size
    out = np.empty(n)
    for s in range(0, n, CHUNK):
        e = min(s + CHUNK, n)
        d = pos[None, :] - pos[s:e, None]  # d[i, j] = pos[j] - pos[i]
        contrib = d / np.power(d * d + EPS, 1.5)
        contrib[np.arange(e - s), np.arange(s, e)] = 0.0
        out[s:e] = contrib.sum(axis=1)
    return [out.astype(np.float32)]
