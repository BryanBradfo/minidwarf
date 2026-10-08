# SPDX-License-Identifier: Apache-2.0
import numpy as np

CHUNK = 1024  # rows per block: bounds the pairwise temporaries to CHUNK x N float64

def run(inputs, shape):
    pos = inputs[0].astype(np.float64); n = pos.size
    out = np.empty(n)
    for s in range(0, n, CHUNK):
        e = min(s + CHUNK, n)
        d = np.abs(pos[s:e, None] - pos[None, :])
        d[np.arange(e - s), np.arange(s, e)] = np.inf
        out[s:e] = d.min(axis=1)
    return [out.astype(np.float32)]
