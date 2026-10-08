# SPDX-License-Identifier: Apache-2.0
import numpy as np

RADIUS = 3.0
CHUNK = 1024  # rows per block: bounds the pairwise temporaries to CHUNK x N float64

def run(inputs, shape):
    pos = inputs[0].astype(np.float64); n = pos.size
    out = np.empty(n)
    for s in range(0, n, CHUNK):
        e = min(s + CHUNK, n)
        within = np.abs(pos[s:e, None] - pos[None, :]) < RADIUS
        within[np.arange(e - s), np.arange(s, e)] = False
        out[s:e] = within.sum(axis=1).astype(np.float64)
    return [out.astype(np.float32)]
