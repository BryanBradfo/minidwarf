# SPDX-License-Identifier: Apache-2.0
import numpy as np

RADIUS = 3.0

def run(inputs, shape):
    pos = inputs[0].astype(np.float64)
    d = np.abs(pos[:, None] - pos[None, :])
    within = d < RADIUS
    np.fill_diagonal(within, False)
    out = within.sum(axis=1).astype(np.float64)
    return [out.astype(np.float32)]
