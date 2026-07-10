# SPDX-License-Identifier: Apache-2.0
import numpy as np

def run(inputs, shape):
    pos = inputs[0].astype(np.float64)
    d = np.abs(pos[:, None] - pos[None, :])
    np.fill_diagonal(d, np.inf)
    out = d.min(axis=1)
    return [out.astype(np.float32)]
