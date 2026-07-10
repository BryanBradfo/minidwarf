# SPDX-License-Identifier: Apache-2.0
import numpy as np

def run(inputs, shape):
    nx, ny, nz = shape
    u = inputs[0].reshape(nx, ny, nz)
    out = u.copy()
    out[1:-1, 1:-1, 1:-1] = (
        u[2:, 1:-1, 1:-1] + u[:-2, 1:-1, 1:-1] +
        u[1:-1, 2:, 1:-1] + u[1:-1, :-2, 1:-1] +
        u[1:-1, 1:-1, 2:] + u[1:-1, 1:-1, :-2] +
        u[1:-1, 1:-1, 1:-1]) / np.float32(7.0)
    return [out.reshape(-1).astype(np.float32)]
