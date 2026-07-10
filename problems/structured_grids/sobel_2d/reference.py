# SPDX-License-Identifier: Apache-2.0
import numpy as np

def run(inputs, shape):
    ny, nx = shape
    u = inputs[0].reshape(ny, nx)
    out = np.zeros_like(u)
    gx = (
        -u[:-2, :-2] + u[:-2, 2:] +
        -2.0 * u[1:-1, :-2] + 2.0 * u[1:-1, 2:] +
        -u[2:, :-2] + u[2:, 2:]
    )
    gy = (
        -u[:-2, :-2] - 2.0 * u[:-2, 1:-1] - u[:-2, 2:] +
        u[2:, :-2] + 2.0 * u[2:, 1:-1] + u[2:, 2:]
    )
    out[1:-1, 1:-1] = np.sqrt(gx * gx + gy * gy)
    return [out.reshape(-1).astype(np.float32)]
