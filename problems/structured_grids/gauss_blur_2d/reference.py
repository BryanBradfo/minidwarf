# SPDX-License-Identifier: Apache-2.0
import numpy as np

def run(inputs, shape):
    ny, nx = shape
    u = inputs[0].reshape(ny, nx)
    out = u.copy()
    out[1:-1, 1:-1] = (
        1.0 * u[:-2, :-2] + 2.0 * u[:-2, 1:-1] + 1.0 * u[:-2, 2:] +
        2.0 * u[1:-1, :-2] + 4.0 * u[1:-1, 1:-1] + 2.0 * u[1:-1, 2:] +
        1.0 * u[2:, :-2] + 2.0 * u[2:, 1:-1] + 1.0 * u[2:, 2:]
    ) / np.float32(16.0)
    return [out.reshape(-1).astype(np.float32)]
