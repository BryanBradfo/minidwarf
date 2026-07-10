# SPDX-License-Identifier: Apache-2.0
import numpy as np

C2 = np.float32(0.25)

def run(inputs, shape):
    ny, nx = shape
    u_prev = inputs[0].reshape(ny, nx)
    u_curr = inputs[1].reshape(ny, nx)
    out = u_curr.copy()
    out[1:-1, 1:-1] = (
        np.float32(2.0) * u_curr[1:-1, 1:-1] - u_prev[1:-1, 1:-1] +
        C2 * (
            u_curr[2:, 1:-1] + u_curr[:-2, 1:-1] +
            u_curr[1:-1, 2:] + u_curr[1:-1, :-2] -
            np.float32(4.0) * u_curr[1:-1, 1:-1]))
    return [out.reshape(-1).astype(np.float32)]
