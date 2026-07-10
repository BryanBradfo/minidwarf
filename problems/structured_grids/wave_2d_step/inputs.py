# SPDX-License-Identifier: Apache-2.0
import numpy as np

def generate(shape, seed):
    ny, nx = shape
    rng = np.random.default_rng(seed)
    u_prev = rng.standard_normal(ny * nx, dtype=np.float32)
    u_curr = rng.standard_normal(ny * nx, dtype=np.float32)
    return [u_prev, u_curr]
