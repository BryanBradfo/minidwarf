# SPDX-License-Identifier: Apache-2.0
import numpy as np

def generate(shape, seed):
    ny, nx = shape
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(ny * nx, dtype=np.float32)]
