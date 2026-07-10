# SPDX-License-Identifier: Apache-2.0
import numpy as np

def generate(shape, seed):
    nx, ny, nz = shape
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(nx * ny * nz, dtype=np.float32)]
