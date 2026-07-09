# SPDX-License-Identifier: Apache-2.0
import numpy as np

def generate(shape, seed):
    n = shape[0]
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(n, dtype=np.float32)
    b = rng.standard_normal(n, dtype=np.float32)
    return [a, b]
