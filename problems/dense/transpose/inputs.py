# SPDX-License-Identifier: Apache-2.0
import numpy as np

def generate(shape, seed):
    m, n = shape
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(m * n).astype(np.float32)
    return [a]
