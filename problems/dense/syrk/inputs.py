# SPDX-License-Identifier: Apache-2.0
import numpy as np

def generate(shape, seed):
    m, k = shape
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(m * k).astype(np.float32)
    return [a]
