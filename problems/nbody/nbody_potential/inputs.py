# SPDX-License-Identifier: Apache-2.0
import numpy as np

def generate(shape, seed):
    n = shape[0]
    rng = np.random.default_rng(seed)
    pos = (rng.standard_normal(n).astype(np.float32)) * 5.0
    return [pos]
