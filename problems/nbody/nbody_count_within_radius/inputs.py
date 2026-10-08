# SPDX-License-Identifier: Apache-2.0
import numpy as np

K = 16  # positions are multiples of 2**-K with |pos| < 64 = 2**6, so each needs <= 6 + 16 = 22 significant bits
        # and every pairwise difference (|d| < 2**7, a multiple of 2**-K) <= 23 bits: both are exact in float32
        # (24-bit significand). |pj - pi| < 3 is then decided identically in float32 and float64.

def generate(shape, seed):
    n = shape[0]
    rng = np.random.default_rng(seed)
    pos = (rng.standard_normal(n).astype(np.float32)) * 5.0
    pos = np.clip(np.round(pos.astype(np.float64) * 2.0**K) / 2.0**K, -63.0, 63.0)
    return [pos.astype(np.float32)]
