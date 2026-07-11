# SPDX-License-Identifier: Apache-2.0
import numpy as np

def generate(shape, seed):
    R, C, NNZ = shape
    if NNZ > R * C:
        raise ValueError(f"NNZ={NNZ} exceeds R*C={R * C}; cannot build a valid CSR matrix")
    rng = np.random.default_rng(seed)

    # Distribute NNZ nonzeros across R rows as evenly as possible, then cap
    # each row's budget at C (a row cannot have more than C distinct columns)
    # and push any surplus from capped rows onto rows with spare capacity, so
    # the total is always exactly NNZ.
    base = NNZ // R
    rem = NNZ - base * R
    counts = np.full(R, base, dtype=np.int64)
    counts[:rem] += 1

    overflow = np.maximum(counts - C, 0)
    counts = np.minimum(counts, C)
    deficit = int(overflow.sum())
    i = 0
    while deficit > 0:
        spare = C - counts[i]
        if spare > 0:
            add = min(int(spare), deficit)
            counts[i] += add
            deficit -= add
        i = (i + 1) % R

    row_ptr = np.zeros(R + 1, dtype=np.int64)
    np.cumsum(counts, out=row_ptr[1:])
    assert row_ptr[R] == NNZ

    col_idx = np.empty(NNZ, dtype=np.int64)
    for r in range(R):
        start, end = row_ptr[r], row_ptr[r + 1]
        k = end - start
        if k > 0:
            cols = rng.choice(C, size=k, replace=False)
            cols.sort()
            col_idx[start:end] = cols

    values = rng.standard_normal(NNZ).astype(np.float32)
    x = rng.standard_normal(C).astype(np.float32)

    return [values, row_ptr.astype(np.int32), col_idx.astype(np.int32), x]
