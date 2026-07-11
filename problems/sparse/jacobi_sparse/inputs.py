# SPDX-License-Identifier: Apache-2.0
import numpy as np

def generate(shape, seed):
    R, C, NNZ = shape
    if R != C:
        raise ValueError(f"jacobi_sparse requires a square matrix, got R={R}, C={C}")
    if NNZ < R:
        raise ValueError(f"NNZ={NNZ} must be >= R={R} so every row can hold a diagonal entry")
    if NNZ > R * C:
        raise ValueError(f"NNZ={NNZ} exceeds R*C={R * C}; cannot build a valid CSR matrix")
    rng = np.random.default_rng(seed)

    # Reserve one nonzero per row for the (guaranteed) diagonal entry, then
    # distribute the remaining NNZ - R nonzeros across rows as evenly as
    # possible, cap each row's budget at C, and push any surplus from capped
    # rows onto rows with spare capacity, so the total is always exactly
    # NNZ and every row keeps at least its diagonal slot.
    extra = NNZ - R
    base = extra // R
    rem = extra - base * R
    counts = np.full(R, 1 + base, dtype=np.int64)
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
    values = np.empty(NNZ, dtype=np.float32)
    for r in range(R):
        start, end = row_ptr[r], row_ptr[r + 1]
        k = end - start
        cols = rng.choice(C, size=k, replace=False)
        if r not in cols:
            cols[0] = r  # force the diagonal column into this row's set
        cols.sort()
        col_idx[start:end] = cols

        vals = rng.standard_normal(k)
        diag_pos = int(np.searchsorted(cols, r))
        # Make the diagonal dominant (and safely nonzero) so a single Jacobi
        # step divides by a well-conditioned value.
        vals[diag_pos] += R
        values[start:end] = vals.astype(np.float32)

    b = rng.standard_normal(R).astype(np.float32)
    x = rng.standard_normal(R).astype(np.float32)

    return [values, row_ptr.astype(np.int32), col_idx.astype(np.int32), b, x]
