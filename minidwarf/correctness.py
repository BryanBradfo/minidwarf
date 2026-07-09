# SPDX-License-Identifier: Apache-2.0
import numpy as np


def check_correct(actual, expected, rtol, atol) -> bool:
    if len(actual) != len(expected):
        return False
    return all(np.allclose(a, e, rtol=rtol, atol=atol, equal_nan=False)
               for a, e in zip(actual, expected))
