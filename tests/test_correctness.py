# SPDX-License-Identifier: Apache-2.0
import numpy as np
from minidwarf.correctness import check_correct


def test_close_passes():
    a = np.ones(4, np.float32)
    b = a + 1e-7
    assert check_correct([a], [b], rtol=1e-5, atol=1e-6)


def test_far_fails():
    a = np.ones(4, np.float32)
    b = a + 1.0
    assert not check_correct([a], [b], rtol=1e-5, atol=1e-6)
