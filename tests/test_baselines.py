# SPDX-License-Identifier: Apache-2.0
import pytest
from minidwarf.baselines import link_flags

def test_known_baselines():
    assert link_flags("author_kernel") == []
    assert link_flags("cublas") == ["-lcublas"]
    assert link_flags("cusparse") == ["-lcusparse"]

def test_unknown_baseline_raises():
    with pytest.raises(ValueError):
        link_flags("nope")
