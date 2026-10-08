# SPDX-License-Identifier: Apache-2.0
import pytest
from minidwarf.baselines import link_flags, lib_flags

def test_known_baselines():
    assert link_flags("author_kernel") == []
    assert link_flags("cublas") == ["-lcublas"]
    assert link_flags("cusparse") == ["-lcusparse"]

def test_unknown_baseline_raises():
    with pytest.raises(ValueError):
        link_flags("nope")

def test_lib_flags():
    assert lib_flags([]) == []
    assert lib_flags(["curand"]) == ["-lcurand"]
    with pytest.raises(ValueError):
        lib_flags(["mkl"])

def test_baselines_compute_in_float32():
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "problems"
    # FP64 runs at ~1/64 of FP32 on consumer GPUs: a double baseline would score the precision switch, not the kernel.
    bad = [p.parent.name for p in sorted(root.glob("*/*/baseline.cu")) if re.search(r"\bdouble\b", p.read_text())]
    assert bad == []
