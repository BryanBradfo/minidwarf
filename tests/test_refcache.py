# SPDX-License-Identifier: Apache-2.0
import shutil
from pathlib import Path
import numpy as np
from minidwarf.refcache import cached_case, case_key

FIX = Path(__file__).parent / "fixtures/smoke/vector_add"

def test_cached_case_matches_direct_and_hits_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("MINIDWARF_CACHE", str(tmp_path))
    ins, outs = cached_case(FIX, [1000], 7)
    np.testing.assert_array_equal(outs[0], ins[0] + ins[1])
    assert len(list(tmp_path.rglob("*.npz"))) == 1
    ins2, outs2 = cached_case(FIX, [1000], 7)
    np.testing.assert_array_equal(outs2[0], outs[0])
    assert ins2[0].dtype == np.float32 and outs2[0].dtype == np.float32

def test_key_changes_with_seed_shape_and_source(tmp_path):
    prob = tmp_path / "vector_add"; shutil.copytree(FIX, prob)
    k = case_key(prob, [10], 1)
    assert k != case_key(prob, [10], 2) and k != case_key(prob, [11], 1)
    (prob / "reference.py").write_text((prob / "reference.py").read_text() + "\n# edited\n")
    assert case_key(prob, [10], 1) != k

def test_corrupt_cache_file_is_recomputed(tmp_path, monkeypatch):
    monkeypatch.setenv("MINIDWARF_CACHE", str(tmp_path))
    path = tmp_path / "cases" / "vector_add" / f"{case_key(FIX, [50], 3)}.npz"
    path.parent.mkdir(parents=True); path.write_bytes(b"truncated")
    ins, outs = cached_case(FIX, [50], 3)
    np.testing.assert_array_equal(outs[0], ins[0] + ins[1])

def test_failed_save_leaves_no_temp_file(tmp_path, monkeypatch):
    import pytest
    import minidwarf.refcache as rc
    monkeypatch.setenv("MINIDWARF_CACHE", str(tmp_path))
    def boom(*a, **k): raise OSError("disk full")
    monkeypatch.setattr(rc.np, "savez", boom)
    with pytest.raises(OSError):
        cached_case(FIX, [10], 1)
    assert list(tmp_path.rglob("*.tmp.npz")) == []
