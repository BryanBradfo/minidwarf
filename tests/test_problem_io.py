# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import numpy as np
from minidwarf.problem_io import load_module_fn, write_arrays, read_arrays

FIX = Path(__file__).parent / "fixtures/smoke/vector_add"

def test_generate_and_reference_roundtrip():
    gen = load_module_fn(FIX, "inputs.py", "generate")
    ref = load_module_fn(FIX, "reference.py", "run")
    ins = gen([16], seed=0)
    assert len(ins) == 2 and ins[0].dtype == np.float32
    outs = ref(ins, [16])
    np.testing.assert_allclose(outs[0], ins[0] + ins[1], rtol=1e-6)

def test_write_read_mixed_dtype_roundtrip(tmp_path):
    f = np.arange(6, dtype=np.float32)
    i = np.array([0, 3, 5, 9], dtype=np.int32)
    p = tmp_path / "mixed.bin"
    write_arrays(p, [f, i])
    back = read_arrays(p, [((6,), np.float32), ((4,), np.int32)])
    np.testing.assert_array_equal(back[0], f); assert back[0].dtype == np.float32
    np.testing.assert_array_equal(back[1], i); assert back[1].dtype == np.int32

def test_array_serialization_roundtrip(tmp_path):
    arrs = [np.arange(8, dtype=np.float32), np.ones(4, dtype=np.float32)]
    p = tmp_path / "buf.bin"
    write_arrays(p, arrs)
    back = read_arrays(p, [((8,), np.float32), ((4,), np.float32)])
    np.testing.assert_array_equal(back[0], arrs[0])
    np.testing.assert_array_equal(back[1], arrs[1])

def test_write_arrays_rejects_unsupported_dtype(tmp_path):
    p = tmp_path / "x.bin"
    try:
        write_arrays(p, [np.arange(4, dtype=np.int64)])
        assert False, "Expected ValueError for int64 dtype"
    except ValueError as e:
        assert "unsupported dtype" in str(e)
