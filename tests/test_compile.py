# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import pytest
from minidwarf.compile import compile_binary, CompileError

FIX = Path(__file__).parent / "fixtures/smoke/vector_add"

def test_compile_expert_succeeds(tmp_path):
    exe = compile_binary(FIX / "solutions/expert_v1.cu", tmp_path)
    assert exe.exists()

def test_compile_broken_raises(tmp_path):
    bad = tmp_path / "bad.cu"
    bad.write_text("extern \"C\" void minidwarf_solve(){ this is not c++ }")
    with pytest.raises(CompileError):
        compile_binary(bad, tmp_path)
