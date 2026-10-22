# SPDX-License-Identifier: Apache-2.0
from minidwarf.extract import extract_kernel

def test_extracts_cuda_fence():
    t = "Here:\n```cuda\nextern \"C\" void minidwarf_solve(){}\n```\nDone."
    assert extract_kernel(t).strip() == 'extern "C" void minidwarf_solve(){}'

def test_prefers_longest_block():
    t = "```c\nint a;\n```\ntext\n```cuda\nlong b; long c; long d;\n```"
    assert "long b" in extract_kernel(t)

def test_plain_triple_fence():
    t = "```\nvoid minidwarf_solve(){}\n```"
    assert "minidwarf_solve" in extract_kernel(t)

def test_no_fence_returns_whole_text():
    t = "extern \"C\" void minidwarf_solve(){}"
    assert extract_kernel(t).strip() == t

def test_fence_with_trailing_words_on_info_line():
    # small models parrot the instruction "in a single ```cuda fenced block"
    src = '```cuda fenced block\n#include <cuda_runtime.h>\nint x;\n```'
    assert extract_kernel(src) == "#include <cuda_runtime.h>\nint x;"

def test_unclosed_fence_from_truncated_response():
    assert extract_kernel("Here:\n```cuda\nint x;\nint y;") == "int x;\nint y;"
