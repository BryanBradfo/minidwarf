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
