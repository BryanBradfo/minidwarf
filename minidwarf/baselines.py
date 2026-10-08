# SPDX-License-Identifier: Apache-2.0
BASELINE_FLAGS = {"author_kernel": [], "cublas": ["-lcublas"], "cusparse": ["-lcusparse"]}

def link_flags(baseline: str) -> list[str]:
    """Return the extra nvcc link flags (e.g. -lcublas) needed for a problem's declared `baseline` type."""
    if baseline not in BASELINE_FLAGS:
        raise ValueError(f"unknown baseline: {baseline!r}")
    return list(BASELINE_FLAGS[baseline])

LIB_FLAGS = {"cublas": ["-lcublas"], "cusparse": ["-lcusparse"], "cufft": ["-lcufft"],
             "curand": ["-lcurand"], "cusolver": ["-lcusolver"]}

def lib_flags(libs) -> list[str]:
    """Return the nvcc link flags for a candidate's `allowed_libs` (raises on an unknown library)."""
    out = []
    for lib in libs:
        if lib not in LIB_FLAGS:
            raise ValueError(f"unknown library: {lib!r}")
        out += LIB_FLAGS[lib]
    return out
