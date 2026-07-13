# SPDX-License-Identifier: Apache-2.0
BASELINE_FLAGS = {"author_kernel": [], "cublas": ["-lcublas"], "cusparse": ["-lcusparse"]}

def link_flags(baseline: str) -> list[str]:
    """Return the extra nvcc link flags (e.g. -lcublas) needed for a problem's declared `baseline` type."""
    if baseline not in BASELINE_FLAGS:
        raise ValueError(f"unknown baseline: {baseline!r}")
    return list(BASELINE_FLAGS[baseline])
