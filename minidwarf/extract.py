# SPDX-License-Identifier: Apache-2.0
import re

_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9+#]*)[ \t]*\r?\n(.*?)```", re.DOTALL)
_LANGS = {"cuda", "cpp", "c", "cu", "c++", ""}

def extract_kernel(text: str) -> str:
    """Extract CUDA source from an LLM response.

    Returns the longest fenced code block whose language tag is a C/CUDA
    variant (or untagged); if no such block exists, returns the whole
    stripped text.
    """
    blocks = [(lang.lower(), body) for lang, body in _FENCE.findall(text or "")]
    candidates = [body for lang, body in blocks if lang in _LANGS]
    pool = candidates or [body for _, body in blocks]
    if not pool:
        return (text or "").strip()
    return max(pool, key=len).strip()
