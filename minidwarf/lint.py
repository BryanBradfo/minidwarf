# SPDX-License-Identifier: Apache-2.0
import re

_LIBS = {"cublas": r"\bcublas\w*", "cusparse": r"\bcusparse\w*", "cufft": r"\bcufft\w*",
         "curand": r"\bcurand\w*", "cusolver": r"\bcusolver\w*"}
_ALWAYS = {
    "thrust": r"\bthrust\s*::|<thrust/",
    "cub": r"\bcub\s*::|<cub/",
    "file_io": r"\b(fopen|freopen|fread|ifstream|ofstream|fstream)\b|\bopen\s*\(",
    "process": r"\b(system|popen|fork|execl|execlp|execle|execv|execvp|execvpe)\s*\(",
    "dynamic_loading": r"\b(dlopen|dlsym)\s*\(",
    "environment": r"\bgetenv\s*\(",
}
_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
_STRING = re.compile(r'"(?:\\.|[^"\\\n])*"')

def _strip(src: str) -> str:
    # Drop comments everywhere and string literals outside preprocessor lines (#include "x.h" stays visible).
    src = _COMMENT.sub(" ", src)
    return "\n".join(l if l.lstrip().startswith("#") else _STRING.sub('""', l) for l in src.splitlines())

def lint_source(src: str, allowed_libs=()) -> list[str]:
    """Return the sorted names of forbidden-API rules matched in `src` (empty when clean)."""
    code = _strip(src)
    rules = {k: v for k, v in _LIBS.items() if k not in set(allowed_libs)} | _ALWAYS
    return sorted(k for k, pat in rules.items() if re.search(pat, code))
