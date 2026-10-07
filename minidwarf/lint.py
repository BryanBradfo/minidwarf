# SPDX-License-Identifier: Apache-2.0
import re

_LIBS = {"cublas": r"\bcublas\w*", "cusparse": r"\bcusparse\w*", "cufft": r"\bcufft\w*",
         "curand": r"\bcurand\w*", "cusolver": r"\bcusolver\w*"}
_ALWAYS = {
    "thrust": r"\bthrust\s*::|[<\"]thrust/",
    "cub": r"\bcub\s*::|[<\"]cub/",
    "file_io": r"\b(fopen|freopen|fread|ifstream|ofstream|fstream)\b|\bopen\s*\(",
    "process": r"\b(system|popen|fork|exec\w*)\s*\(",
    "dynamic_loading": r"\b(dlopen|dlsym)\s*\(",
    "environment": r"\bgetenv\s*\(",
    "constructor": r"__attribute__\s*\(\(\s*constructor|\[\[\s*gnu::constructor",
}
# One pass so a comment marker inside a literal (or a quote inside a char literal) cannot mask code.
_TOKEN = re.compile(r'"(?:\\.|[^"\\\n])*"|(?<![\w])(?:u8|u|U|L)?\'(?:\\.|[^\'\\\n])*\'|//[^\n]*|/\*.*?\*/', re.S)
# Raw strings are not followed by the lexer, so any use fails closed.
_RAW = re.compile(r'(?<![\w])(?:u8|u|U|L)?R"')

def _strip(src: str) -> str:
    # Blank comments and literal contents, except literals on preprocessor lines (#include "x.h" stays visible).
    def repl(m):
        t = m.group(0)
        if t.startswith("/"):
            return " " + "\n" * t.count("\n")
        start = src.rfind("\n", 0, m.start()) + 1
        end = src.find("\n", m.start())
        if src[start:len(src) if end < 0 else end].lstrip().startswith("#"):
            return t
        return '""' if t[0] == '"' else "''"
    return _TOKEN.sub(repl, src)

def lint_source(src: str, allowed_libs=()) -> list[str]:
    """Return the sorted names of forbidden-API rules matched in `src` (empty when clean)."""
    code = _strip(src)
    rules = {k: v for k, v in _LIBS.items() if k not in set(allowed_libs)} | _ALWAYS
    hits = [k for k, pat in rules.items() if re.search(pat, code)]
    if _RAW.search(src):
        hits.append("raw_string")
    return sorted(hits)

_BAN_EXACT = frozenset("""fopen fopen64 freopen fread open open64 openat openat64 creat read pread pread64 mmap mmap64
system popen fork vfork execl execlp execle execv execvp execvpe execve fexecve posix_spawn posix_spawnp syscall
dlopen dlsym dlmopen getenv secure_getenv exit _exit _Exit quick_exit pthread_create cudaSetDevice cudaDeviceReset""".split())
_BAN_SUB = ("_M_start_thread", "basic_ifstream", "basic_ofstream", "basic_fstream", "basic_filebuf")

def lint_symbols(names, allowed_libs=()) -> list[str]:
    """Policy over a compiled candidate's undefined symbols: sorted `symbol:<name>` entries for banned ones."""
    prefixes = tuple(k for k in _LIBS if k not in set(allowed_libs))
    return sorted(f"symbol:{n}" for n in set(names)
                  if n in _BAN_EXACT or n.startswith(prefixes) or any(b in n for b in _BAN_SUB))
