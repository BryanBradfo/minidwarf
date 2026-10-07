# SPDX-License-Identifier: Apache-2.0
import functools, re, subprocess, tempfile, warnings
from pathlib import Path
from .lint import lint_symbols, lint_defined

DRIVER = Path(__file__).resolve().parents[1] / "harness" / "driver.cu"
NVCC_FLAGS = ["-O3"]  # single source of truth: every build (object and one-step) uses these

class CompileError(Exception): ...

def _run(cmd, timeout=120):
    # errors="replace": a candidate's stderr/stdout may hold arbitrary bytes and must never raise.
    return subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=timeout)

def compile_binary(candidate_cu: Path, out_dir: Path, arch: str = "sm_120", extra_flags=None) -> Path:
    """Compile `candidate_cu` together with the harness driver via nvcc, returning the built executable's path."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    exe = out_dir / (Path(candidate_cu).stem + ".bin")
    cmd = ["nvcc", f"-arch={arch}", *NVCC_FLAGS, "-o", str(exe), str(DRIVER), str(candidate_cu)]
    if extra_flags:
        cmd += list(extra_flags)
    r = _run(cmd)
    if r.returncode != 0:
        raise CompileError(r.stderr)
    return exe

def compile_object(candidate_cu: Path, out_dir: Path, arch: str = "sm_120") -> Path:
    """Compile the candidate alone to an object with the real build flags; all checks run on this exact object."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    obj = out_dir / (Path(candidate_cu).stem + ".o")
    r = _run(["nvcc", f"-arch={arch}", *NVCC_FLAGS, "-c", str(candidate_cu), "-o", str(obj)])
    if r.returncode != 0:
        raise CompileError(r.stderr)
    return obj

def _link_cmd(obj, exe, arch, extra_flags=None) -> list[str]:
    # The driver MUST precede the candidate object: weak/comdat duplicates resolve to the first definition,
    # so the driver's own template instantiations win (see lint_defined).
    return ["nvcc", f"-arch={arch}", *NVCC_FLAGS, "-o", str(exe), str(DRIVER), str(obj), *(extra_flags or [])]

def link_binary(obj: Path, out_dir: Path, arch: str = "sm_120", extra_flags=None) -> Path:
    """Link a checked candidate object with the harness driver."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    exe = out_dir / (Path(obj).stem + ".bin")
    r = _run(_link_cmd(obj, exe, arch, extra_flags))
    if r.returncode != 0:
        raise CompileError(r.stderr)
    return exe

def _nm(args, obj) -> list[str]:
    r = _run(["nm", *args, str(obj)], 60)
    if r.returncode != 0:
        raise CompileError(r.stderr)
    return [ln.split()[-1] for ln in r.stdout.splitlines() if ln.split()]

def undefined_symbols(obj: Path) -> list[str]:
    """Undefined symbols of a compiled object (`nm -u`)."""
    return _nm(["-u"], Path(obj))

def _nm_typed(args, obj) -> list[tuple[str, str]]:
    r = _run(["nm", *args, str(obj)], 60)
    if r.returncode != 0:
        raise CompileError(r.stderr)
    return [(f[-1], f[-2]) for f in (ln.split() for ln in r.stdout.splitlines()) if len(f) >= 2]

def defined_symbols(obj: Path) -> list[tuple[str, str]]:
    """(name, nm type) of an object's defined globals."""
    return _nm_typed(["-g", "--defined-only"], obj)

@functools.lru_cache(maxsize=None)
def _forbidden_defined() -> tuple[frozenset, frozenset]:
    """(names forbidden at any strength, driver-defined names forbidden only as strong definitions)."""
    with tempfile.TemporaryDirectory() as d:
        r = _run(["nvcc", "-arch=sm_120", *NVCC_FLAGS, "-c", str(DRIVER), "-o", str(Path(d) / "driver.o")])
        if r.returncode != 0:
            raise CompileError(r.stderr)
        drv = Path(d) / "driver.o"
        # the driver's own globals (incl. weak template code) must not be replaced by candidate strong symbols
        names = set(_nm(["-u"], drv))
        strong = {n for n, _ in defined_symbols(drv) if n != "minidwarf_solve" and not n.startswith("DW.ref.")}
    try:  # libc exports; skipped when the library cannot be resolved to a real file
        g = _run(["gcc", "-print-file-name=libc.so.6"], 30).stdout.strip()
        libc = Path(g).resolve() if g else None
        if libc and libc.is_file():
            names |= set(_nm(["-D", "--defined-only"], libc))
        else:
            warnings.warn("libc not resolved: libc-export interposition check skipped")
    except (OSError, CompileError, subprocess.TimeoutExpired):
        warnings.warn("libc export list unavailable: libc-export interposition check skipped")
    return frozenset(names), frozenset(strong)

_ASM = re.compile(r"^\s*[0-9a-f]+:\s+(syscall|sysenter|int\s+\$0x80)\b", re.M)

def asm_hits(obj: Path) -> list[str]:
    r = _run(["objdump", "-d", "--no-show-raw-insn", str(obj)], 120)
    if r.returncode != 0:
        raise CompileError(r.stderr)
    return ["asm:syscall"] if _ASM.search(r.stdout) else []

def object_hits(obj: Path, allowed_libs=()) -> list[str]:
    """All object-level policy hits (undefined symbols, interposing definitions, inline syscalls), sorted."""
    return sorted(set(lint_symbols(undefined_symbols(obj), allowed_libs)
                      + lint_defined(defined_symbols(obj), *_forbidden_defined()) + asm_hits(obj)))
