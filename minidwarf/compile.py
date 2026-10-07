# SPDX-License-Identifier: Apache-2.0
import subprocess
from pathlib import Path

DRIVER = Path(__file__).resolve().parents[1] / "harness" / "driver.cu"

class CompileError(Exception): ...

def compile_binary(candidate_cu: Path, out_dir: Path, arch: str = "sm_120", extra_flags=None) -> Path:
    """Compile `candidate_cu` together with the harness driver via nvcc, returning the built executable's path."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    exe = out_dir / (Path(candidate_cu).stem + ".bin")
    cmd = ["nvcc", f"-arch={arch}", "-O3", "-o", str(exe),
           str(DRIVER), str(candidate_cu)]
    if extra_flags:
        cmd += list(extra_flags)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise CompileError(r.stderr)
    return exe


def undefined_symbols(candidate_cu: Path, out_dir: Path, arch: str = "sm_120") -> list[str]:
    """Compile the candidate alone to an object and list its undefined symbols (`nm -u`)."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    obj = out_dir / "cand_check.o"
    r = subprocess.run(["nvcc", f"-arch={arch}", "-c", str(candidate_cu), "-o", str(obj)],
                       capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise CompileError(r.stderr)
    r = subprocess.run(["nm", "-u", str(obj)], capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise CompileError(r.stderr)
    return [ln.split()[-1] for ln in r.stdout.splitlines() if ln.split()]
