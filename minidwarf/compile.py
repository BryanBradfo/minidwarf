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
