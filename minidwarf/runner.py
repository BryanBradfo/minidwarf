# SPDX-License-Identifier: Apache-2.0
import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .problem_io import write_arrays, read_arrays

class RunError(Exception): ...

@dataclass
class RunResult:
    median_ms: float
    outputs: list

def run_binary(exe: Path, inputs, dims, n_outputs, reps=30, timeout_s=60) -> RunResult:
    n = int(np.prod(dims))
    with tempfile.TemporaryDirectory() as d:
        din, dout, dt = Path(d) / "in.bin", Path(d) / "out.bin", Path(d) / "t.json"
        write_arrays(din, inputs)
        cmd = [str(exe), str(din), str(dout), str(dt),
               str(len(inputs)), str(n_outputs), str(reps)] + [str(x) for x in dims]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        if r.returncode != 0:
            raise RunError(r.stderr or "non-zero exit")
        median = json.loads(dt.read_text())["median_ms"]
        outs = read_arrays(dout, [(n,)] * n_outputs)
    return RunResult(median_ms=median, outputs=outs)
