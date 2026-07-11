# SPDX-License-Identifier: Apache-2.0
import json, subprocess, tempfile
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .problem_io import write_arrays, read_arrays

class RunError(Exception): ...

@dataclass
class RunResult:
    median_ms: float
    outputs: list

def run_binary(exe, inputs, dims, output_shapes, reps=30, timeout_s=60) -> RunResult:
    in_counts = [int(np.prod(a.shape)) for a in inputs]
    out_counts = [int(np.prod(s)) for s in output_shapes]
    with tempfile.TemporaryDirectory() as d:
        din, dout, dt = Path(d)/"in.bin", Path(d)/"out.bin", Path(d)/"t.json"
        write_arrays(din, inputs)
        argv = [str(exe), str(din), str(dout), str(dt),
                str(len(inputs)), str(len(output_shapes)), str(reps), str(len(dims))]
        argv += [str(int(x)) for x in dims]
        argv += [str(c) for c in in_counts]
        argv += [str(c) for c in out_counts]
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s)
        if r.returncode != 0:
            raise RunError(r.stderr or "non-zero exit")
        median = json.loads(dt.read_text())["median_ms"]
        outs = read_arrays(dout, [(s, np.float32) for s in output_shapes])
    return RunResult(median_ms=median, outputs=outs)
