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
    p25_ms: float
    p75_ms: float
    times_ms: list
    outputs: list  # outputs[data_set][output_index]

def run_binary(exe, input_sets, dims, output_shapes, reps=20, warmup=3, timeout_s=60) -> RunResult:
    """Run a compiled binary with D input data sets rotated through the same device buffers.

    Returns timing stats over `reps` timed calls and, per data set, the outputs of its last timed call."""
    n_sets = len(input_sets)
    if n_sets == 0 or reps < n_sets:
        raise ValueError(f"need reps >= n_sets >= 1, got reps={reps}, n_sets={n_sets}")
    in_counts = [int(np.prod(a.shape)) for a in input_sets[0]]
    out_counts = [int(np.prod(s)) for s in output_shapes]
    with tempfile.TemporaryDirectory() as d:
        din, dout, dt = Path(d)/"in.bin", Path(d)/"out.bin", Path(d)/"t.json"
        write_arrays(din, [a for s in input_sets for a in s])
        argv = [str(exe), str(din), str(dout), str(dt), str(len(in_counts)), str(len(out_counts)),
                str(n_sets), str(reps), str(warmup), str(len(dims))]
        argv += [str(int(x)) for x in dims] + [str(c) for c in in_counts] + [str(c) for c in out_counts]
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s)
        if r.returncode != 0:
            raise RunError(r.stderr or "non-zero exit")
        t = json.loads(dt.read_text())
        flat = read_arrays(dout, [(s, np.float32) for _ in range(n_sets) for s in output_shapes])
    k = len(output_shapes)
    return RunResult(t["median_ms"], t["p25_ms"], t["p75_ms"], t["times_ms"],
                     [flat[i*k:(i+1)*k] for i in range(n_sets)])
