# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path
import numpy as np

def load_module_fn(problem_root: Path, filename: str, fn: str):
    path = Path(problem_root) / filename
    spec = importlib.util.spec_from_file_location(path.stem + "_" + fn, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, fn)

def write_arrays(path: Path, arrays: list[np.ndarray]) -> None:
    with open(path, "wb") as fh:
        for a in arrays:
            if a.dtype == np.int32:
                fh.write(np.ascontiguousarray(a, dtype=np.int32).tobytes())
            else:
                fh.write(np.ascontiguousarray(a, dtype=np.float32).tobytes())

def read_arrays(path: Path, specs) -> list[np.ndarray]:
    raw = np.fromfile(path, dtype=np.uint8)
    out, off = [], 0
    for shape, dtype in specs:
        n = int(np.prod(shape)); nbytes = n * 4
        chunk = raw[off:off + nbytes].view(dtype).reshape(shape)
        out.append(chunk); off += nbytes
    return out
