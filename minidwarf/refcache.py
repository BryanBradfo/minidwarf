# SPDX-License-Identifier: Apache-2.0
import hashlib, json, os, tempfile
from pathlib import Path
import numpy as np
from .problem_io import load_module_fn

def cache_dir() -> Path:
    return Path(os.environ.get("MINIDWARF_CACHE") or Path.home() / ".cache" / "minidwarf")

def case_key(problem_root, shape, seed) -> str:
    """Hash of the problem's generator + reference sources, the shape and the seed."""
    root = Path(problem_root); h = hashlib.sha256()
    for f in ("inputs.py", "reference.py"): h.update((root / f).read_bytes())
    h.update(json.dumps([[int(x) for x in shape], int(seed)]).encode())
    return h.hexdigest()[:32]

def cached_case(problem_root, shape, seed):
    """Return (inputs, expected) for one case, generating and caching it on first use."""
    root = Path(problem_root)
    path = cache_dir() / "cases" / root.name / f"{case_key(root, shape, seed)}.npz"
    if path.exists():
        try:
            with np.load(path) as z:
                return ([z[f"in_{i}"] for i in range(int(z["n_in"]))],
                        [z[f"out_{i}"] for i in range(int(z["n_out"]))])
        except Exception:
            pass  # partial or corrupt file: recompute and overwrite
    ins = load_module_fn(root, "inputs.py", "generate")(shape, seed)
    outs = [np.asarray(e, dtype=np.float32) for e in load_module_fn(root, "reference.py", "run")(ins, shape)]
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp.npz"); os.close(fd)
    arrays = {f"in_{i}": a for i, a in enumerate(ins)} | {f"out_{i}": a for i, a in enumerate(outs)}
    try:
        np.savez(tmp, n_in=len(ins), n_out=len(outs), **arrays)
    except BaseException:
        os.unlink(tmp); raise  # never leave multi-GB orphans behind (e.g. disk full)
    os.replace(tmp, path)  # atomic: concurrent readers never see a half-written file
    return ins, outs
