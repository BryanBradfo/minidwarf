# SPDX-License-Identifier: Apache-2.0
import json, math, os
from pathlib import Path

def sanitize(obj):
    if isinstance(obj, float): return obj if math.isfinite(obj) else None
    if isinstance(obj, dict): return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [sanitize(v) for v in obj]
    return obj

def write_report(path, obj):
    """Atomically rewrite a strict JSON report (non-finite floats become null)."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(sanitize(obj), indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)
