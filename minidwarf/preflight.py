# SPDX-License-Identifier: Apache-2.0
import shutil, subprocess

class GpuBusyError(RuntimeError): ...

_GPU_FIELDS = ["name", "driver_version", "clocks.sm", "clocks.mem", "temperature.gpu"]

def _smi(args):
    if not shutil.which("nvidia-smi"): return None
    r = subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, timeout=30)
    return r.stdout if r.returncode == 0 else None

def parse_compute_apps(text: str) -> list[str]:
    """Parse `nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader` output."""
    return [l.strip() for l in text.splitlines() if l.strip() and not l.strip().lower().startswith("no running")]

def parse_gpu_query(text: str) -> dict:
    """Parse one CSV line of the _GPU_FIELDS query ('[N/A]' -> None; malformed -> {})."""
    line = next((l for l in text.splitlines() if l.strip()), "")
    vals = [v.strip() for v in line.split(",")]
    if len(vals) != len(_GPU_FIELDS): return {}
    return {k: (None if v in ("[N/A]", "N/A", "") else v) for k, v in zip(_GPU_FIELDS, vals)}

def busy_processes() -> list[str]:
    out = _smi(["--query-compute-apps=pid,process_name", "--format=csv,noheader"])
    return parse_compute_apps(out) if out else []

def ensure_gpu_idle(allow_busy: bool = False) -> list[str]:
    """Raise GpuBusyError if another process computes on the GPU (unless allow_busy); return those processes."""
    procs = busy_processes()
    if procs and not allow_busy:
        raise GpuBusyError("GPU busy, refusing to time kernels: " + "; ".join(procs))
    return procs

def env_record() -> dict:
    """GPU name, driver, SM/memory clocks and temperature right now ({} if nvidia-smi is unavailable)."""
    out = _smi(["--query-gpu=" + ",".join(_GPU_FIELDS), "--format=csv,noheader,nounits"])
    return parse_gpu_query(out) if out else {}
