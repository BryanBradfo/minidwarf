# SPDX-License-Identifier: Apache-2.0
import pytest, subprocess
import minidwarf.preflight as pf
from minidwarf.preflight import parse_compute_apps, parse_gpu_query, ensure_gpu_idle, GpuBusyError

def test_parse_compute_apps():
    assert parse_compute_apps("2167609, python\n123, ollama\n") == ["2167609, python", "123, ollama"]
    assert parse_compute_apps("") == []
    assert parse_compute_apps("No running processes found\n") == []

def test_parse_gpu_query_handles_na():
    d = parse_gpu_query("NVIDIA GeForce RTX 5070 Laptop GPU, 580.178.04, 1980, [N/A], 49\n")
    assert d["name"].startswith("NVIDIA") and d["clocks.mem"] is None and d["temperature.gpu"] == "49"
    assert parse_gpu_query("garbage") == {}

def test_ensure_gpu_idle_raises_when_busy(monkeypatch):
    monkeypatch.setattr(pf, "_smi", lambda args: "42, python\n")
    with pytest.raises(GpuBusyError):
        ensure_gpu_idle()
    assert ensure_gpu_idle(allow_busy=True) == ["42, python"]

def test_ensure_gpu_idle_without_nvidia_smi(monkeypatch):
    monkeypatch.setattr(pf, "_smi", lambda args: None)
    monkeypatch.setattr(pf.shutil, "which", lambda n: None)
    assert ensure_gpu_idle() == [] and pf.env_record() == {}

def test_ensure_gpu_idle_fails_closed_when_smi_query_fails(monkeypatch):
    monkeypatch.setattr(pf, "_smi", lambda args: None)
    monkeypatch.setattr(pf.shutil, "which", lambda n: "/usr/bin/nvidia-smi")
    with pytest.raises(GpuBusyError, match="GPU state unknown"):
        ensure_gpu_idle()
    assert ensure_gpu_idle(allow_busy=True) == []
    assert pf.env_record() == {}

@pytest.mark.parametrize("exc", [subprocess.TimeoutExpired("nvidia-smi", 30), OSError("boom")])
def test_smi_returns_none_on_timeout_or_oserror(monkeypatch, exc):
    monkeypatch.setattr(pf.shutil, "which", lambda n: "/usr/bin/nvidia-smi")
    def boom(*a, **k): raise exc
    monkeypatch.setattr(pf.subprocess, "run", boom)
    assert pf._smi(["-L"]) is None
