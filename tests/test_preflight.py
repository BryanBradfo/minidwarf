# SPDX-License-Identifier: Apache-2.0
import pytest
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
    assert ensure_gpu_idle() == [] and pf.env_record() == {}
