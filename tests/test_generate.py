# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
from minidwarf.evalconfig import EvalConfig
from minidwarf.models.dummy import DummyModel
from minidwarf.generate import run_generation, SYSTEM_PROMPT

SMOKE = Path(__file__).parent / "fixtures/smoke/vector_add"

def test_run_generation_writes_artifacts(tmp_path):
    cfg = EvalConfig(model_name="dummy", base_url="u", provider_model_name="p",
                     provider="dummy", n_samples=2)
    kernel = (SMOKE / "solutions/expert_v1.cu").read_text()
    run_dir = run_generation(cfg, [SMOKE], tmp_path, run_id="r1",
                             model=DummyModel(f"```cuda\n{kernel}\n```"))
    assert (run_dir / "results.jsonl").exists()
    rows = [json.loads(l) for l in (run_dir / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 2  # n_samples
    assert all(r["problem"] == "vector_add" for r in rows)
    assert (run_dir / "kernels/vector_add__s0.cu").exists()
    assert "minidwarf_solve" in (run_dir / "kernels/vector_add__s0.cu").read_text()

def test_system_prompt_mentions_abi():
    assert "minidwarf_solve" in SYSTEM_PROMPT

def test_system_prompt_states_v3_rules():
    for s in ("<cuda_runtime.h>", "cuBLAS", "cuSPARSE", "cuFFT", "cuRAND", "cuSOLVER", "Thrust", "CUB", "file I/O",
              "process", "environment", "dynamic loading", "syscall", "constructor", "raw string", "number of times"):
        assert s in SYSTEM_PROMPT, s

from minidwarf.models.base import GenResult, Model

class _Scripted(Model):
    def __init__(self, results): self.results, self.calls = list(results), 0
    def generate(self, prompt):
        self.calls += 1; return self.results.pop(0)

_OK = GenResult("```cuda\nint ok;\n```")
_QUOTA = GenResult("", "error", "Error code: 429 - RESOURCE_EXHAUSTED quota GenerateRequestsPerDayPerProjectPerModel")

def _cfg(n): return EvalConfig(model_name="m", base_url="u", provider_model_name="p", provider="dummy", n_samples=n)

def test_resume_regenerates_only_missing_or_failed(tmp_path):
    run_generation(_cfg(2), [SMOKE], tmp_path, "r", model=_Scripted([_OK, GenResult("", "error", "timeout")]))
    m = _Scripted([_OK])
    run_dir = run_generation(_cfg(2), [SMOKE], tmp_path, "r", model=m, resume=True)
    rows = [json.loads(l) for l in (run_dir / "results.jsonl").read_text().splitlines()]
    assert m.calls == 1 and [r["generation_status"] for r in rows] == ["success", "success"]

def test_daily_quota_stops_generation_without_recording_failures(tmp_path):
    m = _Scripted([_OK, _QUOTA, _OK])
    run_dir = run_generation(_cfg(3), [SMOKE], tmp_path, "r", model=m)
    rows = [json.loads(l) for l in (run_dir / "results.jsonl").read_text().splitlines()]
    assert m.calls == 2 and len(rows) == 1 and rows[0]["generation_status"] == "success"

def test_unavailable_model_stops_generation(tmp_path):
    gone = GenResult("", "error", "Error code: 404 - model is no longer available to new users")
    m = _Scripted([gone, _OK])
    run_dir = run_generation(_cfg(2), [SMOKE], tmp_path, "r", model=m)
    assert m.calls == 1 and (run_dir / "results.jsonl").read_text() == ""
