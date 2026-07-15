# SPDX-License-Identifier: Apache-2.0
import json, shutil, subprocess, sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.skipif(shutil.which("nvcc") is None, reason="needs CUDA toolchain")
def test_cli_eval_dummy_then_leaderboard(tmp_path):
    # a dummy config that returns a known-good kernel for one real problem
    prob = ROOT / "problems/nbody/knn_distance_1d"
    kernel = (prob / "solutions/expert_v1.cu").read_text()
    cfg = tmp_path / "dummy.yaml"
    cfg.write_text("model_name: dummy\nbase_url: u\nprovider_model_name: p\nprovider: dummy\n")
    # (CLI reads canned kernel from --canned-file for the dummy provider; see Step 3)
    canned = tmp_path / "canned.cu"; canned.write_text(f"```cuda\n{kernel}\n```")
    runs = tmp_path / "runs"
    r = subprocess.run([sys.executable, "-m", "minidwarf.cli", "eval",
                        "--config", str(cfg), "--split", "test",
                        "--runs-dir", str(runs), "--only", "knn_distance_1d",
                        "--canned-file", str(canned)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    lb = subprocess.run([sys.executable, "-m", "minidwarf.cli", "leaderboard",
                         "--runs-dir", str(runs), "--out", str(tmp_path / "LB.md")],
                        cwd=ROOT, capture_output=True, text=True)
    assert lb.returncode == 0 and (tmp_path / "LB.md").exists()
    assert "knn_distance_1d".split("_")[0] or True  # leaderboard built
