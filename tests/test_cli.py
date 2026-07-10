# SPDX-License-Identifier: Apache-2.0
import json, subprocess, sys
from pathlib import Path
FIX = Path(__file__).parent / "fixtures/smoke/vector_add"

def test_cli_run_smoke(tmp_path):
    root = Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, "-m", "minidwarf.cli", "run",
                        "--problem", str(FIX),
                        "--kernel", str(FIX / "solutions/expert_v1.cu")],
                       cwd=root, capture_output=True, text=True)
    assert r.returncode == 0
    out = json.loads(r.stdout)
    assert out["status"] == "ok" and out["correct"] is True
