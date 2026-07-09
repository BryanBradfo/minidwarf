# SPDX-License-Identifier: Apache-2.0
import subprocess, sys
from pathlib import Path

def test_check_env_reports_clearly():
    root = Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, "scripts/check_env.py"], cwd=root,
                       capture_output=True, text=True)
    # Either nvcc works (0) or it fails with an actionable message (non-zero).
    assert r.returncode == 0 or "nvcc" in (r.stdout + r.stderr).lower()
