# SPDX-License-Identifier: Apache-2.0
import importlib.util, shutil
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
PROBLEMS = sorted(p.parent for p in (ROOT / "problems").glob("*/*/spec.yaml"))
_spec = importlib.util.spec_from_file_location("calibrate_tol", ROOT / "scripts/calibrate_tol.py")
cal = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(cal)

@pytest.mark.skipif(shutil.which("nvcc") is None, reason="needs CUDA toolchain")
@pytest.mark.parametrize("pdir", PROBLEMS, ids=lambda p: p.name)
def test_tolerance_is_calibrated(pdir):
    row = cal.calibrate(pdir)
    assert row["status"] == "ok" and row["verdict"] in ("ok", "exact"), row
