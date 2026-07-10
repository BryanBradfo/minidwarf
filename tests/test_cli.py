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

def test_cli_suite_missing_kernel_placeholder(tmp_path):
    root = Path(__file__).resolve().parents[1]
    suite_root = tmp_path / "suite_root"
    kernels_dir = tmp_path / "kernels"

    # Create temp suite structure: <suite_root>/mydwarf/myprob/spec.yaml
    prob_dir = suite_root / "mydwarf" / "myprob"
    prob_dir.mkdir(parents=True)
    spec_yaml = prob_dir / "spec.yaml"
    spec_yaml.write_text("""name: myprob
dwarf: mydwarf
difficulty: easy
rtol: 1.0e-5
atol: 1.0e-6
n_inputs: 1
n_outputs: 1
eval_shapes: [[16]]
baseline: author_kernel
""")

    # Create empty kernels dir (so myprob.cu is absent)
    kernels_dir.mkdir(parents=True)

    # Run suite command
    r = subprocess.run([sys.executable, "-m", "minidwarf.cli", "suite",
                        "--root", str(suite_root),
                        "--kernels", str(kernels_dir)],
                       cwd=root, capture_output=True, text=True)
    assert r.returncode == 0
    out = json.loads(r.stdout)

    # Assert on the summarized output
    assert out["n"] == 1
    assert out["compile_rate"] == 0.0
    assert out["correctness_rate"] == 0.0
    assert "fast_p" in out
    assert isinstance(out["fast_p"], dict)
