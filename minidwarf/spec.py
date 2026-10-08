# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from pathlib import Path
import yaml
from .baselines import LIB_FLAGS

_REQUIRED = ["name", "dwarf", "difficulty", "rtol", "atol",
             "n_inputs", "n_outputs", "eval_shapes", "baseline"]

@dataclass(frozen=True)
class Problem:
    root: Path
    name: str
    dwarf: str
    difficulty: str
    rtol: float
    atol: float
    n_inputs: int
    n_outputs: int
    eval_shapes: list[list[int]]
    baseline: str
    check_shapes: list[list[int]] = field(default_factory=list)
    allowed_libs: list[str] = field(default_factory=list)

def load_problem(root: Path) -> Problem:
    """Load and validate a problem's spec.yaml from `root`, returning it as a Problem."""
    root = Path(root)
    spec_path = root / "spec.yaml"
    if not spec_path.exists():
        raise ValueError(f"{spec_path} does not exist")
    data = yaml.safe_load(spec_path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{spec_path} must be a mapping, got {type(data).__name__}")
    missing = [k for k in _REQUIRED if k not in data]
    if missing:
        raise ValueError(f"{root}/spec.yaml missing fields: {missing}")
    allowed = data.get("allowed_libs") or []
    if not isinstance(allowed, list):
        raise ValueError(f"{root}/spec.yaml: allowed_libs must be a list, got {allowed!r}")
    checks = data.get("check_shapes") or []
    if not isinstance(checks, list) or not all(isinstance(s, list) for s in checks):
        raise ValueError(f"{root}/spec.yaml: check_shapes must be a list of shapes, got {checks!r}")
    unknown = [l for l in allowed if l not in LIB_FLAGS]
    if unknown:
        raise ValueError(f"{root}/spec.yaml: unknown allowed_libs {unknown}")
    return Problem(
        root=root, name=data["name"], dwarf=data["dwarf"],
        difficulty=data["difficulty"], rtol=float(data["rtol"]),
        atol=float(data["atol"]), n_inputs=int(data["n_inputs"]),
        n_outputs=int(data["n_outputs"]),
        eval_shapes=[list(map(int, s)) for s in data["eval_shapes"]],
        baseline=data["baseline"],
        check_shapes=[list(map(int, s)) for s in checks],
        allowed_libs=allowed,
    )
