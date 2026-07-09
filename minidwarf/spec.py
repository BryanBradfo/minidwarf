# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from pathlib import Path
import yaml

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

def load_problem(root: Path) -> Problem:
    root = Path(root)
    data = yaml.safe_load((root / "spec.yaml").read_text()) or {}
    missing = [k for k in _REQUIRED if k not in data]
    if missing:
        raise ValueError(f"{root}/spec.yaml missing fields: {missing}")
    return Problem(
        root=root, name=data["name"], dwarf=data["dwarf"],
        difficulty=data["difficulty"], rtol=float(data["rtol"]),
        atol=float(data["atol"]), n_inputs=int(data["n_inputs"]),
        n_outputs=int(data["n_outputs"]),
        eval_shapes=[list(map(int, s)) for s in data["eval_shapes"]],
        baseline=data["baseline"],
    )
