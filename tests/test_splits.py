# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_every_problem_in_exactly_one_split():
    names = {p.parent.name for p in ROOT.glob("problems/*/*/spec.yaml")}
    s = yaml.safe_load((ROOT / "SPLITS.yaml").read_text())
    listed = s["valid"] + s["test"]
    assert set(listed) == names
    assert len(listed) == len(set(listed))  # no duplicates, full coverage


def test_split_is_balanced_six_and_six():
    s = yaml.safe_load((ROOT / "SPLITS.yaml").read_text())
    assert len(s["valid"]) == 6
    assert len(s["test"]) == 6
