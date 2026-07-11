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


def test_split_is_balanced():
    # The benchmark grows across versions, so the split is not pinned to a fixed
    # count; it must stay near-balanced (valid and test differ by at most one).
    s = yaml.safe_load((ROOT / "SPLITS.yaml").read_text())
    assert abs(len(s["valid"]) - len(s["test"])) <= 1
