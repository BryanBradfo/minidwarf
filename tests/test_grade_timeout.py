# SPDX-License-Identifier: Apache-2.0
from minidwarf.grade import candidate_timeout, SLOW_FACTOR
from minidwarf.runner import work_dir_root

def test_candidate_timeout_scales_with_baseline():
    assert candidate_timeout(0.5, 60) == 60  # small problems keep the floor
    assert candidate_timeout(10.0, 60) == 10.0 * SLOW_FACTOR  # slow baselines get proportional room

def test_work_dir_root_honours_env(monkeypatch, tmp_path):
    monkeypatch.delenv("MINIDWARF_TMP", raising=False)
    assert work_dir_root() is None  # system temp dir by default
    monkeypatch.setenv("MINIDWARF_TMP", str(tmp_path))
    assert work_dir_root() == str(tmp_path)
