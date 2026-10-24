import os, numpy as np, pytest
import bench, data

def test_leak_error_fires():
    h = bench.History("a", np.arange(1.0, 11), np.ones(10), np.ones(10), np.arange(10), np.zeros(20))
    with pytest.raises(bench.LeakError):
        h.at(h.t + 1)

def test_holdout_locked_outside_final_report():
    with pytest.raises(PermissionError):
        data.panel(allow_holdout=True)

def test_registry_counts_every_evaluation(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, 'REG', str(tmp_path / 'r.db'))
    n0 = bench.n_evals()
    assert bench.register("x", 0, "dev") == n0 + 1 and bench.n_evals() == n0 + 1
