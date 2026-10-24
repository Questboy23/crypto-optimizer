import numpy as np
from engine import stats as S

def test_sb_indices_shape_and_range():
    idx = S.sb_indices(500, 10, 100, np.random.default_rng(0))
    assert idx.shape == (100, 500) and idx.min() >= 0 and idx.max() < 500

def test_mean_ci_covers_and_pvalue():
    rng = np.random.default_rng(1)
    x = rng.normal(0.3, 1, 400)
    e = S.mean_ci(x)
    assert e.lo < e.point < e.hi and e.p < 0.05
    z = rng.normal(0, 1, 400)
    assert S.mean_ci(z).p > 0.05

def test_ci_coverage_on_null():
    cover = 0
    for s in range(100):
        x = np.random.default_rng(100 + s).normal(0, 1, 300)
        e = S.mean_ci(x, B=400, seed=s)
        cover += e.lo <= 0 <= e.hi
    assert 88 <= cover <= 100

def test_null_pvalue_uniform_under_null():
    rng = np.random.default_rng(2)
    p = [S.null_pvalue(rng.normal(), rng.normal(size=199)) for _ in range(300)]
    assert S.ks_uniform_p(p) > 0.01

def test_bh_monotone():
    p = np.array([0.001, 0.02, 0.03, 0.5])
    a = S.bh_adjust(p)
    assert np.all(a >= p) and np.all(np.diff(a[np.argsort(p)]) >= 0)

def test_search_penalty_grows_with_N():
    ns = np.random.default_rng(3).normal(0, 5, 200)
    assert S.search_penalty(ns, 1) == 0 < S.search_penalty(ns, 10) < S.search_penalty(ns, 150) < S.search_penalty(ns, 10000)

def test_best_of_n_p_larger_than_single():
    ns = np.random.default_rng(4).normal(0, 5, 200)
    assert S.best_of_n_pvalue(8.0, ns, 100) > S.null_pvalue(8.0, ns)

def test_rule_of_three():
    assert abs(S.rule_of_three(30) - 0.0999) < 1e-3

def test_tost_equivalent_vs_not():
    rng = np.random.default_rng(5)
    assert S.tost(rng.normal(0, .02, 2000), 0.1)[0]
    assert not S.tost(rng.normal(0.1, .02, 2000), 0.1)[0]

def test_risk_report_keys():
    r = np.random.default_rng(6).normal(0.0005, 0.03, 700)
    rr = S.risk_report(r, B=200)
    assert rr["max_drawdown"].lo <= rr["max_drawdown"].point * 1.5
    assert 0 <= rr["p_loss_1y"] <= 1 and 0 <= rr["p_50pct_drawdown_1y"] <= 1

def test_pbo_noise_near_half():
    M = np.random.default_rng(7).normal(0, 1, (640, 30))
    assert 0.25 < S.pbo_cscv(M) < 0.75

def test_dsr_decreases_with_trials():
    assert S.deflated_sharpe(0.1, 1, 500) > S.deflated_sharpe(0.1, 500, 500)


def test_fixed_share_mixture_tracks_better_program():
    class H:  # minimal history stub
        def __init__(self, t, c): self.coin, self.t, self.close, self.edges = "a", t, c, np.array([0.0])
    good = lambda h: np.array([0.1, 0.9]); bad = lambda h: np.array([0.9, 0.1])
    m = S.FixedShareMixture([good, bad], alpha=0.01)
    c = [100.0]
    for t in range(60):
        c.append(c[-1] * 1.01)       # always up -> bin 1
        p = m(H(t + 1, np.array(c)))
    assert p[1] > 0.85
