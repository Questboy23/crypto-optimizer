"""One statistics module. Every number in reports, evaluators and certificates comes from here."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from arch.bootstrap import optimal_block_length

PERIODS = 365  # daily crypto, 24/7


@dataclass(frozen=True)
class Est:
    point: float
    lo: float
    hi: float
    p: float | None = None

    def __str__(self):
        s = f"{self.point:.4g} (95% CI {self.lo:.4g} to {self.hi:.4g})"
        return s + (f", p={self.p:.3g}" if self.p is not None else "")


def block_length(x) -> float:
    x = np.asarray(x, float)
    if len(x) < 20 or np.ptp(x) == 0:
        return 1.0
    try:
        b = float(optimal_block_length(x)["stationary"].iloc[0])
    except Exception:
        b = 1.0
    return float(np.clip(b if np.isfinite(b) else 1.0, 1.0, max(1.0, len(x) / 4)))


def sb_indices(n: int, block: float, B: int, rng) -> np.ndarray:
    """Stationary bootstrap index matrix (B, n), vectorized (Politis-Romano)."""
    p = 1.0 / max(block, 1.0)
    new = rng.random((B, n)) < p
    new[:, 0] = True
    starts = rng.integers(0, n, (B, n))
    pos = np.arange(n)
    last = np.maximum.accumulate(np.where(new, pos, 0), axis=1)
    base = np.take_along_axis(starts, last, axis=1)
    return (base + (pos - last)) % n


def _rng(seed):
    return seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)


def boot_dist(x, stat, B=2000, block=None, seed=0):
    """Bootstrap distribution of stat(sample) for rows of resampled x. stat takes (B, n) -> (B,)."""
    x = np.asarray(x, float)
    block = block_length(x) if block is None else block
    idx = sb_indices(len(x), block, B, _rng(seed))
    return stat(x[idx])


def mean_ci(x, B=2000, block=None, seed=0, null=0.0, scale=1.0) -> Est:
    """Mean with 95% CI and one-sided p for H0: true mean <= null."""
    x = np.asarray(x, float)
    d = boot_dist(x, lambda m: m.mean(axis=1), B, block, seed) * scale
    lo, hi = np.quantile(d, [0.025, 0.975])
    # one-sided p: bootstrap distribution of the recentered mean
    p = (1 + np.sum(d - d.mean() >= (x.mean() - null) * scale)) / (B + 1)
    return Est(float(x.mean() * scale), float(lo), float(hi), float(p))


def paired_diff(a, b, B=2000, block=None, seed=0) -> Est:
    """H0: mean(a-b) <= 0 (a not better than b). Paired block bootstrap on per-step differences."""
    return mean_ci(np.asarray(a, float) - np.asarray(b, float), B, block, seed)


def stat_ci(x, fn, B=2000, block=None, seed=0) -> Est:
    """95% CI for an arbitrary row-wise statistic fn((B,n))->(B,)."""
    x = np.asarray(x, float)
    d = boot_dist(x, fn, B, block, seed)
    return Est(float(fn(x[None, :])[0]), *map(float, np.quantile(d, [0.025, 0.975])))


# ---------- null-shuffle skill tests ----------
def null_pvalue(observed: float, null_samples) -> float:
    ns = np.asarray(null_samples, float)
    return float((1 + np.sum(ns >= observed)) / (len(ns) + 1))


def block_shuffle(x, block: int, rng) -> np.ndarray:
    """Permute whole blocks of x; destroys alignment with other series, keeps within-block autocorrelation."""
    x = np.asarray(x)
    n = len(x)
    nb = int(np.ceil(n / block))
    order = rng.permutation(nb)
    return np.concatenate([x[o * block:(o + 1) * block] for o in order])[:n]


def search_penalty(null_totals, N: int) -> float:
    """Bits to subtract for having tried N methods: 1-1/N quantile of the null total-skill distribution
    (the typical best of N worthless methods). Falls back to Gaussian sd*sqrt(2 ln N) when N exceeds the sample."""
    ns = np.asarray(null_totals, float)
    N = max(int(N), 1)
    if N == 1:
        return 0.0
    if N < len(ns):
        return float(np.quantile(ns, 1 - 1.0 / N))
    return float(ns.mean() + ns.std(ddof=1) * np.sqrt(2 * np.log(N)))


def best_of_n_pvalue(observed: float, null_totals, N: int, R=5000, seed=0) -> float:
    """Family-wise p: chance the best of N worthless methods reaches the observed total."""
    ns = np.asarray(null_totals, float)
    rng = _rng(seed)
    mx = ns[rng.integers(0, len(ns), (R, max(int(N), 1)))].max(axis=1)
    return float((1 + np.sum(mx >= observed)) / (R + 1))


def bh_adjust(p) -> np.ndarray:
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    adj = p[o] * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.minimum(adj, 1.0)
    return out


def rule_of_three(n: int, conf=0.95) -> float:
    """Upper bound on a rate after 0 events in n trials."""
    return float(-np.log(1 - conf) / max(n, 1))


def tost(diff_samples, delta: float, alpha=0.05) -> tuple[bool, float]:
    """Equivalence test on bootstrap draws of a difference: (equivalent, p). p = max of the two one-sided p-values."""
    d = np.asarray(diff_samples, float)
    p_low = float(np.mean(d <= -delta))   # H0: diff <= -delta
    p_up = float(np.mean(d >= delta))     # H0: diff >= +delta
    p = max(p_low, p_up)
    return p < alpha, p


def ks_uniform_p(p) -> float:
    from scipy.stats import kstest
    return float(kstest(np.asarray(p, float), "uniform").pvalue)


# ---------- finance ----------
def ann_return(r, periods=PERIODS) -> float:
    return float(np.mean(r) * periods)


def max_drawdown(r) -> float:
    c = np.cumsum(np.asarray(r, float))
    return float(np.max(np.maximum.accumulate(c) - c))  # in log-return units


def time_underwater(r) -> int:
    c = np.cumsum(np.asarray(r, float))
    peak = np.maximum.accumulate(c)
    run = best = 0
    for u in (c < peak):
        run = run + 1 if u else 0
        best = max(best, run)
    return int(best)


def cvar(r, q=0.05) -> float:
    r = np.asarray(r, float)
    k = max(1, int(np.floor(q * len(r))))
    return float(-np.sort(r)[:k].mean())


def risk_report(r, B=1000, seed=0, periods=PERIODS) -> dict:
    """Risk metrics with 95% bootstrap CIs. r = daily log returns after costs."""
    r = np.asarray(r, float)
    blk = block_length(r)
    rng = _rng(seed)
    idx = sb_indices(len(r), blk, B, rng)
    S = r[idx]
    out = {}

    def ci(name, point, dist):
        out[name] = Est(float(point), *map(float, np.quantile(dist, [0.025, 0.975])))

    ci("ann_vol", r.std(ddof=1) * np.sqrt(periods), S.std(axis=1, ddof=1) * np.sqrt(periods))
    c = np.cumsum(S, axis=1)
    ci("max_drawdown", max_drawdown(r), (np.maximum.accumulate(c, axis=1) - c).max(axis=1))
    k = max(1, int(0.05 * len(r)))
    ci("cvar95", cvar(r), -np.sort(S, axis=1)[:, :k].mean(axis=1))
    out["time_underwater_days"] = time_underwater(r)
    # forward simulation: fresh block-bootstrap paths of a given horizon
    for name, h in (("p_loss_1m", 30), ("p_loss_3m", 91), ("p_loss_1y", 365)):
        n = len(r)
        reps = int(np.ceil(h / n))
        ii = sb_indices(n, blk, 2000, rng)
        if reps > 1:
            ii = np.concatenate([ii] + [sb_indices(n, blk, 2000, rng) for _ in range(reps - 1)], axis=1)
        paths = r[ii[:, :h]]
        tot = paths.sum(axis=1)
        out[name] = float(np.mean(tot < 0))
    ii = sb_indices(len(r), blk, 2000, rng)
    reps = int(np.ceil(365 / len(r)))
    if reps > 1:
        ii = np.concatenate([ii] + [sb_indices(len(r), blk, 2000, rng) for _ in range(reps - 1)], axis=1)
    cp = np.cumsum(r[ii[:, :365]], axis=1)
    dd = (np.maximum.accumulate(np.maximum(cp, 0), axis=1) - cp).max(axis=1)
    out["p_50pct_drawdown_1y"] = float(np.mean(dd >= -np.log(0.5)))
    return out


def deflated_sharpe(sr_daily: float, n_trials: int, T: int, skew=0.0, kurt=3.0, sr_var_trials=None) -> float:
    """Probability that the true Sharpe exceeds the best expected from n_trials worthless methods (Bailey & Lopez de Prado)."""
    from scipy.stats import norm
    if n_trials <= 1:
        sr0 = 0.0
    else:
        v = sr_var_trials if sr_var_trials is not None else 1.0 / T
        g = 0.5772156649
        sr0 = np.sqrt(v) * ((1 - g) * norm.ppf(1 - 1.0 / n_trials) + g * norm.ppf(1 - 1.0 / (n_trials * np.e)))
    den = np.sqrt(max(1e-12, 1 - skew * sr_daily + (kurt - 1) / 4 * sr_daily ** 2))
    return float(norm.cdf((sr_daily - sr0) * np.sqrt(T - 1) / den))


def pbo_cscv(M, S=16) -> float:
    """Probability of backtest overfitting via combinatorially symmetric CV. M: (T, N) per-step scores of N methods."""
    from itertools import combinations
    M = np.asarray(M, float)
    T, N = M.shape
    S = min(S, T // 2 * 2)
    parts = np.array_split(np.arange(T), S)
    combos = list(combinations(range(S), S // 2))
    if len(combos) > 3000:
        rng = np.random.default_rng(0)
        combos = [combos[i] for i in rng.choice(len(combos), 3000, replace=False)]
    below = 0
    for c in combos:
        tr = np.concatenate([parts[i] for i in c])
        te = np.concatenate([parts[i] for i in range(S) if i not in c])
        best = int(np.argmax(M[tr].mean(axis=0)))
        te_scores = M[te].mean(axis=0)
        rank = (te_scores < te_scores[best]).sum() / (N - 1 if N > 1 else 1)
        below += rank <= 0.5
    return float(below / len(combos))


def log_growth(pred_edge, outcome_ret, kelly_frac=0.25) -> np.ndarray:
    """Per-step log growth of a fractional-Kelly position f = kelly_frac*edge (clipped to [0,1] long-only spot)."""
    f = np.clip(kelly_frac * np.asarray(pred_edge, float), 0, 1)
    return np.log1p(f * np.expm1(np.asarray(outcome_ret, float)))


def headline(ev: Est, p_after_search: float, N: int, cvar_pct: float, p_loss_1y: float) -> str:
    return (f"EV {ev.point*100:+.1f}% per year (95% CI {ev.lo*100:.1f}% to {ev.hi*100:.1f}%), "
            f"p = {p_after_search:.3g} that this beats chance after {N} tries; "
            f"worst-5%-day average loss {cvar_pct*100:.1f}%; probability of losing money over a year {p_loss_1y*100:.0f}%.")


class FixedShareMixture:
    """Fixed-share mixture of predict functions (Herbster & Warmuth). Prequential: weights update from each coin's realized bin
    at its next call. w_k *= 2^(-bits paid), then alpha of the total weight is mixed back uniformly."""

    def __init__(self, fns, alpha=0.01):
        self.fns, self.alpha, self.state = fns, alpha, {}

    def __call__(self, h):
        K = len(self.fns)
        st = self.state.get(h.coin)
        if st is not None and st["t"] == h.t - 1:
            b = int(np.searchsorted(h.edges, np.log(h.close[-1] / h.close[-2])))
            bits = -np.log2(np.array([p[b] for p in st["ps"]]))
            w = st["w"] * 2.0 ** (-(bits - bits.min()))
            w = w / w.sum()
            w = (1 - self.alpha) * w + self.alpha / K
        else:
            w = np.ones(K) / K
        ps = [np.asarray(f(h), float) for f in self.fns]
        self.state[h.coin] = dict(t=h.t, w=w, ps=ps)
        p = sum(wi * pi for wi, pi in zip(w, ps))
        return p
