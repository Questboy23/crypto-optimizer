"""Benchmark: scores predict(history)->21-bin probs for next-day log return in bits. OpenEvolve-compatible evaluate()."""
import hashlib, importlib.util, json, os, re, sqlite3, time
import numpy as np, polars as pl
from scipy.stats import t as tdist
import data, engine.stats as S

VERSION = "v1"
K = 21
FEE, SLIP = 0.0080, 0.0005          # per side, Kraken Pro tier 1 taker + slippage
ONEWAY = FEE + SLIP
CLIP = 1e-6
EVAL_START = np.datetime64("2019-01-01")
DF = 4.0
ROOT = os.path.dirname(os.path.abspath(__file__))
REG = os.path.join(ROOT, "data", "registry.db")
NULLC = os.path.join(ROOT, "data", "null_cache.json")
DENY = re.compile(r"\binspect\b|_getframe|f_back|f_locals|\bgc\.|__import__|\bopen\(|read_parquet|read_csv|\bsubprocess\b|\bos\.|\bsys\.")


class LeakError(Exception):
    pass


class History:
    """Past-only view. Holds copies of data up to and including day t; nothing later is reachable."""
    __slots__ = ("coin", "t", "close", "volume", "btc_close", "dates", "edges")

    def __init__(self, coin, close, volume, btc, dates, edges):
        self.coin, self.close, self.volume, self.btc_close, self.dates, self.edges = coin, close, volume, btc, dates, edges
        self.t = len(close) - 1

    def at(self, i):
        if i > self.t:
            raise LeakError(f"index {i} is after t={self.t}")
        return self.close[i]


# ---------------- market structures ----------------
class Mkt:
    def __init__(self, dates, close, dv, btc, coins):
        self.dates, self.close, self.dv, self.btc, self.coins = dates, close, dv, btc, coins
        with np.errstate(all="ignore"):
            r = np.log(close[1:] / close[:-1])
        self.R = np.vstack([np.full((1, close.shape[1]), np.nan), r])  # R[t] = return from t-1 to t


def make_mkt(px, vol, coins, split):
    cols = [c for c in coins if c in px.columns]
    dates = px["date"].to_numpy().astype("datetime64[D]")
    close = px.select(cols).to_numpy().astype(float)
    dv = vol.select(cols).to_numpy().astype(float)
    btc = px["XBT"].to_numpy().astype(float)
    return Mkt(dates, close, dv, btc, cols)


def residual_mkt(m: Mkt, win=90):
    """Residual-return 'price' series: r - beta_{t-1} * btc_r, beta from trailing win days (causal)."""
    with np.errstate(all="ignore"):
        mr = np.r_[np.nan, np.log(m.btc[1:] / m.btc[:-1])]
    R = m.R
    T, C = R.shape
    beta = np.zeros((T, C))
    import pandas as pd
    mser = pd.Series(mr)
    for j in range(C):
        rs = pd.Series(R[:, j])
        cov = rs.rolling(win, min_periods=30).cov(mser)
        var = mser.rolling(win, min_periods=30).var()
        beta[:, j] = (cov / var).to_numpy()
    b_prev = np.vstack([np.full((1, C), np.nan), beta[:-1]])
    res = R - np.nan_to_num(b_prev) * mr[:, None]
    res[np.isnan(R)] = np.nan
    price = np.exp(np.nancumsum(np.nan_to_num(res), axis=0))
    price[np.isnan(m.close)] = np.nan
    return Mkt(m.dates, price, m.dv, m.btc, m.coins)


# ---------------- bins ----------------
def bins_for(m: Mkt, dev_end):
    """Equal-mass edges and per-bin mean returns from development returns only."""
    mask = m.dates < np.datetime64(dev_end, "D")
    r = m.R[mask]
    r = r[np.isfinite(r) & (np.abs(r) > 0)]
    edges = np.quantile(r, np.arange(1, K) / K)
    idx = np.searchsorted(edges, r)
    means = np.array([r[idx == k].mean() for k in range(K)])
    return edges, means


def tprobs(scale, edges):
    """Student-t(df=DF) bin probabilities, zero mean, given scale array (...,)."""
    c = tdist.cdf(edges[None, :] / np.asarray(scale)[:, None], DF)
    p = np.diff(np.hstack([np.zeros((len(c), 1)), c, np.ones((len(c), 1))]), axis=1)
    p = np.clip(p, CLIP, None)
    return p / p.sum(axis=1, keepdims=True)


def clip_probs(p):
    p = np.asarray(p, float)
    if p.shape != (K,) or not np.all(np.isfinite(p)) or p.min() < 0 or p.sum() <= 0:
        raise ValueError("predict must return 21 finite non-negative probabilities")
    p = np.clip(p / p.sum(), CLIP, None)
    return p / p.sum()


# ---------------- core run ----------------
def run(fn, m: Mkt, edges, means, t_lo, t_hi, group_ok=None):
    """Run fn over steps t in [t_lo, t_hi); target is return at t+1. Returns arrays of shape (steps, C, ...)."""
    T, C = m.close.shape
    steps = np.arange(t_lo, t_hi)
    # trailing-365 std baseline (causal: uses returns up to and including t)
    import pandas as pd
    sd = pd.DataFrame(m.R).rolling(365, min_periods=60).std().to_numpy()
    first = np.argmax(np.isfinite(m.close), axis=0)
    cv = np.cumsum(np.nan_to_num(m.dv), axis=0)
    elig = np.zeros((len(steps), C), bool)
    Pm = np.full((len(steps), C, K), 1.0 / K, np.float32)
    nxt = m.R[steps + 1]
    for i, t in enumerate(steps):
        v30 = cv[t] - cv[max(t - 30, 0)]
        elig[i] = (steps[i] - first >= 120) & (v30 > 0) & np.isfinite(sd[t]) & np.isfinite(nxt[i])
    for j in range(C):
        for i, t in enumerate(steps):
            if not elig[i, j]:
                continue
            f0 = first[j]
            h = History(m.coins[j], m.close[f0: t + 1, j].copy(), m.dv[f0: t + 1, j].copy(), m.btc[f0: t + 1].copy(), m.dates[f0: t + 1].copy(), edges)
            Pm[i, j] = clip_probs(fn(h))
    scale = sd[steps] * np.sqrt((DF - 2) / DF)
    base = np.full((len(steps), C, K), 1.0 / K, np.float32)
    ii, jj = np.nonzero(elig)
    base[ii, jj] = tprobs(scale[ii, jj], edges)
    O = np.searchsorted(edges, nxt)  # outcome bin
    O[~elig] = 0
    # method's predicted mean and std (bin means) -> symmetric-t reference with same std
    mu = (Pm * means).sum(axis=2)
    sdp = np.sqrt(np.maximum((Pm * (means[None, None, :] - mu[..., None]) ** 2).sum(axis=2), 1e-12))
    sym = np.full_like(base, 1.0 / K)
    sym[ii, jj] = tprobs(sdp[ii, jj] * np.sqrt((DF - 2) / DF), edges)
    return dict(P=Pm, base=base, sym=sym, O=O, elig=elig, mu=mu, nxt=nxt, steps=steps)


def _bits(P, O, elig):
    lp = np.log2(np.take_along_axis(P, O[..., None], axis=2)[..., 0])
    return np.where(elig, -lp, 0.0)


def daily(x, elig):
    return x.sum(axis=1)


def null_totals(r, nshuf=100, block=10, seed=0):
    """Total bits_saved with outcomes block-shuffled across days (keeps within-block autocorrelation)."""
    rng = np.random.default_rng(seed)
    n = len(r["steps"])
    lm, lb = np.log2(r["P"]), np.log2(r["base"])
    out = np.empty(nshuf)
    for k in range(nshuf):
        perm = S.block_shuffle(np.arange(n), block, rng)
        O2 = r["O"][perm]
        ok = r["elig"] & r["elig"][perm]
        a = np.take_along_axis(lm, O2[..., None], axis=2)[..., 0]
        b = np.take_along_axis(lb, O2[..., None], axis=2)[..., 0]
        out[k] = np.where(ok, a - b, 0.0).sum()
    return out


def trade_returns(r):
    """Long-only spot: hold coin next day when predicted mean exceeds round-trip cost. Equal weight over eligible coins."""
    pos = (r["mu"] > 2 * ONEWAY) & r["elig"]
    prev = np.vstack([np.zeros((1, pos.shape[1]), bool), pos[:-1]])
    cost = ONEWAY * np.abs(pos.astype(float) - prev.astype(float))
    gross = np.where(pos, np.nan_to_num(r["nxt"]), 0.0)
    n = np.maximum(r["elig"].sum(axis=1), 1)
    return (gross - cost).sum(axis=1) / n, pos.sum() / max(r["elig"].sum(), 1)


# ---------------- registry ----------------
def _reg():
    os.makedirs(os.path.dirname(REG), exist_ok=True)
    c = sqlite3.connect(REG)
    c.execute("create table if not exists evals(id integer primary key, ts real, version text, hash text, grp int, split text)")
    return c


def register(h, grp, split):
    c = _reg()
    c.execute("insert into evals(ts,version,hash,grp,split) values(?,?,?,?,?)", (time.time(), VERSION, h, grp, split))
    c.commit()
    n = c.execute("select count(*) from evals where version=?", (VERSION,)).fetchone()[0]
    c.close()
    return n


def n_evals():
    c = _reg(); n = c.execute("select count(*) from evals where version=?", (VERSION,)).fetchone()[0]; c.close(); return n


# ---------------- program loading ----------------
def load_fn(path):
    src = open(path).read()
    bad = DENY.search(src)
    if bad:
        raise ValueError(f"forbidden token in program: {bad.group(0)}")
    spec = importlib.util.spec_from_file_location("prog_" + hashlib.md5(src.encode()).hexdigest()[:8], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.predict, hashlib.sha256(src.encode()).hexdigest()[:16]


# ---------------- evaluation ----------------
_CTX = {}


def context(split="dev", group=0):
    key = (split, group)
    if key in _CTX:
        return _CTX[key]
    dev_end, hold = data.split_dates()
    px, vol = data.panel(allow_holdout=(split == "holdout"))
    coins = data.coin_groups()[group] if group >= 0 else data.coin_groups()[0]
    m = make_mkt(px, vol, coins, split)
    mr = residual_mkt(m)
    # bins from development data only (groups' own coins plus pooled would leak across groups; use this group's dev returns)
    edges, means = bins_for(m, dev_end)
    redges, rmeans = bins_for(mr, dev_end)
    d = m.dates
    if split == "dev":
        lo = int(np.searchsorted(d, EVAL_START)); hi = int(np.searchsorted(d, np.datetime64(dev_end, "D"))) - 1
    else:
        lo = int(np.searchsorted(d, np.datetime64(hold, "D"))) - 1; hi = len(d) - 1
    _CTX[key] = (m, mr, edges, means, redges, rmeans, lo, hi)
    return _CTX[key]


def score(fn, split="dev", group=0, resid=True, nshuf=100, seed_sd=None, fn_hash="-", register_eval=True, ctx=None):
    m, mr, edges, means, redges, rmeans, lo, hi = ctx or context(split, group)
    r = run(fn, m, edges, means, lo, hi)
    bm, bb, bs = _bits(r["P"], r["O"], r["elig"]), _bits(r["base"], r["O"], r["elig"]), _bits(r["sym"], r["O"], r["elig"])
    saved_d = (bb - bm).sum(axis=1)            # per-day bits saved vs baseline
    dir_d = (bs - bm).sum(axis=1)
    nulls = null_totals(r, nshuf)
    tot = float(saved_d.sum())
    N = register(fn_hash, group, split) if register_eval else max(n_evals(), 1)
    sd_null = seed_sd if seed_sd is not None else float(nulls.std(ddof=1))
    pen = float(sd_null * np.sqrt(2 * np.log(max(N, 1)))) if N > 1 else 0.0
    ret, frac = trade_returns(r)
    ev = S.mean_ci(ret, B=1000, scale=365)
    rr = S.risk_report(ret, B=300) if ret.std() > 0 else None
    tb = S.mean_ci(saved_d, B=1000)
    out = dict(combined_score=tot - pen, net_bits=tot - pen, total_bits=tot, penalty=pen, N=N,
               bits_per_step=tot / max(r["elig"].sum(), 1), n_pred=int(r["elig"].sum()), n_days=len(saved_d),
               total_bits_ci=(tb.lo * len(saved_d), tb.hi * len(saved_d)),
               direction_bits=float(dir_d.sum()), scale_bits=float(saved_d.sum() - dir_d.sum()),
               p_skill=S.null_pvalue(tot, nulls), sd_null=sd_null,
               ev_ann=ev.point, ev_ci=(ev.lo, ev.hi), p_ev=ev.p, trade_frac=float(frac),
               max_drawdown=rr["max_drawdown"].point if rr else 0.0, cvar95=rr["cvar95"].point if rr else 0.0,
               p_loss_1y=rr["p_loss_1y"] if rr else 0.0)
    if resid:
        rres = run(fn, mr, redges, rmeans, lo, hi)
        sres = (_bits(rres["base"], rres["O"], rres["elig"]) - _bits(rres["P"], rres["O"], rres["elig"])).sum()
        out["resid_bits"] = float(sres)
    out["_saved_daily"] = saved_d
    out["_dir_daily"] = dir_d
    out["_ret"] = ret
    return out


def seed_sd(group, split="dev"):
    key = f"{VERSION}|{group}|{split}"
    cache = json.load(open(NULLC)) if os.path.exists(NULLC) else {}
    if key not in cache:
        fn, h = load_fn(os.path.join(ROOT, "seed_program.py"))
        m, mr, edges, means, redges, rmeans, lo, hi = context(split, group)
        cache[key] = float(null_totals(run(fn, m, edges, means, lo, hi)).std(ddof=1))
        os.makedirs(os.path.dirname(NULLC), exist_ok=True)
        json.dump(cache, open(NULLC, "w"))
    return cache[key]


def evaluate(program_path, group=None, split="dev", resid=None):
    """OpenEvolve entry point."""
    group = int(os.environ.get("BENCH_GROUP", 0)) if group is None else group
    resid = (os.environ.get("BENCH_RESID", "1") == "1") if resid is None else resid
    try:
        fn, h = load_fn(program_path)
        out = score(fn, split, group, resid=resid, seed_sd=seed_sd(group, split), fn_hash=h)
    except Exception as e:  # failed program: worst score, still counted by the registry
        register("fail", group if group is not None else 0, split)
        return {"combined_score": -1e6, "error": f"{type(e).__name__}: {e}"[:300]}
    out.pop("_saved_daily", None)
    return out
