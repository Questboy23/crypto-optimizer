"""Kill gates 1-3 and the required tests. `python gates.py [1|2|3|all]` prints one JSON line."""
import json, sys, tempfile, os
import numpy as np
from scipy.stats import norm
import bench, data
import engine.stats as S


# ---------- gate 1: planted signal ----------
def planted_mkt(mu, sigma=0.03, T=3200, C=5, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.choice([-1.0, 1.0], size=(T, C))
    r = np.zeros((T, C)); r[1:] = mu * z[:-1] + sigma * rng.standard_normal((T - 1, C))
    close = np.exp(np.cumsum(r, axis=0)) * 100
    dv = 1e6 * (1 + 0.5 * z) * np.ones((T, C))
    btc = np.exp(np.cumsum(0.02 * rng.standard_normal(T)))
    dates = np.datetime64("2010-01-01") + np.arange(T)
    return bench.Mkt(dates, close, dv, btc, [f"c{i}" for i in range(C)])


def mi_bits(mu, sigma):
    x = np.linspace(-12 * sigma - abs(mu), 12 * sigma + abs(mu), 40001)
    a, b = norm.pdf(x, mu, sigma), norm.pdf(x, -mu, sigma)
    mix = 0.5 * (a + b)
    dx = x[1] - x[0]
    f = lambda p, q: np.sum(np.where(p > 1e-300, p * np.log2(np.maximum(p, 1e-300) / np.maximum(q, 1e-300)), 0)) * dx
    return 0.5 * f(a, mix) + 0.5 * f(b, mix)


def mu_for_bits(target, sigma=0.03):
    lo, hi = 0.0, 5 * sigma
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if mi_bits(mid, sigma) < target else (lo, mid)
    return lo


def gate1():
    sigma, res = 0.03, {}
    for target in (0.0, 0.05, 0.1):
        mu = mu_for_bits(target, sigma) if target else 0.0
        m = planted_mkt(mu, sigma)
        edges, means = bench.bins_for(m, np.datetime64("2010-01-01") + 1000)
        ctx = (m, m, edges, means, edges, means, 1000, len(m.dates) - 2)

        def cdfp(loc):
            return np.diff(np.r_[0.0, norm.cdf(edges, loc, sigma), 1.0])

        def oracle(h):
            return cdfp(mu * (1.0 if h.volume[-1] > 1e6 else -1.0))

        def marginal(h):
            return 0.5 * (cdfp(mu) + cdfp(-mu))
        ro, rm = (bench.run(f, m, edges, means, 1000, len(m.dates) - 2) for f in (oracle, marginal))
        n = ro["elig"].sum()
        rec = float(((bench._bits(rm["P"], rm["O"], rm["elig"]) - bench._bits(ro["P"], ro["O"], ro["elig"])).sum()) / n)
        sk = bench.score(oracle, ctx=ctx, resid=False, register_eval=False, seed_sd=None)
        nz = bench.score(marginal, ctx=ctx, resid=False, register_eval=False, seed_sd=None)
        res[str(target)] = dict(recovered=round(rec, 4), p_oracle=round(sk["p_skill"], 4), p_marginal=round(nz["p_skill"], 4))
    ok = all(abs(res[str(t)]["recovered"] - t) <= 0.2 * t for t in (0.05, 0.1)) and abs(res["0.0"]["recovered"]) < 0.005 and res["0.0"]["p_marginal"] > 0.05 \
        and res["0.05"]["p_oracle"] <= 0.05
    return ok, res


# ---------- gate 2: 100 signal-free random programs ----------
def random_program(seed):
    from scipy.stats import t as tdist
    rng0 = np.random.default_rng(seed)
    mult, tilt = rng0.uniform(0.8, 1.3), rng0.uniform(0.0, 0.3)
    off = int(rng0.integers(1, 10 ** 6))

    def predict(h):
        r = np.diff(np.log(h.close[-366:]))
        sd = max(r.std(), 1e-4) * mult * np.sqrt(0.5)
        u = np.random.default_rng(off + h.t * 31 + len(h.coin)).standard_normal()
        cdf = tdist.cdf((h.edges - tilt * u * sd) / sd, 4.0)
        return np.diff(np.r_[0.0, cdf, 1.0])
    return predict


def gate2(n=100, group=0):
    sd = bench.seed_sd(group)
    ctx = bench.context("dev", group)
    ps = [bench.score(random_program(s), group=group, resid=False, register_eval=False, seed_sd=sd, ctx=ctx)["p_skill"] for s in range(n)]
    frac = float(np.mean(np.array(ps) <= 0.05))
    return frac <= 0.08, dict(frac_sig=round(frac, 3), ks_p=round(S.ks_uniform_p(ps), 3), median_p=round(float(np.median(ps)), 3))


# ---------- gate 3: look-ahead ----------
CHEATS = {
    "at_next": "import numpy as np\ndef predict(h):\n    h.at(h.t + 1)\n    return np.ones(21)/21\n",
    "index_next": "import numpy as np\ndef predict(h):\n    h.close[h.t + 1]\n    return np.ones(21)/21\n",
    "base_attr": "import numpy as np\ndef predict(h):\n    assert h.close.base is not None\n    return np.ones(21)/21\n",
    "frame": "import numpy as np, inspect\ndef predict(h):\n    return np.ones(21)/21\n",
}


def gate3():
    out = {}
    for k, src in CHEATS.items():
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(src)
        r = bench.evaluate(f.name, group=0, resid=False)
        os.unlink(f.name)
        out[k] = r.get("error", "NOT BLOCKED")[:40]
    ok = all(v != "NOT BLOCKED" for v in out.values())
    return ok, out


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    res, detail = {}, {}
    for name, fn in (("gate1", gate1), ("gate2", gate2), ("gate3", gate3)):
        if which in ("all", name[-1]):
            ok, d = fn()
            res[name], detail[name] = ("pass" if ok else "FAIL"), d
    print(json.dumps(res))
    print(json.dumps(detail, default=str), file=sys.stderr)
