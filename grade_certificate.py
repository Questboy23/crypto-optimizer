"""Gate 4: grade the doneness certificate on planted-signal markets with known optimum (oracle bits)."""
import json, sys, numpy as np, optuna
from scipy.stats import norm
import multiprocessing as mp
import bench, gates
import engine.stats as S
optuna.logging.set_verbosity(optuna.logging.WARNING)
SIG, T, C = 0.03, 1500, 2


def make(mu, seed):
    m = gates.planted_mkt(mu, SIG, T=T, C=C, seed=seed)
    edges, means = bench.bins_for(m, np.datetime64("2010-01-01") + 500)
    return m, edges, means


def fn_for(P, edges):
    mu_h, sg = P
    def pr(loc): return np.diff(np.r_[0.0, norm.cdf(edges, loc, sg), 1.0])
    return lambda h: pr(mu_h * (1.0 if h.volume[-1] > 1e6 else -1.0))


def bits_daily(P, m, edges, means):
    r = bench.run(fn_for(P, edges), m, edges, means, 500, T - 2)
    return (bench._bits(r["base"], r["O"], r["elig"]) - bench._bits(r["P"], r["O"], r["elig"])).sum(axis=1)


def one_search(args):
    mu, dseed, seed, trials = args      # toy problems: same data, different search seeds
    m, edges, means = make(mu, dseed)
    rng = np.random.default_rng(seed)
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=8))
    best, best_P, bestd, imp = -1e18, None, None, []
    for i in range(trials):
        tr = st.ask(); P = (tr.suggest_float("mu", 0.0, 0.1), tr.suggest_float("sg", 0.005, 0.1, log=True))
        d = bits_daily(P, m, edges, means); tot = d.sum(); st.tell(tr, tot)
        imp.append(tot > best + 0.01 * max(abs(best), 1.0) or best < -1e17)   # meaningful improvement: >1% of the leader
        if tot > best: best, best_P, bestd = tot, P, d
    rad = False
    for _ in range(10):                                   # radical round: uniform random programs must not beat the leader
        P = (rng.uniform(0, 0.1), float(np.exp(rng.uniform(np.log(0.005), np.log(0.1)))))
        rad |= bits_daily(P, m, edges, means).sum() > best
    oracle = bits_daily((mu, SIG), m, edges, means).sum()
    return dict(best=float(best), daily=bestd.tolist(), fail_last30=1 - float(np.mean(imp[-30:])), radical_beaten=bool(rad), oracle=float(oracle))


def verdict(runs, B=1000):
    bests = [r["best"] for r in runs]
    delta = 0.10 * abs(max(bests))
    delta = max(delta, 1.0)
    n = min(len(r["daily"]) for r in runs)
    eq, wide = True, False
    for i in range(4):
        for j in range(i + 1, 4):
            d = np.array(runs[i]["daily"][:n]) - np.array(runs[j]["daily"][:n])
            dr = S.boot_dist(d, lambda m: m.sum(axis=1), B=B, seed=i * 4 + j)
            ok, _ = S.tost(dr, delta); eq &= ok
            lo, hi = np.quantile(dr, [0.05, 0.95]); wide |= (hi - lo) > 2 * delta
    stag = all(r["fail_last30"] > 0.9 for r in runs)
    rad = not any(r["radical_beaten"] for r in runs)
    return "done" if (eq and stag and rad) else ("data_limited" if wide and not eq else "not_done")


def instance(args):
    mu, k, trials = args
    runs = [one_search((mu, 1000 * k, 1000 * k + s, trials)) for s in range(4)]
    v = verdict(runs)
    near = [r["best"] >= 0.9 * r["oracle"] if r["oracle"] > 1 else True for r in runs]  # within 10% of the true optimum
    return dict(mu=mu, verdict=v, all_near=bool(all(near)), frac_near=float(np.mean(near)))


if __name__ == "__main__":
    n_per, trials = int(sys.argv[1]), int(sys.argv[2])
    mus = [gates.mu_for_bits(b, SIG) if b else 0.0 for b in (0.0, 0.05, 0.1)]
    jobs = [(mus[i % 3], i, trials) for i in range(3 * n_per)]
    with mp.Pool(4) as pool:
        out = pool.map(instance, jobs)
    done = [o for o in out if o["verdict"] == "done"]
    good = sum(o["all_near"] for o in done)
    conv = [o for o in out if o["all_near"]]
    from statsmodels.stats.proportion import proportion_confint
    prec_ci = proportion_confint(good, len(done), method="wilson") if done else (float("nan"),) * 2
    rec = sum(o["verdict"] == "done" for o in conv)
    rec_ci = proportion_confint(rec, len(conv), method="wilson") if conv else (float("nan"),) * 2
    res = dict(n=len(out), verdicts={v: sum(o["verdict"] == v for o in out) for v in ("done", "not_done", "data_limited")},
               precision=(good / len(done) if done else None), precision_ci=prec_ci, n_done=len(done),
               recall=(rec / len(conv) if conv else None), recall_ci=rec_ci, n_converged=len(conv), instances=out)
    json.dump(res, open("results/gate4.json", "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if k != "instances"}, default=float))
