"""GPU=none path: Optuna over the seed program's numeric params, wrapped by the Thompson-sampling optimizer-optimizer.
Styles: tpe (Optuna), tweak (small perturbation of the incumbent), combine (blend two archive programs), radical (outside the box)."""
import json, sys, time, numpy as np, optuna
from scipy.stats import t as tdist
import bench

optuna.logging.set_verbosity(optuna.logging.WARNING)
BOX = {"win": (10, 365, int), "mom": (3, 120, int), "tilt": (-0.5, 0.5, float), "df": (2.5, 30.0, float)}
STYLES = ["tpe", "tweak", "combine", "radical"]
BUCKETS = ["<50%", "50-90%", ">90%"]


def make_fn(P):
    def predict(h):
        c = h.close
        r = np.diff(np.log(c[-(P["win"] + 1):]))
        sd = r.std() if len(r) > 5 else 0.05
        m = np.diff(np.log(c[-(P["mom"] + 1):])).mean()
        loc = P["tilt"] * np.tanh(m / max(sd, 1e-6)) * sd
        scale = max(sd, 1e-4) * np.sqrt((P["df"] - 2) / P["df"])
        cdf = tdist.cdf((h.edges - loc) / scale, P["df"])
        return np.diff(np.r_[0.0, cdf, 1.0])
    return predict


def clip(P):
    return {k: (type(lo)(min(max(P[k], lo), hi)) if t is float else int(round(min(max(P[k], lo), hi)))) for k, (lo, hi, t) in BOX.items()}


def propose(style, rng, best, archive, study):
    if style == "tpe":
        return None  # Optuna decides
    if style == "tweak":
        return clip({k: best[k] + rng.normal(0, 0.1 * (hi - lo)) for k, (lo, hi, _) in BOX.items()})
    if style == "combine" and len(archive) >= 2:
        a, b = archive[0][1], archive[int(rng.integers(1, min(len(archive), 6)))][1]
        w = rng.uniform(0.2, 0.8)
        return clip({k: w * a[k] + (1 - w) * b[k] for k in BOX})
    # radical: uniform over the box, with one assumption inverted (tilt sign or an extreme window)
    P = {k: rng.uniform(lo, hi) for k, (lo, hi, _) in BOX.items()}
    j = int(rng.integers(0, 3))
    if j == 0: P["tilt"] = -best["tilt"]
    if j == 1: P["win"] = rng.choice([10, 365])
    if j == 2: P["df"] = rng.choice([2.5, 30.0])
    return clip(P)


def stage_bucket(fails):
    return 0 if fails < 0.5 else (1 if fails <= 0.9 else 2)


def run_group(args):
    g, seed, n_batches, batch = args
    rng = np.random.default_rng(seed)
    ctx = bench.context("dev", g)
    sd = bench.seed_sd(g)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=10))
    ts = {b: {s: dict(a=1.0, b=1.0, gain=1.0, cost=1.0, n=0) for s in STYLES} for b in range(3)}  # Beta posteriors per stage
    best_score, best_P, best_daily, archive, history, log = -1e18, dict(bench_seed=1), None, [], [], []
    base = {"win": 60, "tilt": 0.05, "mom": 20, "df": 4.0}
    pending = [("seed", base)]
    t_start = time.time()

    def evaluate(P, style, batch_i):
        nonlocal best_score, best_P, best_daily
        t0 = time.time()
        o = bench.score(make_fn(P), group=g, resid=False, seed_sd=sd, fn_hash=json.dumps(P, sort_keys=True), ctx=ctx)
        dt = time.time() - t0
        improved = o["total_bits"] > best_score + 1e-9   # compare total bits: the penalty depends on N, which grows during the run
        gain = max(o["total_bits"] - best_score, 0.0) if best_score > -1e17 else 0.0
        if improved:
            best_score, best_P, best_daily = o["total_bits"], dict(P), o["_saved_daily"]
        archive.append((o["total_bits"], dict(P)))
        archive.sort(key=lambda x: -x[0])
        history.append(improved)
        log.append(dict(batch=batch_i, style=style, P=P, total_bits=o["total_bits"], net_bits=o["combined_score"], N=o["N"],
                        improved=bool(improved), p_skill=o["p_skill"], direction_bits=o["direction_bits"], scale_bits=o["scale_bits"], secs=dt))
        return improved, gain, dt

    evaluate(base, "seed", -1)
    for bi in range(n_batches):
        last = history[-batch:]
        fails = 1 - float(np.mean(last)) if last else 0.0
        bucket = stage_bucket(fails)
        draws = {s: rng.beta(ts[bucket][s]["a"], ts[bucket][s]["b"]) * ts[bucket][s]["gain"] / ts[bucket][s]["cost"] for s in STYLES}
        style = max(draws, key=draws.get)
        st = ts[bucket][style]
        for _ in range(batch):
            P = propose(style, rng, best_P, archive, study)
            tr = study.ask() if P is None else (study.enqueue_trial(P, skip_if_exists=False), study.ask())[1]
            if P is None:
                P = clip({k: (tr.suggest_float(k, lo, hi) if t is float else tr.suggest_int(k, lo, hi)) for k, (lo, hi, t) in BOX.items()})
            else:
                for k, (lo, hi, t) in BOX.items():
                    (tr.suggest_float if t is float else tr.suggest_int)(k, lo, hi)
            improved, gain, dt = evaluate(P, style, bi)
            study.tell(tr, log[-1]["total_bits"])
            st["a" if improved else "b"] += 1
            st["gain"] += gain; st["cost"] += dt / 100.0; st["n"] += 1
    return dict(group=g, seed=seed, best_P=best_P, best_net=best_score, log=log, ts=ts, daily=best_daily.tolist(), secs=time.time() - t_start)


if __name__ == "__main__":
    import multiprocessing as mp
    n_batches, batch = int(sys.argv[1]), int(sys.argv[2])
    with mp.Pool(4) as pool:
        res = pool.map(run_group, [(g, 100 + g, n_batches, batch) for g in range(4)])
    for r in res:
        json.dump(r, open(f"results/search_g{r['group']}.json", "w"))
    json.dump({f"g{r['group']}": r["ts"] for r in res}, open("strategies.json", "w"))
    print(json.dumps({f"g{r['group']}": (round(r["best_net"], 1), r["best_P"]) for r in res}))
