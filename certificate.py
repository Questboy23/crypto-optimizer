"""Doneness certificate over 4 independent searches (one per coin group). Verdict: done | not_done | data_limited."""
import json, sys, numpy as np
import engine.stats as S
import optimize, bench


def load(n=4, mode="optuna"):
    if mode == "optuna":
        return [json.load(open(f"results/search_g{g}.json")) for g in range(n)]
    out = []   # LLM mode: rebuild the same schema from the evaluation log, the cache and the OpenEvolve best programs
    for g in range(n):
        recs = [json.loads(x) for x in open(f"results/llm_g{g}.jsonl")]
        meta = json.load(open(f"results/llm_meta_g{g}.json"))
        full = [r for r in recs if not r.get("dup")]
        run_max, log = -1e18, []
        for r in full:
            ok = (not r.get("error")) and r["total_bits"] > run_max + 1e-9
            run_max = max(run_max, r["total_bits"]) if not r.get("error") else run_max
            log.append(dict(total_bits=r["total_bits"], improved=bool(ok), hash=r["hash"], style=r.get("style")))
        best = max((r for r in full if not r.get("error")), key=lambda r: r["total_bits"])
        daily = np.load(f"results/cache/g{g}/{best['hash']}.npy").tolist()
        out.append(dict(group=g, log=log, daily=daily, best_hash=best["hash"], best_P=None, meta=meta, best_path=meta["best_path"]))
    return out


def fn_of(r):
    if r.get("best_P") is not None:
        return optimize.make_fn(r["best_P"])
    return bench.load_fn(r["best_path"])[0]


def final_net(r, N_final):
    """Recompute net bits of each trial's best using the final registry N."""
    sd = bench.seed_sd(r["group"])
    best = max(r["log"], key=lambda x: x["total_bits"])
    return best["total_bits"] - sd * np.sqrt(2 * np.log(max(N_final, 2))), best


def certificate(res, reveal=None, radical=None, B=2000):
    N = bench.n_evals()
    nets = [final_net(r, N)[0] for r in res]
    lead = int(np.argmax(nets))
    delta = 0.10 * abs(nets[lead])
    daily = [np.array(r["daily"]) for r in res]
    n = min(len(d) for d in daily)
    pairs, eq_all, wide = {}, True, False
    for i in range(4):
        for j in range(i + 1, 4):
            d = (daily[i][:n] - daily[j][:n])
            draws = S.boot_dist(d, lambda m: m.sum(axis=1), B=B, seed=i * 10 + j)   # difference in total bits
            lo90, hi90 = np.quantile(draws, [0.05, 0.95])
            ok, p = S.tost(draws, delta)
            pairs[f"{i}-{j}"] = dict(diff=float(d.sum()), ci90=(float(lo90), float(hi90)), tost_p=p, equivalent=bool(ok))
            eq_all &= ok
            wide |= (hi90 - lo90) > 2 * delta   # CI wider than the equivalence band: cannot tell either way
    # condition 2: >90% of the last 30 proposals failed to improve (per search)
    def meaningful(log):   # an improvement counts only if it beats the running best by more than 1% (micro-gains do not reset stagnation)
        best, out = -1e18, []
        for x in log:
            out.append(x["total_bits"] > best + 0.01 * max(abs(best), 1.0) or best < -1e17)
            best = max(best, x["total_bits"])
        return out
    fails = [1 - float(np.mean(meaningful(r["log"])[-30:])) for r in res]
    stag = all(f > 0.90 for f in fails)
    rot = S.rule_of_three(30)
    cond3 = radical
    verdict = "done" if (eq_all and stag and (cond3 is True)) else ("data_limited" if wide and not eq_all else "not_done")
    return dict(verdict=verdict, N=N, nets=nets, delta=delta, pairs=pairs, stagnation_fail_rates=fails,
                rule_of_three_upper_bound=rot, radical_round_passed=cond3, reveal=reveal)


def radical_round(res, n=10, seed=7):
    if res[0].get("meta"):   # LLM mode: the radical round is the final forced outside-the-box batch recorded by evolve.py
        rb = [[b for b in r["meta"]["batches"] if b.get("radical")] for r in res]
        if not all(rb):   # radical batch hung (Ollama stall) and was capped by the time box: condition 3 is untested
            return None, [bool(x and x[-1]["radical_beaten"]) for x in rb]
        beaten = [bool(x[-1]["radical_beaten"]) for x in rb]
        return not any(beaten), beaten
    """10 outside-the-box proposals per search (uniform draws with an inverted assumption) must fail to beat the leader."""
    out = []
    for r in res:
        rng = np.random.default_rng(seed + r["group"])
        sd = bench.seed_sd(r["group"]); ctx = bench.context("dev", r["group"])
        lead_bits = max(x["total_bits"] for x in r["log"])
        beaten = False
        for _ in range(n):
            P = optimize.propose("radical", rng, r["best_P"], [], None)
            o = bench.score(optimize.make_fn(P), group=r["group"], resid=False, seed_sd=sd, fn_hash=str(P), ctx=ctx)
            beaten |= o["total_bits"] > lead_bits
        out.append(beaten)
    return not any(out), out


def reveal(res):
    """Show each search the leader's program once; record whether the leader's params beat that search's own best on its group."""
    N = bench.n_evals()
    nets = [final_net(r, N)[0] for r in res]
    lead = int(np.argmax(nets)); out = {}
    for r in res:
        if r["group"] == lead: continue
        o = bench.score(fn_of(res[lead]), group=r["group"], resid=False, seed_sd=bench.seed_sd(r["group"]),
                        fn_hash="reveal" + str(r["group"]) + str(lead), ctx=bench.context("dev", r["group"]))
        mine = max(x["total_bits"] for x in r["log"])
        out[f"g{r['group']}"] = dict(own_best_bits=mine, leader_params_bits=o["total_bits"], leader_improves=bool(o["total_bits"] > mine))
    return dict(leader=lead, results=out)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "optuna"
    res = load(mode=mode)
    rev = reveal(res)
    rad, per = radical_round(res)
    cert = certificate(res, reveal=rev, radical=rad)
    json.dump(cert, open(f"results/certificate_{mode}.json", "w"), indent=1, default=str)
    print(json.dumps({k: cert[k] for k in ("verdict", "N", "radical_round_passed")}), "fail_rates", [round(f, 2) for f in cert["stagnation_fail_rates"]])
