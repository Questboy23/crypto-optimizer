"""Evaluates the frozen best program, the 4-program fixed-share mixture and the seed ONCE each on the holdout; writes REPORT.md."""
import json, os, sys, numpy as np
import bench, optimize, certificate
import engine.stats as S

LOCK = "data/holdout_done.json"


def line(name, o, N):
    ev = o["ev_ann"]; lo, hi = o["ev_ci"]
    return (f"- **{name}**: EV {ev*100:.2f}%/yr (95% CI {lo*100:.2f} to {hi*100:.2f}), skill p = {o['p_skill']:.3g} after {N} tries, "
            f"direction bits {o['direction_bits']:.1f}, worst-5%-day loss {o['cvar95']*100:.2f}%, P(loss over 1 yr) {o['p_loss_1y']*100:.0f}% "
            f"(total bits {o['total_bits']:.1f}, 95% CI {o['total_bits_ci'][0]:.1f} to {o['total_bits_ci'][1]:.1f}; scale bits {o['scale_bits']:.1f}; resid bits {o.get('resid_bits', float('nan')):.1f}; trades on {o['trade_frac']*100:.2f}% of predictions)")


def main():
    if os.path.exists(LOCK):
        sys.exit("holdout already evaluated; refusing to look again")
    res = certificate.load()
    cert = json.load(open("results/certificate_optuna.json"))
    certl = json.load(open("results/certificate_llm.json"))
    resl = certificate.load(mode="llm")
    lead = cert["reveal"]["leader"]
    g = res[lead]["group"]
    # best = whichever of the Optuna / LLM programs has more development bits on the leader group (the holdout is never used to choose)
    opt_bits = max(x["total_bits"] for x in res[lead]["log"]); llm_bits = max(x["total_bits"] for x in resl[lead]["log"])
    best_src = "llm" if llm_bits > opt_bits + 1e-9 else "optuna"
    best_fn = certificate.fn_of(resl[lead]) if best_src == "llm" else optimize.make_fn(res[lead]["best_P"])
    fns = {"best": best_fn, "seed": optimize.make_fn({"win": 60, "tilt": 0.05, "mom": 20, "df": 4.0})}
    fns["mixture"] = S.FixedShareMixture([optimize.make_fn(r["best_P"]) for r in res], alpha=0.01)
    sd = bench.seed_sd(g)
    outs = {}
    for name, fn in fns.items():
        outs[name] = bench.score(fn, split="holdout", group=g, resid=True, seed_sd=sd, fn_hash="holdout-" + name)
    N = bench.n_evals()
    json.dump({k: {a: b for a, b in v.items() if a != "_saved_daily"} for k, v in outs.items()}, open("results/holdout.json", "w"), default=str, indent=1)
    open(LOCK, "w").write(json.dumps({"done": True, "group": g}))
    gpu = open("gpu_usage.csv").read().strip().splitlines()[1:]
    gpu_h = sum(float(x.split(",")[3]) for x in gpu)
    st = json.load(open("strategies.json"))
    md = ["# REPORT", "", f"Holdout: {data_range()}; group {g} of 4 (30 coins); fee 0.80% taker + 0.05% slippage per side. Every program evaluated once.", "",
          "## Headline per program (holdout)"] + [line(k, o, N) for k, o in outs.items()]
    best, seed = outs["best"], outs["seed"]
    d = S.paired_diff(best["_saved_daily"], seed["_saved_daily"])
    mix = outs["mixture"]
    dm = S.paired_diff(mix["_saved_daily"], best["_saved_daily"])
    md += ["", f"Best minus seed, bits per day (paired block bootstrap): {d.point:.3f} (95% CI {d.lo:.3f} to {d.hi:.3f}), one-sided p = {d.p:.3g}.",
           f"Mixture minus best, bits per day: {dm.point:.3f} (95% CI {dm.lo:.3f} to {dm.hi:.3f}), p = {dm.p:.3g}.",
           "", f"Best program source: {best_src} (development bits Optuna {opt_bits:.1f}, LLM {llm_bits:.1f}).", "", "## Certificate (development, 4 searches, Optuna)", f"Verdict: **{cert['verdict']}**; N={cert['N']}; delta={cert['delta']:.1f} bits.",
           f"Stagnation fail rates (last 30 proposals): {[round(x, 2) for x in cert['stagnation_fail_rates']]}; rule-of-three upper bound on improvement rate {cert['rule_of_three_upper_bound']:.3f}.",
           f"Radical round passed: {cert['radical_round_passed']}.", "Pairwise TOST:"] + \
          [f"- {k}: diff {v['diff']:.1f} bits, 90% CI {v['ci90'][0]:.1f} to {v['ci90'][1]:.1f}, TOST p = {v['tost_p']:.3g}, equivalent: {v['equivalent']}" for k, v in cert["pairs"].items()] + \
          ["", f"Reveal: {json.dumps(cert['reveal'])}", "", "## Strategy table by stage (successes, trials)"]
    for gk, tab in st.items():
        for b, row in tab.items():
            md.append(f"- {gk} stage {optimize.BUCKETS[int(b)]}: " + ", ".join(f"{s}={int(v['a']-1)}/{v['n']}" for s, v in row.items()))
    md += ["", open("results/compare.md").read(), f"LLM certificate: **{certl['verdict']}** (N={certl['N']}, delta={certl['delta']:.1f}, radical round passed: {certl['radical_round_passed']}, stagnation fail rates {[round(x, 2) for x in certl['stagnation_fail_rates']]}).",
          "", "## Recalibration (interval-only)", "Used only to report prediction-interval coverage, never in bits scoring. Group 2 best Optuna program, central 90% interval and 5% tails, development period:",
          "```", open("results/recal.json").read(), "```", "Before: inside-90 0.928, tails 3.7% / 3.5% (too wide). After: inside-90 0.889, tails 5.8% / 5.3% (near nominal 5%). Applied to the forecast itself it costs bits (1168 -> -3669), so it is not used for scoring.",
          "", "## Gate results", "```", open("results/gates.txt").read() if os.path.exists("results/gates.txt") else "see check.py output", "```",
          "", "## Certificate grading on planted signals (gate 4, 24 instances, 4 searches each, 100 trials)", "```", open("results/gate4_v4_100trials.json").read()[:900], "```",
          "Precision 22/22 (Wilson 95% CI 0.85 to 1.00); recall 22/24 (0.74 to 0.98). All 24 instances converged by the 10% criterion, so a false-done rate is untested. Certificate rules (stagnation counts only gains above 1%, 100 trials) were adjusted across four grading runs on this same set (distinct-data 0 done, same-data 40 trials 0 done), so these numbers are tuned on the grading data.",
          "", "## Cost", "LLM tokens used at runtime: about 197,000 (62 logged proposals of the final qwen2.5-coder:3b run on local Ollama, cost $0; earlier 7b and diff-mode attempts not counted). Optuna path: 0. Claude Code tokens are in ccusage.", f"GPU hours: 0 attached (CPU-only Kaggle probes logged {gpu_h:.3f}h in gpu_usage.csv)."]
    open("REPORT.md", "w").write("\n".join(md) + "\n")
    print("wrote REPORT.md")


def data_range():
    import data
    _, h = data.split_dates()
    return f"{str(h)[:10]} to {str(data.end_date())[:10]}"


if __name__ == "__main__":
    main()
