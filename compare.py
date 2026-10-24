"""Optuna-only vs LLM loop: net bits, EV after costs, direction bits, certificate verdict; paired differences with CI and p."""
import json, numpy as np
import bench, certificate, optimize
import engine.stats as S


def main():
    ro, rl = certificate.load(mode="optuna"), certificate.load(mode="llm")
    N = bench.n_evals()
    rows, lines = [], []
    for g in range(4):
        sd = bench.seed_sd(g); ctx = bench.context("dev", g)
        sc = {}
        for name, r in (("optuna", ro[g]), ("llm", rl[g])):
            sc[name] = bench.score(certificate.fn_of(r), group=g, resid=False, seed_sd=sd, register_eval=False, fn_hash=name, ctx=ctx)
        pen = sd * np.sqrt(2 * np.log(max(N, 2)))
        d_bits = S.paired_diff(sc["llm"]["_saved_daily"], sc["optuna"]["_saved_daily"])
        d_dir = S.paired_diff(sc["llm"]["_dir_daily"], sc["optuna"]["_dir_daily"])
        d_ev = S.paired_diff(sc["llm"]["_ret"], sc["optuna"]["_ret"], B=1000)
        rows.append(dict(group=g, optuna_net=sc["optuna"]["total_bits"] - pen, llm_net=sc["llm"]["total_bits"] - pen,
                         optuna_dir=sc["optuna"]["direction_bits"], llm_dir=sc["llm"]["direction_bits"],
                         optuna_ev=sc["optuna"]["ev_ann"], llm_ev=sc["llm"]["ev_ann"],
                         bits_per_day_diff=(d_bits.point, d_bits.lo, d_bits.hi, d_bits.p), dir_per_day_diff=(d_dir.point, d_dir.lo, d_dir.hi, d_dir.p),
                         ev_diff=(d_ev.point, d_ev.lo, d_ev.hi, d_ev.p),
                         optuna_bits_ci=sc["optuna"]["total_bits_ci"], llm_bits_ci=sc["llm"]["total_bits_ci"]))
    # pooled over groups: concatenate paired daily differences (groups are disjoint coin sets)
    json.dump(rows, open("results/compare.json", "w"), indent=1, default=float)
    cert = {m: json.load(open(f"results/certificate_{m}.json")) for m in ("optuna", "llm")}
    md = ["## Optuna-only vs LLM loop (development period, per coin group; N=%d tries so far)" % N, "",
          "| group | Optuna net bits | LLM net bits | LLM minus Optuna, bits/day (95% CI, p) | direction bits O / L | EV%/yr O / L (diff CI, p) |", "|---|---|---|---|---|---|"]
    for r in rows:
        b, e = r["bits_per_day_diff"], r["ev_diff"]
        md.append(f"| {r['group']} | {r['optuna_net']:.0f} | {r['llm_net']:.0f} | {b[0]:.3f} ({b[1]:.3f} to {b[2]:.3f}, p={b[3]:.3g}) | {r['optuna_dir']:.0f} / {r['llm_dir']:.0f} | "
                  f"{r['optuna_ev']*100:.2f} / {r['llm_ev']*100:.2f} ({e[1]*100:.2f} to {e[2]*100:.2f}, p={e[3]:.3g}) |")
    md += ["", f"Certificate verdict: Optuna **{cert['optuna']['verdict']}**, LLM **{cert['llm']['verdict']}**."]
    meta = {g: json.load(open(f"results/llm_meta_g{g}.json")) for g in range(4)}
    md += ["", "## Proposal style by stage (LLM loop; successes/proposals summed over 4 groups)"]
    for b, bn in enumerate(optimize.BUCKETS):
        cells = []
        for s in meta[0]["ts"][str(b)]:
            a = sum(m["ts"][str(b)][s]["a"] - 1 for m in meta.values()); n = sum(m["ts"][str(b)][s]["n"] for m in meta.values())
            cells.append(f"{s} {int(a)}/{int(n)}")
        md.append(f"- stage {bn}: " + ", ".join(cells))
    open("results/compare.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
