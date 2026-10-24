"""OpenEvolve loop on one coin group with Ollama (qwen2.5-coder:3b), batches of N iterations chosen by Thompson sampling over proposal styles."""
import glob, json, os, sys, time, re
import numpy as np
os.environ.setdefault("OPENAI_API_KEY", "ollama")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE = """You improve a Python function predict(history) that forecasts the next-day log return distribution of one crypto coin.
It returns a numpy array of 21 probabilities (bins; history.edges holds the 20 ascending bin edges). Score = bits saved versus a Student-t baseline, summed over many days and coins.
history.close, history.volume (USD), history.btc_close, history.dates are numpy arrays up to today only. history.t is today's index. Never use data after today.
Use only numpy and scipy. Keep the function fast (under 1 ms per call). Smaller, simpler code wins ties."""
STYLES = {
    "rewrite": ("Rewrite the structure of predict substantially (new volatility estimator, new shape, new features) while keeping the output format.", 0.7),
    "tweak": ("Make one small change: adjust a parameter, a window length, or one coefficient of the current program. Return the complete program.", 0.6),
    "combine": ("Combine the best ideas from the top programs and inspirations shown into one program.", 0.7),
    "outside": ("Think outside the box. Borrow a mechanism from an unrelated field (physics, ecology, epidemiology, auctions, information theory) or invert one assumption of the current best program, then implement it as a predict function.", 1.0),
}
BUCKETS = ["<50%", "50-90%", ">90%"]
MODEL = os.environ.get("LLM_MODEL", "qwen2.5-coder:3b")
OLLAMA_API_BASE = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434/v1")


def template(P):
    return f'''import numpy as np
from scipy.stats import t as tdist
from tunable import TUNABLE

# EVOLVE-BLOCK-START
PARAMS = {{
    "win": TUNABLE("win", 10, 200, {P.get("win", 60)}, param_type="int"),
    "mom": TUNABLE("mom", 5, 100, {P.get("mom", 20)}, param_type="int"),
    "df": TUNABLE("df", 2.1, 30.0, {P.get("df", 4.0)}),
    "tilt": TUNABLE("tilt", -1.0, 1.0, {P.get("tilt", 0.05)}),
}}


def predict(history):
    c = history.close
    r = np.diff(np.log(c[-(PARAMS["win"] + 1):]))
    sd = r.std() if len(r) > 5 else 0.05
    m = np.diff(np.log(c[-(PARAMS["mom"] + 1):])).mean()
    df = PARAMS["df"]
    loc = PARAMS["tilt"] * np.tanh(m / max(sd, 1e-6)) * sd
    scale = max(sd, 1e-4) * np.sqrt((df - 2) / df)
    cdf = tdist.cdf((history.edges - loc) / scale, df)
    return np.diff(np.r_[0.0, cdf, 1.0])
# EVOLVE-BLOCK-END
'''


def diagnostics(path, g):
    """Where the incumbent loses bits: by volatility regime, weekday, worst coins (one run of the program)."""
    import bench
    ctx = bench.context("dev", g); m, _, edges, means, _, _, lo, hi = ctx
    fn, _ = bench.load_fn(path)
    r = bench.run(fn, m, edges, means, lo, hi)
    saved = bench._bits(r["base"], r["O"], r["elig"]) - bench._bits(r["P"], r["O"], r["elig"])
    el, steps = r["elig"], r["steps"]
    import pandas as pd
    vol = pd.DataFrame(m.R).rolling(30, min_periods=10).std().to_numpy()[steps]
    q = np.nanquantile(vol[el], [1 / 3, 2 / 3])
    lines = []
    for nm, mk in (("low-vol", vol <= q[0]), ("mid-vol", (vol > q[0]) & (vol <= q[1])), ("high-vol", vol > q[1])):
        mk = mk & el; lines.append(f"{nm}: {saved[mk].sum():.0f} bits over {mk.sum()} predictions ({saved[mk].mean():.4f}/pred)")
    wd = (m.dates[steps + 1].astype("datetime64[D]").astype(int) + 3) % 7
    lines.append("weekday bits/pred Mon..Sun: " + " ".join(f"{saved[(wd == d)[:, None] & el].mean():.3f}" for d in range(7)))
    cs = np.where(el, saved, 0).sum(axis=0)
    worst = np.argsort(cs)[:3]
    lines.append("worst coins: " + ", ".join(f"{m.coins[j]} {cs[j]:.0f}" for j in worst))
    return "Error profile of the current best program (bits saved versus baseline; negative is a loss):\n" + "\n".join(lines)


def read_log(g):
    f = f"results/llm_g{g}.jsonl"
    return [json.loads(x) for x in open(f)] if os.path.exists(f) else []


def run_group(g, n_batches, batch, seed_params, tag="llm"):
    from openevolve.api import run_evolution
    from openevolve.config import Config, LLMModelConfig
    os.environ["EVOLVE_GROUP"] = str(g)
    rng = np.random.default_rng(900 + g)
    out = f"results/oe_g{g}"
    os.makedirs(out, exist_ok=True)
    seed_path = f"{out}/seed.py"
    open(seed_path, "w").write(template(seed_params))
    ts = {b: {s: dict(a=1.0, b=1.0, gain=1.0, cost=1.0, n=0) for s in STYLES} for b in range(3)}
    best_path, ckpt, batches = seed_path, None, []
    start_bi = 0
    if os.environ.get("RESUME") and os.path.exists(f"results/{tag}_meta_g{g}.json"):   # continue after a hung batch
        meta = json.load(open(f"results/{tag}_meta_g{g}.json"))
        ts, batches, best_path = meta["ts"], meta["batches"], meta["best_path"]
        ts = {int(k): v for k, v in ts.items()}
        cks = sorted(glob.glob(f"{out}/checkpoints/checkpoint_*"), key=lambda p: int(p.rsplit("_", 1)[1]))
        ckpt, start_bi = (cks[-1] if cks else None), len(batches)
    for bi in range(start_bi, n_batches + 1):
        radical = bi == n_batches          # final radical round: 10 outside-the-box proposals
        nb_iter = int(os.environ.get("RADICAL_N", 10)) if radical else batch
        log = read_log(g)
        flags = [] if not log else None
        # stage = fraction of the last `batch` full proposals that failed to improve the running best
        run_max, imp = -1e18, []
        for rec in log:
            imp.append(rec["total_bits"] > run_max + 1e-9 and not rec.get("dup")); run_max = max(run_max, rec["total_bits"] if not rec.get("error") else run_max)
        fails = 1 - float(np.mean(imp[-batch:])) if len(imp) > 1 else 0.0
        bucket = 0 if fails < 0.5 else (1 if fails <= 0.9 else 2)
        draws = {s: rng.beta(ts[bucket][s]["a"], ts[bucket][s]["b"]) * ts[bucket][s]["gain"] / ts[bucket][s]["cost"] for s in STYLES}
        style = "outside" if radical else max(draws, key=draws.get)
        instr, temp = STYLES[style]
        diag = diagnostics(best_path, g)
        cfg = Config()
        cfg.max_iterations = nb_iter
        cfg.checkpoint_interval = nb_iter
        cfg.diff_based_evolution = False   # Full rewrites parse reliably with free models
        cfg.llm.api_base = OLLAMA_API_BASE
        cfg.llm.models = [LLMModelConfig(name=MODEL, api_base=OLLAMA_API_BASE, api_key="ollama", weight=1.0, random_seed=int(rng.integers(1, 10**6)))]
        cfg.llm.temperature, cfg.llm.max_tokens, cfg.llm.timeout = temp, 1200, int(os.environ.get("LLM_TIMEOUT", 600))
        cfg.llm.retries = 1
        cfg.prompt.system_message = f"{BASE}\n\nTHIS BATCH: {instr}\n\n{diag}"
        cfg.database.num_islands, cfg.database.population_size = 2, 60
        cfg.evaluator.cascade_evaluation, cfg.evaluator.timeout, cfg.evaluator.parallel_evaluations = False, 400, 1
        n0, t0 = len(log), time.time()
        res = run_evolution(seed_path, "evolve_eval.py", config=cfg, iterations=nb_iter, output_dir=out, cleanup=False, checkpoint_path=ckpt)
        if res.best_code:
            best_path = f"{out}/best_so_far.py"; open(best_path, "w").write(res.best_code)
        cks = sorted(glob.glob(f"{out}/checkpoints/checkpoint_*"), key=lambda p: int(p.rsplit("_", 1)[1]))
        ckpt = cks[-1] if cks else None
        new = read_log(g)[n0:]
        run_max = max([r["total_bits"] for r in log if not r.get("error")] or [-1e18])
        st, tot = ts[bucket][style], 0
        lead0 = run_max
        for rec in new:
            ok = (not rec.get("error")) and (not rec.get("dup")) and rec["total_bits"] > run_max + 1e-9
            gain = max(rec["total_bits"] - run_max, 0.0) if ok and run_max > -1e17 else 0.0
            if ok: run_max = rec["total_bits"]
            st["a" if ok else "b"] += 1; st["gain"] += gain; tot += ok
        st["cost"] += (time.time() - t0) / 100.0; st["n"] += len(new)
        radical_beaten = bool(radical and run_max > lead0 + 1e-9)
        batches.append(dict(radical=radical, radical_beaten=radical_beaten, batch=bi, bucket=BUCKETS[bucket], style=style, proposals=len(new), improved=tot, secs=time.time() - t0, best=run_max))
        json.dump(dict(group=g, batches=batches, ts=ts, best_path=best_path), open(f"results/{tag}_meta_g{g}.json", "w"), indent=1)
        print(json.dumps(batches[-1]), flush=True)


if __name__ == "__main__":
    g, nb, b = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    sp = json.loads(open(f"results/search_g{g}.json").read())["best_P"] if os.path.exists(f"results/search_g{g}.json") else {"win": 60, "tilt": 0.05, "mom": 20, "df": 4.0}
    run_group(g, nb, b, sp)
