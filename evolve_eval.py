"""OpenEvolve evaluator: bench scoring + Optuna tuning + 200-step probe duplicate filter + per-program cache. Group via EVOLVE_GROUP."""
import json, os, pickle, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import bench
from tunable import tune_program

G = int(os.environ.get("EVOLVE_GROUP", 0))
CACHE = f"results/cache/g{G}"
ARCH = f"{CACHE}/archive.pkl"
LOGF = f"results/llm_g{G}.jsonl"

# Optuna tuning config
N_TRIALS = int(os.environ.get("OPTUNA_TRIALS", "15"))
OPTUNA_SAMPLER = os.environ.get("OPTUNA_SAMPLER", "gpsampler")
OPTUNA_SEED = int(os.environ.get("OPTUNA_SEED", "42"))


def probe_hists(ctx, n=200):
    m, _, edges, _, _, _, lo, hi = ctx
    cnt = np.isfinite(m.close).sum(axis=0)
    j = int(np.argmax(cnt))
    f0 = int(np.argmax(np.isfinite(m.close[:, j])))
    ts = np.linspace(max(lo, f0 + 130), hi - 1, n).astype(int)
    return [bench.History(m.coins[j], m.close[f0:t + 1, j].copy(), m.dv[f0:t + 1, j].copy(), m.btc[f0:t + 1].copy(), m.dates[f0:t + 1].copy(), edges) for t in ts]


def _log(rec):
    with open(LOGF, "a") as f:
        f.write(json.dumps(rec) + "\n")


def _evaluate_with_tuning(program_path, program_source):
    """Evaluate a program with Optuna tuning of TUNABLE parameters."""
    ctx = bench.context("dev", G)

    # Define evaluation function for Optuna
    def eval_fn(source):
        # Write temporary program
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(source)
            temp_path = f.name
        try:
            fn, h = bench.load_fn(temp_path)
            o = bench.score(fn, group=G, resid=False, seed_sd=bench.seed_sd(G), fn_hash=h, ctx=ctx)
            return {"combined_score": o["total_bits"]}
        finally:
            os.unlink(temp_path)

    # Run Optuna tuning
    tuned_source, best_params, best_score = tune_program(
        program_source,
        eval_fn,
        n_trials=N_TRIALS,
        sampler=OPTUNA_SAMPLER,
        seed=OPTUNA_SEED,
    )

    # Evaluate the tuned program
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(tuned_source)
        tuned_path = f.name

    try:
        fn, h = bench.load_fn(tuned_path)
        o = bench.score(fn, group=G, resid=False, seed_sd=bench.seed_sd(G), fn_hash=h, ctx=ctx)
        # Add tuning info to the result
        o["tuned_params"] = best_params
        o["tuned_score"] = best_score
        return o
    finally:
        os.unlink(tuned_path)


def evaluate(program_path):
    os.makedirs(CACHE, exist_ok=True)
    t0 = time.time()

    # Read program source
    with open(program_path) as f:
        program_source = f.read()

    ctx = bench.context("dev", G)

    try:
        # First, run probe on the original program to check for duplicates
        fn, h = bench.load_fn(program_path)
        pr = np.concatenate([bench.clip_probs(fn(x)) for x in probe_hists(ctx)])
    except Exception as e:
        bench.register("fail", G, "dev")
        _log(dict(ts=t0, hash="fail", total_bits=-1e6, error=f"{type(e).__name__}: {e}"[:200], dup=False))
        return {"combined_score": -1e6, "error": str(e)[:200]}

    # Check for duplicates in archive
    arch = pickle.load(open(ARCH, "rb")) if os.path.exists(ARCH) else {}
    for h2, (pr2, met2) in arch.items():
        if h2 == h or np.corrcoef(pr, pr2)[0, 1] > 0.99:
            _log(dict(ts=t0, hash=h, total_bits=met2["total_bits"], dup=True, of=h2, secs=time.time() - t0))
            return dict(met2, duplicate=1.0)

    # Run evaluation with Optuna tuning
    try:
        o = _evaluate_with_tuning(program_path, program_source)
    except Exception as e:
        bench.register("fail", G, "dev")
        _log(dict(ts=t0, hash=h, total_bits=-1e6, error=f"{type(e).__name__}: {e}"[:200], dup=False))
        return {"combined_score": -1e6, "error": str(e)[:200]}

    daily = o.pop("_saved_daily").astype(np.float32)
    np.save(f"{CACHE}/{h}.npy", daily)
    met = {k: v for k, v in o.items() if isinstance(v, (int, float))}
    met["combined_score"] = met["total_bits"]  # penalty depends on N (grows during the run)

    # Store tuning info
    if "tuned_params" in o:
        met["tuned_params"] = o["tuned_params"]
        met["tuned_score"] = o["tuned_score"]

    arch[h] = (pr, met)
    pickle.dump(arch, open(ARCH, "wb"))
    _log(dict(ts=t0, hash=h, total_bits=o["total_bits"], direction_bits=o["direction_bits"], scale_bits=o["scale_bits"], p_skill=o["p_skill"], dup=False, secs=time.time() - t0, tuned_params=o.get("tuned_params", {})))
    return met


if __name__ == "__main__":
    # Test the evaluator
    import sys
    if len(sys.argv) > 1:
        result = evaluate(sys.argv[1])
        print(json.dumps(result, indent=2, default=str))