import json, sys, numpy as np
import bench, optimize, certificate
from engine.recal import Recalibrated, coverage
g = int(sys.argv[1]); res = json.load(open(f"results/search_g{g}.json"))
ctx = bench.context("dev", g); m, _, edges, means, _, _, lo, hi = ctx
fn = optimize.make_fn(res["best_P"])
out = {}
for name, f in (("before", fn), ("after", Recalibrated(fn))):
    r = bench.run(f, m, edges, means, lo, hi)
    bits = (bench._bits(r["base"], r["O"], r["elig"]) - bench._bits(r["P"], r["O"], r["elig"])).sum()
    out[name] = dict(coverage(r["P"], r["O"], r["elig"]), total_bits=float(bits))
json.dump(out, open("results/recal.json", "w"), indent=1); print(json.dumps(out))
