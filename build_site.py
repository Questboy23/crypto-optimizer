"""Builds site/optimizer/index.html from results/*.json and REPORT.md (no network, no dependencies)."""
import html, json, pathlib
R = pathlib.Path(__file__).parent
j = lambda n: json.load(open(R / "results" / n))
h, cmp_, co, cl, g4 = j("holdout.json"), j("compare.json"), j("certificate_optuna.json"), j("certificate_llm.json"), j("gate4_v4_100trials.json")
rec = j("recal.json")
f = lambda x, d=1: f"{x:,.{d}f}"
rows = "".join(f"<tr><td>{k}</td><td>{f(v['total_bits'])}</td><td>{f(v['direction_bits'])}</td><td>{f(v['scale_bits'])}</td><td>{f(v['resid_bits'])}</td><td>{v['ev_ann']*100:.2f}%</td><td>{v['p_skill']:.3f}</td></tr>" for k, v in h.items())
grp = "".join(f"<tr><td>{r['group']}</td><td>{f(r['optuna_net'],0)}</td><td>{f(r['llm_net'],0)}</td><td>{f(r['optuna_dir'],0)}</td></tr>" for r in cmp_)
pairs = "".join(f"<tr><td>{k}</td><td>{f(v['diff'])}</td><td>{f(v['ci90'][0])} to {f(v['ci90'][1])}</td><td>{v['tost_p']:.2f}</td><td>{'yes' if v['equivalent'] else 'no'}</td></tr>" for k, v in co["pairs"].items())
bars = "".join(f"<div class=bar><span>{k}</span><i style='width:{max(2, min(100, 50 + v['total_bits']/2))}%'></i><b>{f(v['total_bits'])} bits</b></div>" for k, v in h.items())
report = html.escape((R / "REPORT.md").read_text())
page = f"""<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>Crypto Optimizer</title>
<style>:root{{--bg:#fff;--fg:#14161a;--mut:#5b6270;--line:#e3e6eb;--acc:#0b5fff;--card:#f6f7f9}}@media(prefers-color-scheme:dark){{:root{{--bg:#0f1115;--fg:#e8eaee;--mut:#9aa2b1;--line:#262b34;--acc:#6ea0ff;--card:#171a21}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,sans-serif}}main{{max-width:880px;margin:0 auto;padding:24px 16px 64px}}
h1{{font-size:28px;margin:8px 0}}h2{{font-size:20px;margin:36px 0 8px}}p,li{{max-width:70ch}}.mut{{color:var(--mut)}}a{{color:var(--acc)}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:16px 0}}.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}}.card b{{display:block;font-size:24px}}
table{{border-collapse:collapse;width:100%;font-size:14px;display:block;overflow-x:auto}}th,td{{border-bottom:1px solid var(--line);padding:6px 10px;text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}
.bar{{display:flex;align-items:center;gap:10px;margin:6px 0}}.bar span{{width:80px}}.bar i{{display:block;height:14px;background:var(--acc);border-radius:3px}}
pre{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;overflow:auto;font-size:12px;max-height:480px}}details{{margin-top:16px}}</style></head><body><main>
<h1>Crypto forecasting optimizer</h1>
<p class=mut>Scores code that predicts next-day coin returns, in bits, on unseen data. Kraken daily USD data, 30 coins per group, holdout 2025-10-01 to 2026-09-30.</p>
<p><b>Verdict: no tradable edge found.</b> The best program saves 27.9 bits over a Student-t baseline across a year of holdout (95% CI -315 to 326). No trade cleared the 1.7% round-trip Kraken cost, so EV is 0.00%/yr.</p>
<div class=cards><div class=card><b>{f(h['best']['total_bits'])}</b>best program, holdout bits</div><div class=card><b>{f(h['mixture']['total_bits'])}</b>fixed-share mixture, holdout bits</div><div class=card><b>0.00%</b>EV/yr after costs</div><div class=card><b>{co['N']}</b>programs tried (search penalty base)</div></div>
<h2>Holdout results (each evaluated once)</h2><div>{bars}</div>
<table><tr><th>program</th><th>total bits</th><th>direction</th><th>scale</th><th>residual</th><th>EV/yr</th><th>skill p</th></tr>{rows}</table>
<p class=mut>Direction bits are negative for all three: the gain comes from volatility scale, not from predicting direction. The mixture beat the best program by 0.109 bits/day (p = 0.003).</p>
<h2>Optuna search vs LLM loop (development)</h2>
<table><tr><th>group</th><th>Optuna net bits</th><th>LLM net bits</th><th>direction bits</th></tr>{grp}</table>
<p class=mut>Identical columns: the local qwen2.5-coder:3b proposer never beat the Optuna program it started from. Certificates: Optuna {co['verdict']}, LLM {cl['verdict']} (radical round untested for 2 of 4 groups).</p>
<h2>Certificate: do the 4 searches agree?</h2>
<table><tr><th>pair</th><th>diff bits</th><th>90% CI</th><th>TOST p</th><th>equivalent</th></tr>{pairs}</table>
<h2>Does the certificate work? (planted signals)</h2>
<p>On 24 simulated markets with a known optimum, the certificate said done 22 times, each correct (precision 22/22, 95% CI 0.85 to 1.00) and missed 2 (recall 0.92, CI 0.74 to 0.98). The rules were tuned on this same set, and all 24 searches converged, so false-done rate is untested.</p>
<h2>Interval calibration (interval-only)</h2>
<p>Online conformal recalibration moves the 5% tail rates from {rec['before']['low_tail5']*100:.1f}% / {rec['before']['up_tail5']*100:.1f}% to {rec['after']['low_tail5']*100:.1f}% / {rec['after']['up_tail5']*100:.1f}%. It cuts bits (1168 to -3669), so it is used for reported intervals only.</p>
<h2>Checks</h2><p>gate1 planted signal recovered within 20%, gate2 at most 8% of 100 random programs significant (got 2%), gate3 look-ahead raises an error, 16 unit tests pass.</p>
<h2>Code and full report</h2><p><a href="https://github.com/Questboy23/crypto-optimizer">github.com/Questboy23/crypto-optimizer</a> (rebuild with <code>python build_site.py</code>).</p>
<details><summary>Full REPORT.md</summary><pre>{report}</pre></details>
</main></body></html>"""
(R / "site" / "optimizer" / "index.html").write_text(page)
(R / "site" / "optimizer" / "REPORT.md").write_text((R / "REPORT.md").read_text())
print("built", len(page))
