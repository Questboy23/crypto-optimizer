# REPORT

Holdout: 2025-10-01 to 2026-09-30; group 3 of 4 (30 coins); fee 0.80% taker + 0.05% slippage per side. Every program evaluated once.

## Headline per program (holdout)
- **best**: EV 0.00%/yr (95% CI 0.00 to 0.00), skill p = 0.0099 after 1432 tries, direction bits -78.9, worst-5%-day loss 0.00%, P(loss over 1 yr) 0% (total bits 27.9, 95% CI -315.3 to 325.6; scale bits 106.8; resid bits 132.3; trades on 0.00% of predictions)
- **seed**: EV 0.00%/yr (95% CI 0.00 to 0.00), skill p = 0.0099 after 1432 tries, direction bits -52.7, worst-5%-day loss 0.00%, P(loss over 1 yr) 0% (total bits 26.5, 95% CI -286.1 to 321.4; scale bits 79.2; resid bits 108.3; trades on 0.00% of predictions)
- **mixture**: EV 0.00%/yr (95% CI 0.00 to 0.00), skill p = 0.0099 after 1432 tries, direction bits -62.9, worst-5%-day loss 0.00%, P(loss over 1 yr) 0% (total bits 67.8, 95% CI -283.9 to 382.5; scale bits 130.7; resid bits 178.5; trades on 0.00% of predictions)

Best minus seed, bits per day (paired block bootstrap): 0.004 (95% CI -0.330 to 0.313), one-sided p = 0.498.
Mixture minus best, bits per day (one of 2 holdout comparisons, so Bonferroni threshold is p < 0.025; it still passes): 0.109 (95% CI 0.029 to 0.187), p = 0.003.

Best program source: optuna (development bits Optuna 1174.2, LLM 1174.2).

## Certificate (development, 4 searches, Optuna)
Verdict: **data_limited**; N=1342; delta=21.5 bits.
Stagnation fail rates (last 30 proposals): [0.97, 0.97, 1.0, 1.0]; rule-of-three upper bound on improvement rate 0.100.
Radical round passed: True.
Pairwise TOST:
- 0-1: diff 124.3 bits, 90% CI -179.6 to 426.0, TOST p = 0.726, equivalent: False
- 0-2: diff -393.4 bits, 90% CI -1311.6 to 273.1, TOST p = 0.762, equivalent: False
- 0-3: diff -399.4 bits, 90% CI -937.1 to 141.0, TOST p = 0.866, equivalent: False
- 1-2: diff -517.7 bits, 90% CI -931.3 to -141.2, TOST p = 0.983, equivalent: False
- 1-3: diff -523.7 bits, 90% CI -931.2 to -160.3, TOST p = 0.988, equivalent: False
- 2-3: diff -6.0 bits, 90% CI -679.1 to 756.2, TOST p = 0.505, equivalent: False

Reveal: {"leader": 3, "results": {"g0": {"own_best_bits": 774.8446655273438, "leader_params_bits": 711.0294189453125, "leader_improves": false}, "g1": {"own_best_bits": 650.5584716796875, "leader_params_bits": 589.75927734375, "leader_improves": false}, "g2": {"own_best_bits": 1168.2244873046875, "leader_params_bits": 1146.0076904296875, "leader_improves": false}}}

## Strategy table by stage (successes, trials)
- g0 stage <50%: tpe=2/20, tweak=0/0, combine=0/0, radical=0/0
- g0 stage 50-90%: tpe=0/20, tweak=0/20, combine=2/20, radical=0/0
- g0 stage >90%: tpe=0/0, tweak=0/0, combine=6/60, radical=0/20
- g1 stage <50%: tpe=0/20, tweak=0/0, combine=0/0, radical=0/0
- g1 stage 50-90%: tpe=0/0, tweak=0/0, combine=0/0, radical=0/0
- g1 stage >90%: tpe=0/0, tweak=3/100, combine=0/20, radical=0/20
- g2 stage <50%: tpe=0/0, tweak=0/0, combine=2/20, radical=0/0
- g2 stage 50-90%: tpe=0/0, tweak=1/20, combine=0/20, radical=0/0
- g2 stage >90%: tpe=0/0, tweak=0/0, combine=6/80, radical=0/20
- g3 stage <50%: tpe=0/0, tweak=0/20, combine=0/0, radical=0/0
- g3 stage 50-90%: tpe=0/0, tweak=0/20, combine=0/0, radical=0/20
- g3 stage >90%: tpe=3/40, tweak=0/0, combine=6/60, radical=0/0

## Optuna-only vs LLM loop (development period, per coin group; N=1425 tries so far)

| group | Optuna net bits | LLM net bits | LLM minus Optuna, bits/day (identical because the LLM never found a better program than its Optuna seed) (95% CI, p) | direction bits O / L | EV%/yr O / L (diff CI, p) |
|---|---|---|---|---|---|
| 0 | -241 | -241 | 0.000 (0.000 to 0.000, p=1) | -261 / -261 | 0.00 / 0.00 (0.00 to 0.00, p=1) |
| 1 | -662 | -662 | 0.000 (0.000 to 0.000, p=1) | -285 / -285 | 0.00 / 0.00 (0.00 to 0.00, p=1) |
| 2 | 28 | 28 | 0.000 (0.000 to 0.000, p=1) | -148 / -148 | 0.00 / 0.00 (0.00 to 0.00, p=1) |
| 3 | 211 | 211 | 0.000 (0.000 to 0.000, p=1) | -280 / -280 | 0.00 / 0.00 (0.00 to 0.00, p=1) |

Certificate verdict: Optuna **data_limited**, LLM **data_limited**.

## Proposal style by stage (LLM loop; successes/proposals summed over 4 groups)
- stage <50%: rewrite 0/0, tweak 2/10, combine 0/0, outside 2/14
- stage 50-90%: rewrite 0/6, tweak 0/0, combine 0/0, outside 0/0
- stage >90%: rewrite 0/0, tweak 0/0, combine 0/6, outside 0/24

LLM certificate: **data_limited** (N=1425, delta=21.1, radical round passed: None, stagnation fail rates [0.92, 0.92, 0.94, 0.92]).

## Recalibration (interval-only)
Used only to report prediction-interval coverage, never in bits scoring. Group 2 best Optuna program, central 90% interval and 5% tails, development period:
```
{
 "before": {
  "inside90": 0.9283959513904811,
  "low_tail5": 0.03691571947983884,
  "up_tail5": 0.03468832912967998,
  "total_bits": 1168.2242431640625
 },
 "after": {
  "inside90": 0.8887287497133872,
  "low_tail5": 0.05784663762324347,
  "up_tail5": 0.05342461266336926,
  "total_bits": -3669.203369140625
 }
}
```
Before: inside-90 0.928, tails 3.7% / 3.5% (too wide). After: inside-90 0.889, tails 5.8% / 5.3% (near nominal 5%). Applied to the forecast itself it costs bits (1168 -> -3669), so it is not used for scoring.

## Gate results
```
uv run pytest -q tests > logs/pytest.log 2>&1; uv run python check.py
{"gate1": "pass", "gate2": "pass", "gate3": "pass", "tests": "pass"}

```

## Certificate grading on planted signals (gate 4, 24 instances, 4 searches each, 100 trials)
```
{
 "n": 24,
 "verdicts": {
  "done": 22,
  "not_done": 0,
  "data_limited": 2
 },
 "precision": 1.0,
 "precision_ci": [
  0.8513451253913791,
  1.0
 ],
 "n_done": 22,
 "recall": 0.9166666666666666,
 "recall_ci": [
  0.741511978067893,
  0.9768411847029567
 ],
 "n_converged": 24,
 "instances": [
  {
   "mu": 0.0,
   "verdict": "done",
   "all_near": true,
   "frac_near": 1.0
  },
  {
   "mu": 0.008037375350951179,
   "verdict": "done",
   "all_near": true,
   "frac_near": 1.0
  },
  {
   "mu": 0.01157063889887753,
   "verdict": "done",
   "all_near": true,
   "frac_near": 1.0
  },
  {
   "mu": 0.0,
   "verdict": "done",
   "all_near": true,
   "frac_near": 1.0
  },
  {
   "mu": 0.008037375350951179,
   "verdict": "done",
   "all_near": true,
   "frac_near": 1.0
  },
  {
   "mu": 0.01157063889887753,
   "verdict": "done",
   "all_near": true,
   "frac_near": 1.0
  },
  {
   "mu": 0.0,
   "
```
Precision 22/22 (Wilson 95% CI 0.85 to 1.00); recall 22/24 (0.74 to 0.98). All 24 instances converged by the 10% criterion, so a false-done rate is untested. Certificate rules (stagnation counts only gains above 1%, 100 trials) were adjusted across four grading runs on this same set (distinct-data 0 done, same-data 40 trials 0 done), so these numbers are tuned on the grading data.

## Cost
LLM tokens used at runtime: about 197,000 (62 logged proposals of the final qwen2.5-coder:3b run on local Ollama, cost $0; earlier 7b and diff-mode attempts not counted). Optuna path: 0. Claude Code tokens are in ccusage.
GPU hours: 0 attached (CPU-only Kaggle probes logged 0.162h in gpu_usage.csv).
