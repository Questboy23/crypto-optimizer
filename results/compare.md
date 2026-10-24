## Optuna-only vs LLM loop (development period, per coin group; N=1425 tries so far)

| group | Optuna net bits | LLM net bits | LLM minus Optuna, bits/day (95% CI, p) | direction bits O / L | EV%/yr O / L (diff CI, p) |
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
