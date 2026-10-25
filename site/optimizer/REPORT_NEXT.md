# REPORT_NEXT: prompt-style selection ablation

Metric: proposals until best-so-far reaches 95% of the known optimum (never reached counts as 40).

A1 fixed style: `structural_rewrite`

## Arms

| arm | runs | mean proposals to 95% | reached 95% |
|---|---|---|---|
| A1 | 36 | 6.5 | 89% |
| A2 | 36 | 7.3 | 89% |
| B | 36 | 9.2 | 86% |
| C | 36 | 8.3 | 89% |

## Paired speedups (reference / challenger, mean proposals)

| comparison | pairs | speedup | 95% CI | p (one-sided) | p (BH) |
|---|---|---|---|---|---|
| B vs A1 | 36 | 0.71x | 0.44 to 0.90 | 0.974 | 0.974 |
| B vs A2 | 36 | 0.79x | 0.53 to 0.99 | 0.936 | 0.974 |
| C vs B | 36 | 1.10x | 0.86 to 1.47 | 0.211 | 0.634 |

## Per problem (mean proposals to 95%)

| problem | A1 | A2 | B | C |
|---|---|---|---|---|
| bbob_sphere | 1.5 | 1.2 | 2.0 | 1.2 |
| bbob_ellipsoidal | 2.0 | 3.0 | 7.2 | 2.8 |
| bbob_rastrigin | 2.0 | 2.5 | 4.0 | 8.2 |
| bbob_rosenbrock | 2.8 | 4.2 | 14.8 | 4.2 |
| tsplib_burma14 | 1.0 | 1.8 | 1.5 | 1.8 |
| tsplib_ulysses16 | 1.8 | 1.5 | 2.2 | 3.5 |
| tsplib_ulysses22 | 1.2 | 3.5 | 4.5 | 8.5 |
| tsplib_att48 | 6.5 | 7.8 | 6.5 | 4.8 |
| synthetic_market | 40.0 | 40.0 | 40.0 | 40.0 |

## Controls (same compare_arms code)

- Positive control (true 2x speedup): speedup 2.03x, p=0.0005, detected.
- Negative control (no difference, 20 draws): 20/20 quiet, ok.

Proposals by provider: {'groq': 81, 'nim': 335, 'groq-qwen': 74, 'kilo': 425, 'none': 213}; total tokens 3701449.
