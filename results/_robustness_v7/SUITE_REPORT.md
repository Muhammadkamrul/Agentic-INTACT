# Cross-scenario suite

Held-out seeds: 31001,31002,31003,31004,31005. Paired differences with bootstrap 95% intervals, over seeds both methods completed. Note: the oracle is INTACT-RA's own decision rule reading true current slopes -- a perfect-knowledge reference, not an upper bound.

## Fulfilment by scenario

| method | S16_high_ceiling | S19_persistent_drift |
|---|---|---|
| All-reject | 0.751 (n=5) | 0.760 (n=5) |
| B0 all-admit | 0.608 (n=5) | 0.597 (n=5) |
| B3 | 0.832 (n=5) | 0.452 (n=5) |
| INTACT-RA | 0.739 (n=5) | 0.619 (n=5) |
| INTACT-RA (cell regime) | 0.835 (n=5) | 0.453 (n=5) |
| INTACT-RA (per-tenant sweep) | 0.787 (n=5) | 0.633 (n=5) |
| INTACT-RA-Agentic | 0.862 (n=5) | 0.841 (n=5) |
| Oracle* | 0.860 (n=5) | 0.878 (n=5) |

## Paired comparisons by scenario

| comparison | S16_high_ceiling | S19_persistent_drift |
|---|---|---|
| Agentic − INTACT-RA | +0.122 [+0.100, +0.144] 5/5 — positive (CI above 0) | +0.222 [+0.182, +0.252] 5/5 — positive (CI above 0) |
| Agentic − INTACT-RA (cell) | +0.027 [+0.001, +0.059] 3/5 — positive (CI above 0) | +0.388 [+0.353, +0.422] 5/5 — positive (CI above 0) |
| INTACT-RA (cell) − INTACT-RA | +0.095 [+0.055, +0.123] 5/5 — positive (CI above 0) | -0.166 [-0.194, -0.129] 0/5 — negative (CI below 0) |
| Agentic − B3 | +0.030 [+0.013, +0.050] 5/5 — positive (CI above 0) | +0.389 [+0.355, +0.423] 5/5 — positive (CI above 0) |
| INTACT-RA − B3 | -0.092 [-0.118, -0.063] 0/5 — negative (CI below 0) | +0.167 [+0.130, +0.195] 5/5 — positive (CI above 0) |
| oracle − INTACT-RA | +0.120 [+0.096, +0.143] 5/5 — positive (CI above 0) | +0.259 [+0.241, +0.277] 5/5 — positive (CI above 0) |

## INTACT-RA-Agentic vs the best frozen configuration

The best of INTACT-RA (per-tenant), INTACT-RA (cell regime) and B3 in each scenario, chosen AFTER seeing the results -- a comparison deliberately biased against the agent.

| scenario | best frozen | its IF | Agentic − best frozen | verdict |
|---|---|---|---|---|
| S16_high_ceiling | INTACT-RA (cell regime) | 0.835 | +0.027 [+0.001, +0.059] 3/5 | positive (CI above 0) |
| S19_persistent_drift | INTACT-RA (per-tenant sweep) | 0.633 | +0.208 [+0.180, +0.233] 5/5 | positive (CI above 0) |

## What each scenario shows

- **S16_high_ceiling:** INTACT-RA-Agentic vs INTACT-RA positive (CI above 0); INTACT-RA vs B3 negative (CI below 0).
- **S19_persistent_drift:** INTACT-RA-Agentic vs INTACT-RA positive (CI above 0); INTACT-RA vs B3 positive (CI above 0).

**INTACT-RA-Agentic beats INTACT-RA with a CI above zero in 2 of 2 scenario(s) where both were run.** Any claim of winning 'in every scenario' requires this to read 2 of 2.