# Cross-scenario suite

Held-out seeds: 31001,31002,31003,31004,31005. Paired differences with bootstrap 95% intervals, over seeds both methods completed. Note: the oracle is INTACT-RA's own decision rule reading true current slopes -- a perfect-knowledge reference, not an upper bound.

## Fulfilment by scenario

| method | S16_high_ceiling | S19_persistent_drift |
|---|---|---|
| All-reject | 0.751 (n=5) | 0.788 (n=5) |
| B0 all-admit | 0.608 (n=5) | 0.622 (n=5) |
| B3 | 0.832 (n=5) | 0.440 (n=5) |
| INTACT-RA | 0.739 (n=5) | 0.621 (n=5) |
| INTACT-RA (cell regime) | 0.835 (n=5) | 0.441 (n=5) |
| INTACT-RA (per-tenant sweep) | 0.787 (n=5) | 0.623 (n=5) |
| INTACT-RA-Agentic | 0.862 (n=5) | 0.849 (n=5) |
| Oracle* | 0.860 (n=5) | 0.860 (n=5) |

## Paired comparisons by scenario

| comparison | S16_high_ceiling | S19_persistent_drift |
|---|---|---|
| Agentic − INTACT-RA | +0.122 [+0.100, +0.144] 5/5 — positive (CI above 0) | +0.229 [+0.191, +0.266] 5/5 — positive (CI above 0) |
| Agentic − INTACT-RA (cell) | +0.027 [+0.001, +0.059] 3/5 — positive (CI above 0) | +0.408 [+0.383, +0.430] 5/5 — positive (CI above 0) |
| INTACT-RA (cell) − INTACT-RA | +0.095 [+0.055, +0.123] 5/5 — positive (CI above 0) | -0.179 [-0.195, -0.159] 0/5 — negative (CI below 0) |
| Agentic − B3 | +0.030 [+0.013, +0.050] 5/5 — positive (CI above 0) | +0.409 [+0.384, +0.431] 5/5 — positive (CI above 0) |
| INTACT-RA − B3 | -0.092 [-0.118, -0.063] 0/5 — negative (CI below 0) | +0.180 [+0.162, +0.196] 5/5 — positive (CI above 0) |
| oracle − INTACT-RA | +0.120 [+0.096, +0.143] 5/5 — positive (CI above 0) | +0.239 [+0.208, +0.275] 5/5 — positive (CI above 0) |

## INTACT-RA-Agentic vs the best frozen configuration

The best of INTACT-RA (per-tenant), INTACT-RA (cell regime) and B3 in each scenario, chosen AFTER seeing the results -- a comparison deliberately biased against the agent.

| scenario | best frozen | its IF | Agentic − best frozen | verdict |
|---|---|---|---|---|
| S16_high_ceiling | INTACT-RA (cell regime) | 0.835 | +0.027 [+0.001, +0.059] 3/5 | positive (CI above 0) |
| S19_persistent_drift | INTACT-RA (per-tenant sweep) | 0.623 | +0.227 [+0.204, +0.248] 5/5 | positive (CI above 0) |

## What each scenario shows

- **S16_high_ceiling:** INTACT-RA-Agentic vs INTACT-RA positive (CI above 0); INTACT-RA vs B3 negative (CI below 0).
- **S19_persistent_drift:** INTACT-RA-Agentic vs INTACT-RA positive (CI above 0); INTACT-RA vs B3 positive (CI above 0).

**INTACT-RA-Agentic beats INTACT-RA with a CI above zero in 2 of 2 scenario(s) where both were run.** Any claim of winning 'in every scenario' requires this to read 2 of 2.