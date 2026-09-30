# Held-out evaluation report

Scenario `S16_high_ceiling`, configuration fingerprint `0a3a7f0ab28a2e80`, frozen 2026-09-26 14:52:53. Held-out seeds [31001, 31002, 31003, 31004, 31005]; development seeds [20260925, 20260926, 20260927] were never used here.

## Results

Means over the 5 seed(s) completed by EVERY method: [31001, 31002, 31003, 31004, 31005].

| method | seeds | IF | shortfall | C1 | C2 | causal crossings | writes/ep | p95 latency (ms) |
|---|---|---|---|---|---|---|---|---|
| oracle | 5 | **0.8596** | 0.126 | 0.0 | 0.0 | 6.4 | 0.55 | 2.27 |
| INTACT-RA-Agentic | 5 | **0.8617** | 0.145 | 0.0 | 0.0 | 3.0 | 0.45 | 0.63 |
| B3 | 5 | **0.8318** | 0.199 | 0.0 | 0.0 | 2.6 | 0.14 | 0.13 |
| INTACT-RA | 5 | **0.7394** | 0.382 | 0.0 | 0.0 | 3.6 | 0.75 | 2.26 |
| all-reject | 5 | **0.7514** | 0.377 | 0.0 | 0.0 | 0.0 | 0.00 | 0.10 |
| B0 all-admit | 5 | **0.6076** | 0.423 | 208.2 | 11.6 | 126.6 | 2.92 | 0.16 |

## Acceptance gates

| gate | verdict | evidence |
|---|---|---|
| 1. Oracle beats frozen INTACT-RA (current sensitivities matter) | **PASS** | +0.1202 over 5 seed(s), 95% CI [+0.0963, +0.1433]; threshold 0.03 |
| 2. INTACT-RA-Agentic beats INTACT-RA (headline) | **PASS** | +0.1222 over 5 seed(s), 95% CI [+0.1004, +0.1441]; requires CI entirely above 0 |
| 3. Improvement is not merely from accepting more actions | **PASS** | Agentic writes 0.45/epoch vs INTACT-RA 0.75; the method that writes most (B0, 2.92/epoch) scores 0.6076, the lowest of all |
| 4. Held-out consistency (Agentic > INTACT-RA per seed) | **PASS** | wins on 5/5 held-out seed(s); need >= 80% |
| 5. Real B3 included; Agentic vs B3 (reported, not a gate) | **REPORTED** | +0.0299 over 5 seed(s), 95% CI [+0.0127, +0.0495], Agentic ahead on 5/5 |
| 6a. Useful control is necessary (best mediated - all-reject) | **PASS** | INTACT-RA-Agentic 0.8617 vs all-reject 0.7514 (+0.1103); threshold 0.03 |
| 6b. Indiscriminate intervention is harmful (B0) | **PASS** | B0 0.6076 vs all-reject 0.7514; C1 208.2, C2 11.6 violations per run |
| 7. Safety: mediated methods respect C1/C2; causal crossings reported | **PASS** | total C1+C2 across mediated methods 0.0; causal crossings per run: Agentic 3.0, INTACT-RA 3.6 |
| 8. Near-RT decision latency (twin compute reported separately) | **PASS** | Agentic p95 decision latency 0.63 ms (budget 10.0 ms); digital-twin calibration 852 ms per call, in the slow loop |

## Ablations (same held-out seeds, paired)

Each row removes or changes exactly one thing. Differences are paired over the seeds both methods completed, with bootstrap 95% intervals.

| variant | seeds | IF | vs INTACT-RA-Agentic | vs INTACT-RA |
|---|---|---|---|---|
| agentic-lean | 5 | 0.8500 | -0.0117 [-0.0228, -0.0005], above on 1/5 | +0.1106 [+0.0907, +0.1320], above on 5/5 |

## Verdict

**HEADLINE SUPPORTED** on the held-out seeds, within the scope conditions of the scenario.

Gate 9 (mechanism) is demonstrated by the slope-trajectory and decision figures, not by this table.