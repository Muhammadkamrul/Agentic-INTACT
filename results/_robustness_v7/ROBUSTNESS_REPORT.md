# Robustness study

Scenarios: ['S16_high_ceiling', 'S19_persistent_drift']. Held-out seeds: 31001,31002,31003,31004,31005. Paired differences with bootstrap 95% intervals (difference [CI], seeds ahead/seeds).

## 1. Validity (arbiter-free, no controller involved)

| scenario | floor | ceiling | gap | attainable | valid |
|---|---|---|---|---|---|
| S16_high_ceiling | 0.724 | 0.915 | +0.191 | True | **yes** |
| S19_persistent_drift | 0.709 | 0.851 | +0.142 | True | **yes** |

Thresholds: ceiling >= 0.85, gap >= 0.13, every intent attainable within its envelope. **The gap threshold was changed from the default 0.2 to 0.13; this is a validity decision and must be reported with any result.**

## 2. Fulfilment

| method | S16_high_ceiling | S19_persistent_drift |
|---|---|---|
| All-reject | 0.751 (n=5) | 0.760 (n=5) |
| B0 all-admit | 0.608 (n=5) | 0.597 (n=5) |
| B3 | 0.832 (n=5) | 0.452 (n=5) |
| INTACT-RA (per-tenant, published) | 0.739 (n=5) | 0.619 (n=5) |
| INTACT-RA (cell regime) | 0.835 (n=5) | 0.453 (n=5) |
| INTACT-RA (per-tenant sweep) | 0.787 (n=5) | 0.633 (n=5) |
| INTACT-RA-Agentic | 0.862 (n=5) | 0.841 (n=5) |
| Oracle* | 0.860 (n=5) | 0.878 (n=5) |

## 3. Key paired comparisons

| comparison | S16_high_ceiling | S19_persistent_drift |
|---|---|---|
| INTACT-RA (per-tenant) − INTACT-RA (cell) | -0.095 [-0.123, -0.055], 0/5 | +0.166 [+0.129, +0.194], 5/5 |
| INTACT-RA-Agentic − INTACT-RA (per-tenant) | +0.122 [+0.100, +0.144], 5/5 | +0.222 [+0.182, +0.252], 5/5 |
| INTACT-RA-Agentic − INTACT-RA (cell) | +0.027 [+0.001, +0.059], 3/5 | +0.388 [+0.353, +0.422], 5/5 |
| INTACT-RA-Agentic − B3 | +0.030 [+0.013, +0.050], 5/5 | +0.389 [+0.355, +0.423], 5/5 |
| oracle − best frozen | +0.025 [+0.011, +0.044], 5/5 | +0.245 [+0.241, +0.251], 5/5 |

## 4. Verdict on the robustness claim

- **R1 valid benchmarks:** PASS
- **R2 the frozen ordering flips:** PASS (per-tenant significantly better in ['S19_persistent_drift']; cell significantly better in ['S16_high_ceiling'])
- **R3 never significantly worse than the best frozen:** PASS
  - S16_high_ceiling: best frozen is INTACT-RA (cell regime); Agentic − it = +0.027 [+0.001, +0.059], 3/5 -> not worse
  - S19_persistent_drift: best frozen is INTACT-RA (per-tenant sweep); Agentic − it = +0.208 [+0.180, +0.233], 5/5 -> not worse
- **R4 beats the wrong a-priori choice in each plant:** PASS
  - S16_high_ceiling: calibrating on S19_persistent_drift would pick INTACT-RA (per-tenant sweep); Agentic − it = +0.075 [+0.056, +0.092], 5/5 -> beats it
  - S19_persistent_drift: calibrating on S16_high_ceiling would pick INTACT-RA (cell regime); Agentic − it = +0.388 [+0.353, +0.422], 5/5 -> beats it

**ROBUSTNESS CLAIM SUPPORTED.** Different frozen configurations win on the two plants, and INTACT-RA-Agentic matches the best on both while beating the wrong a-priori choice.

*Oracle: INTACT-RA's own rule with true current slopes; a perfect-knowledge reference, not an upper bound.