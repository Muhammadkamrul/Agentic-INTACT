# A worked example: one real INTACT-RA-Agentic decision, by hand

Every number below was recorded from an actual run — scenario
`S16_high_ceiling`, development seed 20260925, epoch 118 (the first scored
epoch after the 100-epoch burn-in in which the agent executed more than one
write). Nothing is invented or rounded to make the arithmetic neater. You
can reproduce each step with a calculator, and regenerate the trace itself
with the command at the end.

The example is chosen because it shows two mechanisms at once: a **digital
twin calibration probe** and a **pool-conserving transfer**. It also
contains a prediction that turned out badly wrong, and shows how the
estimator corrects itself — which is the point of an online estimator.

---

## Step 0 — The setting

The cell has **106 PRBs** under **hard slicing**: each tenant's reservation
is dedicated to it, and a PRB left unreserved is idle, not shared.

Six intents, three tenants (from the scenario file):

| intent | tenant | KPI | target | direction |
|---|---|---|---|---|
| i1 | T1 | served ratio | 0.95 | higher is better |
| i2 | T1 | delay p90 | 100 ms | lower is better |
| i3 | T2 | served ratio | 0.95 | higher is better |
| i4 | T2 | delay p90 | 30 ms | lower is better |
| i5 | T3 | served ratio | 0.95 | higher is better |
| i6 | T3 | delay p90 | 300 ms | lower is better |

Every intent has safety floor **ε = 0.02**. The arbiter's robustness
coefficient is **β = 1.5**, and it charges an **action cost of 0.0003** per
write.

---

## Step 1 — Telemetry and margins

At the start of epoch 118 the **committed reservations** (commanded
controls) are:

| knob | tenant | value (PRB) |
|---|---|---|
| quota_T1 | T1 | 32 |
| quota_T1b | T1 | 16 |
| quota_T2 | T2 | 21 |
| quota_T2b | T2 | 28 |
| quota_T3 | T3 | 9 |
| **total** | | **106** — the pool is full |

From the pre-decision half of the epoch, the measured **margins**
g = d·(KPI − target)/|target| are:

| i1 | i2 | i3 | i4 | i5 | i6 |
|---|---|---|---|---|---|
| 0.067 | 0.309 | 0.053 | 0.933 | 0.159 | 0.988 |

All six are above their floor, so all six are **protected**: no admitted
portfolio may push any of them below 0.02 in its robust lower bound.

## Step 2 — Context

The context estimator reports every intent in regime **L0C0** — low
resource pressure, not coverage-limited. (Pressure is demand divided by the
capacity of the tenant's *own reservation*; L0 means below 0.9, i.e. the
reservation is not binding.) So every slope below is read from the L0C0
cells of the sensitivity model.

## Step 3 — Proposals

Five xApps proposed a write this epoch, and the supervisor injected one
calibration probe as a synthetic claim:

| claim | xApp | knob | current | requested | dose |
|---|---|---|---|---|---|
| j1 | srv_T1 | quota_T1 | 32 | 30 | −2 |
| j3 | srv_T2 | quota_T2 | 21 | 20 | −1 |
| j4 | lat_T2 | quota_T2b | 28 | 22 | −6 |
| j5 | srv_T3 | quota_T3 | 9 | 6 | −3 |
| j6 | lat_T3 | quota_T3 | 9 | 6 | −3 |
| _probe | supervisor | quota_T1b | 16 | 17.75 → **18** | **+2** |

Two points worth noticing. j5 and j6 both write `quota_T3`, so choosing
both would be a **C1** conflict. And the probe's request of 17.75 is
projected onto the knob's 1-PRB grid, giving 18. The probe passes through
exactly the same filters as any xApp claim; it gets no special treatment.

## Step 4 — Current sensitivity estimates

The Kalman tracker's current estimates (slope, and its standard deviation
σ = √C_jj) for the pairs that matter. Every other pair has a slope below
the model's `sigma_min` and is treated as **not affected** — it contributes
neither predicted change nor uncertainty, exactly as the original INTACT
safety check does.

| knob → intent | slope ŝ | σ |
|---|---|---|
| quota_T1b → i2 | +0.00646 | 0.00743 |
| quota_T3 → i2 | −0.00239 | 0.00424 |
| quota_T3 → i5 | +0.00180 | 0.00041 |

These are *posterior* estimates: they started from the offline table and
have been updated by earlier twin measurements and live observations.

## Step 5 — Candidate portfolios and C2 feasibility

The arbiter scores candidate portfolios. The two that matter here:

**Candidate A: the probe alone, {_probe}.** It would raise the total
reservation from 106 to 106 + 2 = **108 > 106**. There is no release in the
portfolio to fund it, so the pool-conserving transfer rule cannot resize it
into feasibility.
→ **Rejected: C2.**

**Candidate B: {_probe, j6}.** The raise is funded by T3's release:
106 + 2 − 3 = **105 ≤ 106**. The raise already fits, so no resizing is
needed (the recorded admitted doses are exactly +2 and −3).
→ **C2 feasible.** C1 is also satisfied: one write each to `quota_T1b` and
`quota_T3`.

## Step 6 — Predicted effect of Candidate B

Predicted change in each margin is Σ_j ŝ_ij · dose_j:

- **i2 (T1 delay):** 0.00646 × (+2) + (−0.00239) × (−3)
  = 0.01292 + 0.00717 = **+0.0201**
- **i5 (T3 served ratio):** 0.00180 × (−3) = **−0.0054**
- **i1, i3, i4, i6:** no affected pair → **0**

## Step 7 — Uncertainty

Predicted standard deviation is √(Σ_j (σ_ij · dose_j)²):

- **i2:** √[(0.00743 × 2)² + (0.00424 × 3)²]
  = √[(0.01486)² + (0.01272)²]
  = √[0.0002208 + 0.0001618] = √0.0003826 = **0.0196**
- **i5:** 0.00041 × 3 = **0.0012**
- all others: **0**

## Step 8 — The robust safety check

For each protected intent, the lower bound g_low = g_now + Δĝ − β·σ must
stay at or above ε = 0.02:

- **i2:** 0.309 + 0.0201 − 1.5 × 0.0196 = 0.309 + 0.0201 − 0.0294
  = **0.2997 ≥ 0.02** ✓
- **i5:** 0.159 − 0.0054 − 1.5 × 0.0012 = 0.159 − 0.0054 − 0.0018
  = **0.1518 ≥ 0.02** ✓
- **i1, i3, i4, i6:** unaffected, so g_low = g_now, all above 0.02 ✓

→ **Candidate B is admissible.**

## Step 9 — Utility and selection

Utility is the unweighted mean predicted margin gain across the six
intents, minus the action cost:

(0.0201 + (−0.0054)) / 6 − 2 × 0.0003
= 0.0147 / 6 − 0.0006
= 0.00245 − 0.0006 = **0.00185**

The recorded top candidates:

| portfolio | utility | admissible |
|---|---|---|
| {_probe} | 0.00185 | no — C2 |
| **{_probe, j6}** | **0.00185** | **yes** |
| {} (no action) | 0.00000 | yes |
| {j6} | −0.00000 | yes |
| {j5} | −0.00000 | yes |
| {j3} | −0.00018 | yes |
| {_probe, j4, j5} | −0.00067 | yes |
| {j4} | −0.00252 | yes |

**{_probe, j6} wins.** Note that the probe alone scores the same utility —
the release on its own is predicted to have essentially no net value — but
it is infeasible without the release to fund it. The release exists in the
winning portfolio *because* it makes the raise possible. That is what a
transfer is.

Every unselected claim receives an explicit **REJECT** record (j1, j3, j4,
j5), so the audit trail shows what was refused, not only what was done.

## Step 10 — Execution

The mediated portfolio is applied **releases first, raises last**, so no
intermediate state commits more than the pool:

1. `quota_T3`: 9 → 6 (the release; the pool goes to 103)
2. `quota_T1b`: 16 → 18 (the raise; the pool goes to 105)

At each write the C1 and C2 checks run again, and both pass. Each write
triggers the plant's reconfiguration transient on the affected slice.

## Step 11 — What actually happened

Margins measured in the post-decision half of the epoch:

| | i1 | i2 | i3 | i4 | i5 | i6 |
|---|---|---|---|---|---|---|
| before | 0.067 | 0.309 | 0.053 | 0.933 | 0.159 | 0.988 |
| after | 0.186 | 0.816 | 0.025 | 0.610 | −0.164 | 0.644 |
| observed change | +0.119 | **+0.507** | −0.028 | −0.323 | **−0.323** | −0.344 |
| predicted change | 0 | +0.020 | 0 | 0 | −0.005 | 0 |

**Read this honestly.** The direction was right for the two intents the
decision was predicted to affect: T1's delay improved (i2) and T3's served
ratio got worse (i5). But the *magnitudes* were far off — T3's served ratio
fell by 0.32 against a predicted 0.005, and **it crossed below zero**, so
this intent is no longer fulfilled.

Part of each observed change is the decision; part is the plant moving on
its own. The estimator's measured plant-noise variance — estimated from
epochs in which *nothing* was written — tells us how much:

| intent | i1 | i2 | i3 | i4 | i5 | i6 |
|---|---|---|---|---|---|---|
| noise variance R | 0.020 | 1.227 | 0.044 | 0.297 | 0.045 | 0.311 |
| noise s.d. √R | 0.14 | 1.11 | 0.21 | 0.54 | 0.21 | 0.56 |

So i2's +0.507 is well within its own noise (s.d. 1.11), and the drops in
i4 and i6 — which this decision could not have caused — are within theirs.
**i5's −0.32 is about 1.5 standard deviations**: plausibly part decision,
part noise. T3 was left with only 6 reserved PRBs, which is tight even in
its light phase.

(The benchmark's *causal* metrics separate these cleanly, by re-running the
post-decision half on a paired copy of the plant with no write and
identical random draws. The agent does not have that luxury; it must learn
from the noisy observation, which is the next step.)

## Step 12 — The sensitivity update

The Kalman filter now updates every affected cell, jointly over the knobs
written. For each intent i in regime cell (L0C0, i):

- dose vector **x**: +2 on `quota_T1b`, −3 on `quota_T3`
- innovation **e** = observed − predicted
- innovation variance **S** = xᵀ C x + R_i
- gain **K** = C x / S
- update **θ ← θ + K·e**, **C ← C − K xᵀ C**

Take **i5**, where the miss was largest. The filter's state for cell
(L0C0, i5), recorded at decision time, for the two knobs written:

| | quota_T1b | quota_T3 |
|---|---|---|
| slope θ | −0.000073 | +0.001800 |
| C (covariance), row quota_T1b | 5.4809×10⁻⁸ | 1.97×10⁻¹⁵ |
| C (covariance), row quota_T3 | 1.97×10⁻¹⁵ | 1.6507×10⁻⁷ |

Note that the filter tracks **every** coefficient, including
quota_T1b → i5. The *arbiter* ignores that pair (its slope is below
`sigma_min`, so it is "not affected"), but the *filter* still updates it.
Plant-noise variance for i5: **R = 0.044653**.

- dose vector x = (+2, −3) on (quota_T1b, quota_T3)
- predicted change xᵀθ = 2 × (−0.000073) + (−3) × 0.001800 = **−0.005545**
- observed change y = **−0.322515**
- innovation e = y − xᵀθ = **−0.316970** (the recorded residual)
- xᵀCx = 2² × 5.4809×10⁻⁸ + (−3)² × 1.6507×10⁻⁷ + 2 × 2 × (−3) × 1.97×10⁻¹⁵
  = 2.192×10⁻⁷ + 1.4856×10⁻⁶ + (negligible) = **1.7049×10⁻⁶**
- S = xᵀCx + R = 1.7049×10⁻⁶ + 0.044653 = **0.044654** — almost entirely
  plant noise
- gain K = Cx / S:
  - for quota_T1b: (5.4809×10⁻⁸ × 2) / 0.044654 = **+2.4548×10⁻⁶**
  - for quota_T3: (1.6507×10⁻⁷ × (−3)) / 0.044654 = **−1.1090×10⁻⁵**
- slope change K·e:
  - quota_T3 → i5: −1.1090×10⁻⁵ × (−0.316970) = **+3.515×10⁻⁶**, so the
    slope moves from 0.001800 to **0.001803**
  - quota_T1b → i5: +2.4548×10⁻⁶ × (−0.316970) = **−7.78×10⁻⁷**

That is tiny, and deliberately so: one surprising observation, most of
which the filter attributes to plant noise because the noise variance
(0.0447) dwarfs the predicted variance from slope uncertainty
(1.7×10⁻⁶), is not allowed to overturn a well-established estimate.

For comparison, the same update for **i2**, where the observation was
also far from the prediction: its noise variance is R = 1.2268 — T1's p90
delay is very noisy epoch to epoch — so S = 1.2272 and the T1b slope moves
only from +0.006461 to +0.006505.

What *does* move the estimate quickly is either **repetition** — the
change detector widens C back toward its prior when normalised innovations
e²/S stay large over several epochs — or a **twin measurement**. At the end
of this same epoch the digital twin re-measures the slopes of the knobs
that had proposals, on a deliberately imperfect copy of the plant (its own
random stream, UE positions wrong by ~25 m, miscalibrated noise figure and
neighbour load), and folds each measured slope in as a direct observation
of that coefficient. Those observations have far smaller variance than a
single noisy live epoch, so they are what actually recalibrates the table
between decisions — informing epoch 119 onward, never epoch 118 itself.

---

## What this example demonstrates

1. **C2 is a real constraint.** The probe alone was infeasible; only a
   portfolio that funded it from a release could execute it.
2. **Pool-conserving transfers** move capacity between tenants in a single
   decision without ever overcommitting the pool.
3. **Uncertainty is counted only where a write can have an effect**, so
   isolated tenants do not block each other through phantom risk.
4. **The robust floor** is checked against a lower confidence bound, not the
   point prediction.
5. **Execution is ordered** releases-first so no intermediate state
   violates C2.
6. **The estimator is humble about single observations** and relies on
   repetition and twin calibration to move — which is what makes a large
   miss like i5's a signal rather than a catastrophe.

## Reproducing the trace

```bash
python3 scripts/trace_decision.py
```

re-runs scenario `S16_high_ceiling` on development seed 20260925 and
writes `results/S16_high_ceiling/worked_example/trace.json`, containing
every value used above — including the full Kalman state at decision time.
The trace is deterministic: re-running it captures the same decision at
epoch 118. A copy of the trace used for this document is in
`docs/worked_example_trace.json`.
