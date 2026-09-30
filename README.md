# INTACT-RA-Agentic

Multi-tenant xApp conflict mediation for the O-RAN near-real-time RIC, with
**sensitivities that stay current**.

This single document describes the whole system: what it models, how every
controller in the comparison works, what the agent observes and when it
acts, how results are measured, and how to read them. The companion
`RUNBOOK.md` gives the exact commands, in order. `docs/WORKED_EXAMPLE.md`
walks through one real decision by hand, and `docs/METRICS.md` defines
every metric.

---

## 1. What problem this addresses

Several xApps, belonging to several tenants, each ask the RIC to change
radio controls — here, PRB reservations. Their requests conflict: two may
write the same knob (**C1**), or a tenant's requests may together exceed
what it is entitled to or what the cell can supply (**C2**). An arbiter
must decide, every control epoch, which requests to execute.

**INTACT-RA** does this by predicting each candidate portfolio's effect on
every tenant's intent margin from a table of **cross-sensitivities** — how
much each control moves each intent — and refusing any portfolio whose
robust lower bound would push a currently-satisfied intent below its
safety floor. That table is calibrated offline and then frozen.

**INTACT-RA-Agentic** keeps INTACT-RA's decision procedure and replaces the
frozen table with one that tracks the plant as it changes. The headline
claim is that this improves intent fulfilment over INTACT-RA, in a
scenario where current sensitivities genuinely matter.

---

## 2. Scope: the conditions under which the claim is made

The claim is **not** that current sensitivities help in every RAN. The
investigation behind this package showed, repeatedly and by measurement,
that in many plausible configurations they buy nothing — and that it is
easy to build a benchmark in which no controller, however good, can show a
difference. The headline scenario, `configs/scenarios/S16_high_ceiling.yaml`,
is therefore built to satisfy stated conditions, each established by an
arbiter-free measurement **before** any controller was scored on it. These
conditions are the scope, and they should be reported with any result.

1. **Hard slicing** (`ran.allocation: dedicated`). Each tenant's PRB
   reservation is dedicated to it; a reserved PRB it does not use is not
   lent to other slices, and an unreserved PRB is idle. Under the
   alternative — a work-conserving shared remainder — releasing every
   reservation into the shared pool is optimal whenever load peaks do not
   coincide, and arbitration becomes pointless (measured: all-admit scored
   0.941, above every controller).
2. **A finite pool** (`arbiter.c2_cell_pool: true`). C2 requires both each
   tenant's committed reservations to fit its envelope and all
   reservations together to fit the 106-PRB pool. Envelopes are oversold
   relative to the pool, as operators do.
3. **Service-defined intents.** Each tenant must deliver at least 95% of
   its offered traffic, and meet a class delay bound at p90 (eMBB 100 ms,
   enterprise 30 ms, mMTC 300 ms). Targets are not percentiles of the
   uncontrolled plant; percentile targets make doing nothing look good by
   construction.
4. **Jointly feasible, individually shifting need.** Two tenants' loads
   swing out of phase (2.0× ↔ 0.3×), so their combined need fits the pool
   at almost every moment, but which of them needs the capacity flips. A
   fixed split is therefore wrong most of the time.
5. **A constant-load inward migration.** The third tenant's users move from
   the cell edge toward the site at unchanged offered load, so its need
   collapses while its load — the variable INTACT-RA's table is indexed
   by — does not move.

Measured on this scenario with no controller at all (`scripts/control_gain.py`,
three seeds): doing nothing meets **72%** of intents; a controller that
re-partitions the pool from each tenant's true current need meets **92%**.
All six intents are attainable within their envelopes
(`scripts/intent_attainability.py`). These two numbers — the floor and the
ceiling — are what make the comparison meaningful: doing nothing is clearly
inadequate, and good control is clearly achievable.

---

## 3. Repository layout

```
intact_agentic/
  types.py            Tenant, Intent, Claim, Decision, Registry
  config.py           YAML base + scenario merge, overrides, fingerprint
  methods.py          every controller in the comparison, declaratively
  experiment.py       THE epoch loop -- one loop runs every method
  telemetry.py        CSV sinks and the metric catalogue
  xapps.py            the black-box xApps and the shared proposal stream
  gates.py            scenario-validity gates
  ran/
    channel.py        3GPP TR 38.901 path loss, shadowing, fading,
                      3D antenna pattern, CQI/MCS, BLER, HARQ
    mobility.py       random waypoint, persistent placement, group drift
    traffic.py        on/off sessions, phase-shifted load profiles
    simulator.py      the plant: allocation, queues, delay, actuation lag,
                      reconfiguration transients, clone() and reseed()
  arbiter/
    margins.py        intent margins and fulfilment
    sensitivity.py    StaticSensitivity (frozen), OnlineSensitivity (RLS),
                      KalmanSensitivity (the agent's), ScrambledSensitivity
                      (a diagnostic control), the offline sweep
    regime.py         cell, per-tenant load, learned, oracle, fixed
    core.py           C1/C2, pool-conserving transfers, prediction, the
                      robust safety floor, scoring, ordered execution
    oracle.py         true slopes by paired finite differences on clones
  agent/
    twin.py           the digital-twin calibration loop
    context_head.py   GCN estimate of resource pressure and coverage, with
                      online model selection against an analytic estimate
    candidate_policy.py  PPO top-K portfolio proposer
    supervisor.py     staleness detection and calibration probes
    gcn.py, nn.py     NumPy networks with hand-derived gradients
  report/             metrics, figures, automatic interpretation

configs/base.yaml     defaults for everything
configs/scenarios/    S16 is the headline scenario; S0-S15 document the
                      investigation that led to it and are kept runnable

scripts/              see RUNBOOK.md for what each does and in what order
docs/                 WORKED_EXAMPLE.md, METRICS.md
```

Everything is NumPy; there is no deep-learning framework dependency.
Gradients are derived by hand and checked against central finite
differences in `scripts/selftest.py`.

---

## 4. The simulator

A single cell at 3.5 GHz with **106 PRBs** (40 MHz at 30 kHz SCS), six
interfering neighbour sites, and three tenants plus the host. One **slot**
is 10 ms; one **control epoch** is 16 slots (160 ms), split into a
pre-decision half (8 slots, from which the arbiter sees the state) and a
post-decision half (8 slots, in which the decision's effect is measured).

**Radio.** 3GPP TR 38.901 UMa path loss with LOS probability; spatially
correlated shadowing; speed-dependent fast fading; a 3D antenna pattern
with electrical downtilt; thermal noise and neighbour-cell interference;
CQI → MCS → spectral efficiency per 3GPP 38.214; a sigmoid BLER curve with
HARQ retransmissions.

**Traffic** is strictly exogenous: on/off sessions per UE, scaled by a
per-tenant load profile. No control ever changes offered traffic.

**Allocation** (`ran.allocation`). Four models are implemented, because the
choice decided whether any benchmark could discriminate between
controllers:

| mode | behaviour | consequence |
|---|---|---|
| `legacy` | each slice capped at its quota; scale-down only when demand exceeds the pool | with slack in the pool, raising a cap costs nobody anything |
| `shares` | work-conserving weighted-fair water-filling | a tenant's slope depends on everyone's backlog |
| `dedicated` | normalised dedicated reservations; unreserved PRBs idle | **hard slicing** — used by S16 |
| `dedicated_shared` | as dedicated, remainder shared max-min | releasing everything becomes optimal |

**Actuation.** Every executed write reprograms the slice: affected slices
lose 35% of capacity for 2 of the 16 slots. Knobs follow commanded values
through a first-order lag. The plant keeps both the **commanded** controls
(what policy committed) and the **actual** lagged ones (what the radio is
doing); decisions reason from commanded values.

**Two facilities matter for rigorous evaluation.** `clone()` copies the full
plant state *including* its random stream, so two clones stepped
identically see identical traffic, fading and mobility — this is what makes
paired, noise-cancelled comparisons possible. `reseed()` gives a clone, and
every stochastic sub-model, a fresh stream, which is what makes replicates
genuinely independent. (An early version reseeded only the top level;
every "replicate" then silently replayed the same future.)

---

## 5. The controllers

Every method uses the same plant, the same xApps and the same proposal
stream, and all mediated methods share the same arbiter core. What differs
is only which components are plugged in (`intact_agentic/methods.py`).

### B0 — all-admit (`all-accept`)

No mediation, exactly as the original code defines it: every xApp's raw
request is written, in claim order; later writes see the state earlier
ones produced; violating writes still land. **C1** is counted when a knob
was already written that epoch (last writer wins, and each write pays its
own reconfiguration transient). **C2** is counted when a tenant's committed
total after the write exceeds its envelope, or the pool is exceeded. The
RAN honours over-commitment, and the finite pool shaves every tenant.

### All-reject

Executes nothing. It holds the provisioned allocation for the whole run,
so its score is the **floor**: what doing nothing buys.

### B3 — value arbitration

As the original code defines it (mode `value_only`): for each claim,
`V = Σᵢ s(regime_now, param, i) · Δν`, **unweighted**, read at **one
cell-level regime** shared by every intent; the admissible subset with the
largest total value is admitted directly, with **no safety stage**.

### INTACT-RA

The frozen published method: the same unweighted linear signed-margin
objective as B3 (frozen as "INTACT-RA-lean, no contract weights"), but
with the table read at each intent's **own tenant's** measured-load regime,
and a **safety stage** — the robust floor in §6.9. Its table comes from the
offline sweep, which scales every tenant's load together, exactly as the
original.

### INTACT-RA-Agentic

INTACT-RA's decision procedure with its sensitivity table, context and
candidate generation replaced, as described in §6.

### Oracle (reference, not a method)

The same arbiter and decision rule as INTACT-RA, reading **true** local
slopes measured by paired finite differences on clones of the live plant,
refreshed every epoch for the knobs that have a proposal, with the same
uncertainty as the frozen table. It answers one question: *what would
INTACT-RA's decision rule achieve with perfect, current sensitivities?*
It is **not** an upper bound on what is attainable — its slopes are raw
two-replicate measurements taken fresh every epoch, and a smoothed
estimate can beat them at a single decision. The demand-tracking ceiling
in §2 is the statement of what is attainable.

### Ablations

`agentic-noTwin`, `agentic-noLearnedRegime`, `agentic-noDRL`,
`agentic-noProbe`, `agentic-noOnline` and `agentic+analyticPrior` each
change exactly one component of INTACT-RA-Agentic. `agentic-rls-v1` is the
superseded first estimator, kept so earlier results remain reproducible.
The per-tenant offline sweep is reported as a separate ablation.

---

## 6. INTACT-RA-Agentic in detail

### 6.1 The epoch, step by step

Every method runs this loop (`Experiment.step_epoch`):

1. **Arrivals** — activate any tenant whose arrival epoch has come.
2. **Pre-decision half** — step the plant 8 slots; read each intent's
   margin `g_before`.
3. **Context** — pass the latest telemetry to the sensitivity model;
   estimate each intent's regime.
4. **Proposals** — poll every xApp once. Each proposal is cached, so every
   method sees an identical stream.
5. **Probe** — the supervisor may inject one bounded calibration probe as a
   synthetic claim, which passes the same filters as any xApp's.
6. **Candidates** — the PPO proposer offers a top-K shortlist of
   portfolios; the no-action portfolio and a deterministic greedy chain are
   always added, so a poor shortlist can never stop the arbiter acting.
7. **Arbitration** — for each candidate: C1; C2 with pool-conserving
   transfers; predicted effect and uncertainty; the robust safety floor;
   utility. The best admissible candidate wins.
8. **Execution** — releases first, raises last; C1/C2 re-checked at each
   write.
9. **Post-decision half** — step the plant 8 slots; read `g_after`.
10. **Learning** — the Kalman tracker updates from the observed margin
    changes; the context head trains; the PPO proposer trains (burn-in
    only).
11. **Digital twin** — measure fresh slopes on an imperfect copy of the
    plant and fold them into the tracker. These inform the **next** epoch.
12. **Record** — metrics, telemetry and, periodically, a checkpoint.

### 6.2 What the agent observes

Only what a RIC can: per-tenant KPIs from the pre-decision half
(throughput, served ratio, delay percentiles, BLER, spectral efficiency,
PRBs used, offered traffic), cell KPIs, the xApp proposals, and the
reservations it has itself committed. It never reads the true slopes, the
true pressure label at decision time, or the plant's future random draws.

### 6.3 What state it maintains

- **Kalman posteriors** per (regime, intent): slopes θ and their covariance
  C, plus a per-intent estimate of plant-noise variance.
- **Context-head** network weights, and the recent accuracy of both the
  network and the analytic pressure estimate for each tenant.
- **PPO policy** weights (frozen during evaluation).
- **Supervisor** history for probe budgeting and staleness.

### 6.4 How candidates are generated

With few claims, exhaustive enumeration is cheap; with many it is not,
which is what the PPO top-K proposer is for. Its state is a GCN embedding
of the session/slice graph plus 14 features per claim; it outputs an
independent inclusion probability per claim and returns the argmax
portfolio plus sampled alternatives. It is trained during burn-in, with a
reward penalising search regret (utility lost versus the exhaustive
optimum) and latency, and **frozen with no exploration noise during
evaluation**. The no-action and greedy fallbacks are always appended.

### 6.5 How sensitivities are estimated and refreshed

`KalmanSensitivity`, a random-walk Kalman tracker. Per (regime, intent),
jointly over the knobs:

- **Prior**: θ₀ = the offline table's slope; C₀ = diag(σ₀²) with
  σ₀ = max(offline standard error, κ·|θ₀|, floor), κ = 0.5. The κ term is
  structural uncertainty: an offline slope is only valid near the operating
  point where it was measured.
- **Measurement noise** R: the variance of the margin change in epochs with
  **no** write — measured plant noise, not a guess.
- **Update** after an epoch with writes x:
  S = xᵀCx + R; K = Cx/S; θ ← θ + K(y − xᵀθ); C ← C − KxᵀC.
  This only ever narrows uncertainty.
- **Drift**: C ← C + Q each epoch, so slopes may move.
- **Change detection**: if normalised innovations e²/S stay large, C is
  widened toward 4·C₀ so a genuine shift is learned quickly.
- **Twin observations**: each twin-measured slope is folded in as a direct
  observation of that coefficient, with its replicate variance inflated 4×
  for model distrust.

### 6.6 The digital twin

A safety-constrained arbiter does not excite its own knobs enough to learn
from its own writes: the states where learning matters most are exactly
those in which the safety floor refuses to act on an uncertain slope. The
twin breaks that deadlock. After each epoch it measures paired finite
differences, for the knobs that had proposals, on a **deliberately
imperfect** copy of the plant:

- its own random stream — no knowledge of the future;
- every UE position perturbed by N(0, 25 m) per axis;
- noise figure and neighbour-cell load miscalibrated by errors drawn once
  per run;
- a one-epoch lag before its measurements inform a decision.

Its fidelity is a scope condition, set under `agent.twin` in `base.yaml`.
Its compute runs in the slow calibration loop and is reported separately
(about 1 s per call), never folded into the near-RT decision latency.

### 6.7 Context: resource pressure

The regime is each tenant's **resource pressure**: offered demand divided
by the capacity of its own reservation, so pressure above 1 means the
reservation binds. Bands: L0 below 0.9, L1 up to 1.1, L2 above. A GCN head
estimates it from the session/slice graph; an analytic estimate computes it
from observable telemetry. **Each epoch, whichever has had the lower recent
error against realised pressure is used.** (The network once output exactly
0.0 for a tenant at true pressure 3+, while being trusted unconditionally;
the selection exists because of that.)

### 6.8 How uncertainty is represented

Each portfolio's predicted change is Δĝᵢ = Σⱼ ŝᵢⱼ Δνⱼ, with standard
deviation σᵢ = √(Σⱼ (σᵢⱼ Δνⱼ)²), counted **only over pairs the write is
predicted to affect** (|ŝ| ≥ `sigma_min`) or whose effect is unknown — as
the original INTACT safety check does. A missing entry means "refuse",
never "assume zero".

### 6.9 Safety and admissibility

A candidate is admissible only if:

1. **C1** — at most one write per knob;
2. **C2** — each tenant within its envelope and all reservations within the
   pool, after **pool-conserving transfers**: if a portfolio would
   overflow, its raises are scaled down to what its own releases free up
   (floored to the knob step), and a portfolio with no raise to shrink
   stays infeasible;
3. **the robust floor** — for every intent currently at or above its floor
   ε, g_now + Δĝ − β·σ ≥ ε, with β = 1.5.

The no-action portfolio is always admissible.

### 6.10 Burn-in versus evaluation

Epochs 0–99 are an unscored **burn-in**, in which every method runs and
learns. At epoch 100 every method is handed the **same** plant state — the
passive plant at that epoch, provisioned controls, identical random draws —
so damage done while learning is discarded and only what was learned
carries over. During evaluation the PPO proposer is frozen; the Kalman
tracker, context head, probes and twin keep adapting, because adapting is
the method under test. The inward migration begins at epoch 100.

---

## 7. How results are measured

**Primary: intent fulfilment (IF)** — the fraction of (intent, epoch) pairs
with non-negative margin, **unweighted**, because no controller here
optimises contract weights. The legacy π-weighted wIF is also recorded.

**Mean shortfall** — how far below target intents fall, since a fulfilment
count cannot tell a margin of −0.01 from one of −1.00.

**Causal safety.** A naive crossing count attributes any threshold crossing
to a controller that wrote in that epoch; about 70% of those were plant
noise. At every epoch the post-decision half is therefore also run on a
**paired copy of the plant with no write** and identical random draws. A
causal crossing requires that copy to have stayed above the floor. Causal
prediction error is measured the same way.

**C1 and C2 violations**, counted at write time, identically for every
method.

**Decision latency** (near-RT, per epoch) and **twin compute** (slow loop)
are reported separately.

---

## 8. Checkpointing and reproducibility

- Every run checkpoints periodically. A checkpoint restores the plant
  (including its random stream), the metrics and every learned component,
  and a resumed run is bit-identical to an uninterrupted one (asserted by
  the self-test).
- Seeds are explicit. Development seeds and held-out seeds are separate
  lists, and the held-out runner **refuses** to run if they overlap.
- Each held-out evaluation writes a manifest with the configuration
  fingerprint and **refuses to resume** if the configuration has changed,
  so results from two configurations cannot be mixed.
- Results persist per run, so an interrupted evaluation continues from
  where it stopped.

---

## 9. Reading the results

`scripts/heldout_report.py` produces `HELDOUT_REPORT.md`, which states for
each acceptance criterion **PASS**, **FAIL** or **INCONCLUSIVE**, with the
numbers behind each verdict, against thresholds fixed before any held-out
result was seen:

1. the oracle beats frozen INTACT-RA — current sensitivities matter;
2. INTACT-RA-Agentic beats INTACT-RA, with a paired 95% interval entirely
   above zero — the headline;
3. the improvement is not merely from writing more;
4. it holds on at least 80% of held-out seeds;
5. the real B3 is included (reported, not a gate);
6. useful control is necessary, and indiscriminate intervention harmful;
7. mediated methods respect C1/C2, and causal crossings are reported;
8. decision latency fits the near-RT budget, with twin compute separate.

The mechanism — the change in sensitivities, and the corresponding
difference in decisions — is shown by the figures, not the table. The
report never converts a failure into a pass, and it says so when there are
too few seeds for a verdict.

---

## 10. What this simulator does not model

No figure here should be read as evidence about handover, uplink,
per-subband CQI, MIMO rank adaptation, carrier aggregation, transport or
TCP behaviour, core-network delay, or energy in watts. The digital twin is
a perturbed copy of the same simulator: its imperfections are modelled, not
measured from a real network.
