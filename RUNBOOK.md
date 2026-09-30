# RUNBOOK

This runbook contains every command needed to reproduce the paper's results, in order, with what each produces and how to read
it. The design and reasoning behind each step are in `README.md`; this
document is only about what to do.

**Every step is marked:**

- **[AUTO]** — a script does all of it. You run one command and read the
  result it prints and writes.
- **[MANUAL]** — a decision is yours. The scripts give you the numbers;
  nothing is changed for you.

Run every command from the repository root — the directory that contains
`configs/`, `scripts/` and `intact_agentic/`.

Time estimates are for one CPU core.

---

## Step 0 — Set up  [MANUAL, once]

```bash
python3 --version                       # 3.10 or newer
python3 -m venv .venv
source .venv/bin/activate               # Windows: .venv\Scripts\activate
python3 -m pip install -r requirements.txt
```

No GPU and no deep-learning framework are needed. Everything is NumPy.

---

## Step 1 — Self-test  [AUTO]  ~1 min fast, ~10 min full

```bash
python3 scripts/selftest.py --fast
python3 scripts/selftest.py --epochs 80
```

**Expected:** every line reads `pass`, ending in `N passed, 0 failed`.

The full test must run for more epochs than the PPO proposer's warm-up
(`agent.policy.warmup_epochs`, 40), because its latency check measures the
method's operating regime after warm-up; if it does not, the test fails
and says so. Cold-start latency (during warm-up) is printed separately as
`[info]` or `[WARN]`. A `[WARN]` there is a known limitation, not a
failure: until the proposer is trained it scores the full candidate list,
which exceeds the 10 ms budget on scenarios with many claims. S16 keeps
warm-up inside its unscored burn-in; elsewhere, set
`agent.policy.cold_start_max_candidates` to bound it.

**What it checks, and why each matters:**

| check | a failure means |
|---|---|
| hand-derived gradients vs finite differences | a sign error in backpropagation — it trains to a worse optimum without crashing |
| physically admissible KPMs | throughput above Shannon capacity, BLER outside [0,1], delay below the processing floor |
| a write costs throughput | reconfiguration is free, so accepting everything would be optimal |
| xApps polled twice give the same answer | the arbiter would score one value and execute another |
| a missing table entry reports full uncertainty | the agent would treat "never measured" as "no effect" |
| same seed gives identical results | nothing is reproducible |
| telemetry on/off gives identical results | logging consumes random numbers, so logged and unlogged runs differ |
| a resumed run matches an uninterrupted one | checkpoints are lossy |
| zero C1/C2 violations for INTACT-RA | the arbiter admitted an infeasible portfolio |

**If anything fails, stop.** Every later number is suspect until it passes.

---

## Step 2 — Is the scenario worth benchmarking?  [AUTO]  ~10 min

These checks run **no controller at all**, so they cannot favour any
method. Run them before any benchmark, on the headline scenario and on any
scenario you create or edit.

### 2a. Floor and ceiling

```bash
python3 scripts/control_gain.py --scenario S16_high_ceiling \
    --seeds 20260925,20260926,20260927
```

- **Floor** — the provisioned allocation held untouched, exactly what
  all-reject does.
- **Ceiling** — a controller that re-partitions the pool every epoch from
  each tenant's true current need.

**Writes:** `results/S16_high_ceiling/_control_gain/CONTROL_GAIN.md`

**Expected on S16:** floor ≈ 0.72, ceiling ≈ 0.92, gap ≈ +0.20, both lines
reading `OK`.

**How to read it:** a small gap means doing nothing is nearly as good as
ideal control, so every controller fights for crumbs and a single bad write
loses to leaving the default alone. A low ceiling means the scenario is so
infeasible that even perfect control fails intents most of the time.
Either invalidates the benchmark.

### 2b. Can every intent be met at all?

```bash
python3 scripts/intent_attainability.py --scenario S16_high_ceiling
```

For each tenant it measures fulfilment at the provisioned allocation, at
the tenant's full envelope while the others yield, and with the whole
cell.

**Writes:** `results/S16_high_ceiling/_attainability/ATTAINABILITY.md`

**Expected on S16:** `All intents are attainable within their envelopes.`

**How to read it:** an intent unattainable even at its envelope attracts
capacity under a linear objective without ever turning it into
fulfilment, and corrupts every method's ranking. "Unmeetable even with the
whole cell" means the target itself is wrong; "unattainable within its
envelope" means the envelope is too small for the load.

### 2c. Optional deeper diagnostics  [AUTO]

| script | question it answers |
|---|---|
| `probe_zerosum.py` | when one tenant's quota rises, does anyone pay? (paired counterfactual) |
| `attainable_value.py` | does the best *fixed* setting change after the plant change? |
| `policy_headroom.py` | how much is an *adaptive* schedule worth, across seeds? |

Each accepts `--scenario` and writes its own report under `results/`.

---

## Step 3 — Does current sensitivity knowledge matter here?  [AUTO]  ~45 min

```bash
python3 scripts/oracle_diagnostic.py --scenario S16_high_ceiling \
    --seeds 20260925,20260926 --per-epoch
```

Runs the same arbiter with true current slopes (the oracle), the frozen
table, scrambled slopes and the agent, plus the naïve baselines.

**Writes:** `results/S16_high_ceiling/_oracle_diagnostic/ORACLE_DIAGNOSTIC.md`
with an explicit **CASE A / CASE B** verdict.

**How to read it:**

- **CASE A** (oracle − frozen INTACT-RA below 0.03): current sensitivities
  are not decision-relevant in this scenario. **Do not tune the agent
  against it** — there is nothing to win, and any gain found would be
  manufactured. Go back to Step 2.
- **CASE B**: perfect current knowledge is worth something, and improving
  the agent's estimation is justified.

Use `--per-epoch`: without it the oracle's slopes can be up to 20 epochs
old, which was measured to cost it about 0.06 IF and to make it strand
capacity — a staleness artefact, not a property of the method.

The offline sensitivity table INTACT-RA uses is built automatically on
first use and cached in `artifacts/prior_S16_high_ceiling.json`. Delete
that file to force a fresh sweep.

---

## Step 4 — Development runs  [AUTO]  ~10 min per agentic run

Use **only** the development seeds (20260925, 20260926, 20260927) for any
tuning.

One method, one seed:

```bash
python3 scripts/run_experiment.py --scenario S16_high_ceiling \
    --method intact-ra-agentic --seed 20260925
```

Any configuration value can be overridden from the command line:

```bash
python3 scripts/run_experiment.py --scenario S16_high_ceiling \
    --method intact-ra-agentic --seed 20260925 \
    --set agent.twin.pos_sigma_m=40
```

**Writes:** `results/S16_high_ceiling/<method>/seed<N>/summary.json` and
`epochs.csv`.

---

## Step 5 — Freeze  [MANUAL]

Before any held-out run, stop changing the method and the scenario.
Changing either in response to held-out numbers turns the held-out seeds
into development seeds, and the evaluation stops meaning anything. The
next step records a fingerprint of the configuration and refuses to
resume if it changes.

---

## Step 6 — Held-out evaluation  [AUTO]  ~2 hours for 5 seeds

```bash
python3 scripts/run_heldout.py --scenario S16_high_ceiling \
    --seeds 31001,31002,31003,31004,31005
```

Runs all six methods — all-reject, B0 all-admit, B3, INTACT-RA,
INTACT-RA-Agentic and the per-epoch oracle — on seeds never used in
development, seed by seed.

**Writes, into `results/S16_high_ceiling/heldout/`:**

- `manifest.json` — scenario, configuration fingerprint, seeds, freeze time
- `runs.jsonl` — one line per completed (method, seed)
- `<method>/seed<N>/` — per-run summary and per-epoch records

**Built-in refusals:**

- held-out seeds overlapping the development seeds → refuses to start;
- configuration fingerprint changed since the manifest was written →
  refuses to resume;
- a different seed list from the manifest's → refuses to resume.

### Resuming  [AUTO]

If the run is interrupted for any reason, **re-run the identical
command**. Completed (method, seed) pairs are skipped, and a partly
finished run resumes from its last checkpoint.

---

## Step 7 — Held-out report  [AUTO]  seconds

```bash
python3 scripts/heldout_report.py \
    --runs results/S16_high_ceiling/heldout/runs.jsonl \
    --manifest results/S16_high_ceiling/heldout/manifest.json \
    --out results/S16_high_ceiling/heldout
```

It works on partial data, so you can run it at any point during Step 6.

**Writes:** `HELDOUT_REPORT.md` — a results table and a verdict for each
acceptance criterion.

**How to read it:** each criterion reads **PASS**, **FAIL**,
**INCONCLUSIVE** (too few seeds for a verdict — fewer than 3) or
**REPORTED** (shown, not gated). The final line gives the overall verdict
on the headline claim. The thresholds were fixed before any held-out
result was seen, and the script never converts a failure into a pass.

| criterion | what it tests |
|---|---|
| 1 | the oracle beats frozen INTACT-RA — current sensitivities matter |
| 2 | **INTACT-RA-Agentic beats INTACT-RA, paired 95% interval above zero** |
| 3 | the gain is not merely from writing more |
| 4 | it holds on at least 80% of held-out seeds |
| 5 | Agentic versus the real B3 (reported) |
| 6a / 6b | control is necessary; indiscriminate intervention is harmful |
| 7 | mediated methods have zero C1/C2 violations; causal crossings shown |
| 8 | decision latency within the near-RT budget; twin compute separate |

---

## Step 8 — Ablations  [AUTO]  ~1 hour per ablation for 5 seeds

Run on the **same held-out seeds**, into a separate output directory so the
main manifest is untouched:

```bash
python3 scripts/run_heldout.py --scenario S16_high_ceiling \
    --seeds 31001,31002,31003,31004,31005 \
    --methods agentic-noTwin,agentic-noLearnedRegime,agentic-noDRL \
    --out results/S16_high_ceiling/heldout_ablations
```

Each ablation removes exactly one component of INTACT-RA-Agentic; the drop
in fulfilment is that component's contribution.

---

## Step 8b — The robustness study  [AUTO]  ~4–5 hours

This tests the claim that no single frozen INTACT-RA configuration is safe
across plants, while INTACT-RA-Agentic matches the best one on each.

**It uses two plants in the same simulator:**

| | S16_high_ceiling | S18_oversubscribed |
|---|---|---|
| PRB reservations provisioned | 106 for a 106-PRB pool | **118** for a 106-PRB pool |
| C2 at admission | envelopes **and** the pool | envelopes only (original INTACT) |
| what a release does | strands capacity (hard slicing) | first reduces over-subscription |
| everything else | identical | identical |

**Run it:**

```bash
python3 scripts/robustness_study.py \
  --reuse S16_high_ceiling=results/S16_high_ceiling/heldout,results/S16_high_ceiling/heldout_cell,results/S16_high_ceiling/heldout_ablations
```

The `--reuse` line uses the S16 results the package already contains, so
only S18 is computed. Omit it to recompute both. The run is resumable:
re-run the identical command after an interruption.

**What it does, automatically:**

1. **Preflight** — for each scenario, the arbiter-free floor/ceiling and
   attainability checks from Step 2. No controller runs.
2. **Held-out runs** — all eight methods on the five held-out seeds:
   all-reject, B0, B3, INTACT-RA (per-tenant, as published), INTACT-RA (cell
   regime), INTACT-RA (per-tenant sweep), INTACT-RA-Agentic and the oracle.
3. **Verdict** — writes `results/_robustness/ROBUSTNESS_REPORT.md` and two
   figures, with a verdict on four criteria fixed in the script before any
   result:

| criterion | what it requires |
|---|---|
| R1 | both scenarios pass the validity checks |
| R2 | per-tenant INTACT-RA significantly beats the cell variant in one scenario, and the reverse in the other |
| R3 | INTACT-RA-Agentic is never significantly worse than the best frozen configuration |
| R4 | in each scenario, INTACT-RA-Agentic significantly beats the configuration you would have chosen by calibrating on the other |

**How to read it:** the last paragraph reads **SUPPORTED**, **NOT
SUPPORTED** (naming the failed criteria) or **INCONCLUSIVE** (too few
seeds). If R2 fails, the two plants do not favour different frozen
configurations and cannot test the claim at all — report S16 on its own
terms instead.

**Known before you run it [MANUAL — read this]:** S18's floor/ceiling gap
was measured at **+0.162**, below the +0.20 validity threshold, so **R1 is
expected to fail** unless you accept that deviation. The script will not
relax the threshold for you. S18 still has a high ceiling (0.909) and every
intent is attainable; whether a +0.16 gap is acceptable is your decision,
and it must be stated if you report the result.

## Step 8c — The factor walk  [AUTO]  ~10 min per variant per seed

Finds which property of a plant makes a per-tenant regime beat a cell-level
one, by transplanting properties of S8 (per-tenant wins) into S16 (valid,
cell-level wins) one at a time. Frozen methods only; development seeds only.

```bash
python3 scripts/factor_walk.py --factors shared_knobs,loads,migration_out,no_burnin,percentile
python3 scripts/factor_walk.py --factors percentile+no_burnin --seeds 20260925,20260926
```

Factors: `shared_knobs`, `host_intent`, `loads`, `migration_out`,
`no_burnin`, `percentile`; join with `+` to combine. It writes
`results/_factor_walk/FACTOR_WALK.md`, which labels each variant **PER-TENANT
WINS**, **CELL WINS** or **no clear order**, reports whether control beats
doing nothing there, and names any candidate per-tenant plant. A candidate
must then pass Step 2's validity checks and a held-out evaluation before
any claim is made. The run is resumable.

## Step 8d — Lean versus full INTACT-RA-Agentic  [AUTO]  ~40 min

`agentic-lean` removes both the PPO proposer (exhaustive scoring instead)
and the learned GCN context (the per-tenant load regime instead), keeping
the Kalman tracker, the digital twin and the calibration probes:

```bash
python3 scripts/run_heldout.py --scenario S16_high_ceiling \
    --seeds 31001,31002,31003,31004,31005 --methods agentic-lean \
    --out results/S16_high_ceiling/heldout_lean --allow-code-change
python3 scripts/heldout_report.py \
    --runs results/S16_high_ceiling/heldout/runs.jsonl \
    --manifest results/S16_high_ceiling/heldout/manifest.json \
    --ablations results/S16_high_ceiling/heldout_lean/runs.jsonl \
    --out results/S16_high_ceiling/heldout_lean
```

The ablation table in the resulting `HELDOUT_REPORT.md` shows the lean
variant's fulfilment and its paired difference from the full method.

## Step 8e — Twin fidelity and calibration cadence  [AUTO]  ~5 h (--quick: ~3 h)

How good, and how fresh, must the digital twin be? Re-runs
INTACT-RA-Agentic on the held-out seeds with one twin setting changed at a
time — position error 50/100 m, noise-figure error 3 dB, neighbour-load
error 0.5, all three together, and calibration every 3/6/12 epochs — and
compares each with INTACT-RA, the cell-regime variant and the default twin.

```bash
python3 scripts/twin_sweep.py            # all 8 settings, 5 seeds
python3 scripts/twin_sweep.py --quick    # 3 seeds
python3 scripts/twin_sweep.py --only pos100,every6
```

Writes `results/S16_high_ceiling/twin_sweep/TWIN_SWEEP.md`, which states
automatically at which settings, if any, the advantage over INTACT-RA is
lost, and where INTACT-RA-Agentic becomes significantly worse than the
cell-regime variant. `every6` approximates a twin hosted in the non-RT RIC
pushing sensitivities over A1 about once a second; cadence is a proxy for
lag, not an explicit delay queue. Resumable; each setting has its own
fingerprinted manifest.

## Step 8f — Billing and overhead accounting  [AUTO]  ~1 min per frozen run, ~8 min per agentic run

**What it is.** Economic and overhead figures for every method, measured
from what each method already does. **No method is changed, and none was
designed to minimise cost.** It replays each recorded run deterministically
while observing the live plant, and **proves each replay is the recorded run**
by requiring its fulfilment to match exactly; any run that does not
reproduce is reported and excluded.

```bash
python3 scripts/accounting.py --scenario S16_high_ceiling \
  --runs-dirs results/S16_high_ceiling/heldout,results/S16_high_ceiling/heldout_cell
python3 scripts/accounting.py --scenario S19_persistent_drift \
  --runs-dirs results/S19_persistent_drift/robustness_heldout
```

Add `--methods` or `--seeds` to run a subset first. Resumable: finished
replays are cached in `accounting.jsonl`.

**Writes** `results/<scenario>/accounting/ACCOUNTING.md`, `fig_billing.png`
and `fig_overhead.png`:

| figure | meaning |
|---|---|
| reserved PRB-seconds, per tenant | capacity leased — what a tenant pays under hard slicing |
| used PRB-seconds | capacity actually scheduled (pay-per-use) |
| idle leased PRB-seconds | paid for but carried nothing |
| unsold pool PRB-seconds | pool capacity no tenant reserved |
| delivered gigabits | traffic carried (pay-per-traffic) |
| PRB-seconds per fulfilled intent-epoch | cost-effectiveness |
| E2 control messages per epoch | one control request per executed write (signalling) |
| reconfiguration loss | capacity removed by reconfiguration transients, measured by the simulator, as % of the epoch's capacity |
| candidates scored, decision p95 | near-RT computation |
| twin calls and seconds | slow-loop computation (INTACT-RA-Agentic only) |

**How to read it [MANUAL — important].** A lower bill is **not**
automatically better. A method that strands capacity leases less and so
looks cheaper while failing intents — check unsold pool PRB-seconds and
fulfilment beside it. The fair comparison is **cost per fulfilled
intent-epoch**, shown against fulfilment in `fig_billing.png`.

## Step 9 — The worked example trace  [AUTO]  ~5 min

```bash
python3 scripts/trace_decision.py
```

Re-runs scenario S16 on development seed 20260925 and records one real decision end to end — telemetry, proposals, slopes and
uncertainties, feasibility, predicted effects, safety checks, the executed
writes, the outcome and the full Kalman state — to `results/S16_high_ceiling/worked_example/trace.json`. It is deterministic.
`docs/WORKED_EXAMPLE.md` walks through such a trace by hand.

---

## What each output directory contains

```
artifacts/prior_<scenario>.json      the frozen offline table (INTACT-RA's)
results/<scenario>/
  _control_gain/CONTROL_GAIN.md      floor and ceiling (Step 2a)
  _attainability/ATTAINABILITY.md    per-intent attainability (Step 2b)
  _oracle_diagnostic/                CASE A/B verdict (Step 3)
  <method>/seed<N>/                  development runs (Step 4)
  heldout/                           manifest, runs, HELDOUT_REPORT.md
  heldout_ablations/                 ablations on the held-out seeds
```

---

## Troubleshooting

**`REFUSING TO RESUME: configuration fingerprint changed`.** Something in
the configuration changed after the held-out run started. Either restore
it, or start a fresh output directory with `--out` — never mix the two.

**`REFUSING: held-out seeds overlap development seeds`.** Choose held-out
seeds that were never used for tuning.

**A background run stopped without an error.** Re-run the identical
command; it resumes. On a shared machine, start long runs with `nohup`
(Linux/macOS) so that closing the terminal does not stop them.

**All-reject beats every controller.** The scenario gives control nothing
to win. Check Step 2a: if the gap is small, the scenario — not the
controllers — needs changing.

**All-admit beats every controller.** Check that `ran.allocation` is
`dedicated` and `arbiter.c2_cell_pool` is `true`. With a shared remainder,
releasing every reservation becomes optimal and arbitration is pointless.

**The oracle loses to all-reject.** Check it was run with `--per-epoch`.
Slopes that are many epochs old make it strand capacity.

**INTACT-RA-Agentic makes almost no writes.** Check `agent.twin.enabled`
is `true` for the scenario. Without the twin, a safety-constrained arbiter
rarely excites its own knobs enough to learn from its own writes.

**Runs are slow.** INTACT-RA-Agentic and the oracle each take about ten
minutes per 500-epoch seed, dominated by twin and oracle rollouts. The
twin's compute is in the slow loop; the near-RT decision itself takes
under a millisecond.
