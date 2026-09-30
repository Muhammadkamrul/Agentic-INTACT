"""Automatic interpretation.

Turns a results directory into prose that states what happened, whether
the headline claim is supported, and -- importantly -- where it is NOT.

The design rule for everything in this module: a finding that goes
against the proposal is reported in the same voice and at the same length
as one that supports it.  An interpreter that only knows how to announce
success is a press release, and it will eventually let a broken result
through.  Every verdict here is a comparison against a stated threshold
with the actual number printed next to it, so a reader can disagree with
the threshold without having to re-derive the number.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


def _f(x, nd=4, na="n/a") -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return na
    return na if not np.isfinite(v) else f"{v:.{nd}f}"


def _verdict(ok: bool) -> str:
    return "SUPPORTED" if ok else "NOT SUPPORTED"


# ---------------------------------------------------------------------------
def interpret_benchmark(summaries: Dict[str, Dict], cfg: Dict,
                        scenario: str,
                        ci: Optional[Dict] = None,
                        paired: Optional[Dict] = None,
                        gate_report=None) -> str:
    g = cfg.get("gates", {}) or {}
    out: List[str] = []
    A = out.append

    A(f"# Interpretation: {scenario}")
    A("")
    note = cfg.get("_scenario_note")
    if note:
        A(f"**Scenario.** {' '.join(str(note).split())}")
        A("")

    ar = summaries.get("all-reject", {})
    aa = summaries.get("all-accept", {})
    ra = summaries.get("intact-ra", {})
    ag = summaries.get("intact-ra-agentic", {})
    sp = summaries.get("static-priority", {})

    # ---- 1. is the benchmark non-degenerate at all? ----------------------
    A("## 1. Is this benchmark non-degenerate?")
    A("")
    A("Before comparing controllers, the scenario has to establish that "
      "control was needed and that indiscriminate control was not enough. "
      "If either degenerate baseline wins, nothing else on this page means "
      "anything.")
    A("")
    controllers = {k: v for k, v in summaries.items()
                   if k not in ("all-accept", "all-reject", "oracle")}
    best_name = max(controllers, key=lambda m: controllers[m].get("wIF", -9),
                    default=None)
    if best_name and ar and aa:
        best = controllers[best_name]
        cg = best.get("wIF", np.nan) - ar.get("wIF", np.nan)
        qg = best.get("wIF", np.nan) - aa.get("wIF", np.nan)
        tc = float(g.get("control_gain_min", 0.03))
        tq = float(g.get("conflict_gain_min", 0.03))
        A(f"- **Control gain** (best controller minus all-reject): "
          f"**{_f(cg)}** against a required {_f(tc, 3)}. "
          f"{_verdict(cg >= tc)}.")
        A(f"  Doing nothing scores {_f(ar.get('wIF'))}; the best controller "
          f"({best.get('label', best_name)}) scores "
          f"{_f(best.get('wIF'))}.")
        A(f"- **Conflict gain** (best controller minus all-accept): "
          f"**{_f(qg)}** against a required {_f(tq, 3)}. "
          f"{_verdict(qg >= tq)}.")
        A(f"  Accepting every proposal scores {_f(aa.get('wIF'))} while "
          f"writing {_f(aa.get('writes_per_epoch'), 2)} controls per epoch "
          f"and causing {int(aa.get('safety_crossings', 0))} safety "
          f"crossings.")
        A("")
        if cg >= tc and qg >= tq:
            A("Both degenerate baselines lose, so the outcome depends on "
              "*which* proposals are executed rather than on how many. "
              "Arbitration is doing real work here.")
        else:
            A("**At least one degenerate baseline is competitive.** Every "
              "comparison below should be treated as uninformative until "
              "the scenario is fixed: a plant where doing nothing (or "
              "doing everything) is near-optimal cannot discriminate "
              "between mediation policies.")
        A("")

    # ---- 2. the headline comparison --------------------------------------
    A("## 2. Agentic against frozen INTACT-RA")
    A("")
    if ra and ag:
        d = ag.get("wIF", np.nan) - ra.get("wIF", np.nan)
        thr = float(g.get("agentic_gain_min", 0.01))
        A(f"- wIF: INTACT-RA {_f(ra.get('wIF'))} -> "
          f"INTACT-RA-Agentic {_f(ag.get('wIF'))} "
          f"(**{d:+.4f}**, required {_f(thr, 3)}). {_verdict(d >= thr)}.")
        if paired and "wIF" in paired:
            p = paired["wIF"]
            A(f"  Paired bootstrap over {p.get('n', 0)} seeds: "
              f"mean {_f(p.get('mean'))}, 95% CI "
              f"[{_f(p.get('lo'))}, {_f(p.get('hi'))}], "
              f"positive on {_f(100 * p.get('positive_frac', np.nan), 0)}% "
              f"of seeds.")
            if p.get("lo", -1) is not None and np.isfinite(p.get("lo", np.nan)) \
                    and p["lo"] <= 0 <= p.get("hi", 0):
                A("  **The interval spans zero.** The point estimate "
                  "favours the agentic method but the evidence does not "
                  "exclude no difference. Report it that way.")
        dc = ag.get("safety_crossings", np.nan) - ra.get("safety_crossings",
                                                         np.nan)
        A(f"- Safety crossings: {int(ra.get('safety_crossings', 0))} -> "
          f"{int(ag.get('safety_crossings', 0))} ({dc:+.0f}).")
        A(f"- Prediction error (MAE of predicted vs observed margin "
          f"change): {_f(ra.get('prediction_mae'))} -> "
          f"{_f(ag.get('prediction_mae'))}.")
        A(f"- Actuation: {_f(ra.get('writes_per_epoch'), 2)} -> "
          f"{_f(ag.get('writes_per_epoch'), 2)} writes per epoch.")
        A(f"- Decision latency p95: {_f(ra.get('latency_ms_p95'), 2)} ms -> "
          f"{_f(ag.get('latency_ms_p95'), 2)} ms, over "
          f"{_f(ra.get('candidates_mean'), 1)} -> "
          f"{_f(ag.get('candidates_mean'), 1)} candidates scored.")
        A("")
        mae_better = (ag.get("prediction_mae", 9) <
                      ra.get("prediction_mae", 9))
        if d < thr and mae_better:
            A("**Note the dissociation.** The agentic method predicts the "
              "plant better but does not convert that into fulfilment. "
              "The usual cause is that an online estimator's honest "
              "standard error is wider than a clean offline sweep's, and a "
              "robust arbiter turns wider uncertainty into fewer writes. "
              "Check the writes-per-epoch column: if the agentic method is "
              "writing substantially less, it is being penalised for "
              "admitting what it does not know, and the scenario has not "
              "yet made the frozen table wrong enough to repay that.")
            A("")

    # ---- 3. is it beating the simple baseline for the right reason? ------
    if sp and ra:
        A("## 3. Against the naive priority baseline")
        A("")
        A(f"Static priority reaches wIF {_f(sp.get('wIF'))} by writing "
          f"{_f(sp.get('writes_per_epoch'), 2)} controls per epoch and "
          f"causing {int(sp.get('safety_crossings', 0))} safety crossings. "
          f"INTACT-RA reaches {_f(ra.get('wIF'))} with "
          f"{_f(ra.get('writes_per_epoch'), 2)} writes and "
          f"{int(ra.get('safety_crossings', 0))} crossings.")
        if ra.get("wIF", 0) >= sp.get("wIF", 0):
            ratio = (sp.get("safety_crossings", 0)
                     / max(ra.get("safety_crossings", 1), 1))
            A(f"The deliberative controller is ahead on fulfilment *and* "
              f"causes {ratio:.0f}x fewer safety crossings, so the "
              f"comparison is not a value-for-risk trade.")
        else:
            A("**Static priority is ahead on wIF.** That is a problem for "
              "the framing, even though it buys the score with far more "
              "writes and far more crossings. Either the safety metric "
              "carries the argument and wIF should not be the headline, "
              "or the arbiter is too conservative in this plant.")
        A("")

    # ---- 4. ablations -----------------------------------------------------
    abl = {k: v for k, v in summaries.items() if k.startswith("agentic-")}
    if abl and ag:
        A("## 4. Which component earns its place?")
        A("")
        base = ag.get("wIF", np.nan)
        rows = sorted(abl.items(), key=lambda kv: kv[1].get("wIF", 0))
        for name, s in rows:
            d = s.get("wIF", np.nan) - base
            verdict = ("removing it HURTS, so it is load-bearing"
                       if d < -0.005 else
                       "removing it HELPS, so it is not earning its place"
                       if d > 0.005 else
                       "no measurable effect in this scenario")
            A(f"- `{name}`: wIF {_f(s.get('wIF'))} ({d:+.4f}) -- {verdict}.")
        A("")
        useless = [n for n, s in abl.items()
                   if s.get("wIF", 0) - base > 0.005]
        if useless:
            A(f"**{len(useless)} component(s) are net-negative here.** A "
              f"component that the ablation cannot justify should be "
              f"removed from the proposal or restricted to the scenarios "
              f"where it does pay, not carried along because it is part of "
              f"the architecture diagram.")
            A("")

    # ---- 5. distance from the bound --------------------------------------
    oc = summaries.get("oracle")
    if oc and ag:
        gap = oc.get("wIF", np.nan) - ag.get("wIF", np.nan)
        tot = oc.get("wIF", np.nan) - ra.get("wIF", np.nan) if ra else np.nan
        A("## 5. How much is left on the table?")
        A("")
        A(f"With true finite-difference slopes the arbiter reaches "
          f"{_f(oc.get('wIF'))}. The agentic method is {_f(gap)} short of "
          f"that bound.")
        if np.isfinite(tot) and tot > 1e-9:
            A(f"Of the {_f(tot)} available from perfect estimation, online "
              f"learning recovered "
              f"{_f(100 * (ag.get('wIF', 0) - ra.get('wIF', 0)) / tot, 0)}%.")
        A("")

    # ---- 6. gates ----------------------------------------------------------
    if gate_report is not None:
        A("## 6. Scenario validity gates")
        A("")
        for r in gate_report.results:
            mark = "pass" if r.passed else ("FAIL" if r.critical else "warn")
            A(f"- [{mark}] `{r.name}`: {_f(r.value)} "
              f"(threshold {_f(r.threshold)}) {r.detail}")
        A("")
        A(f"**Scenario {'accepted' if gate_report.passed else 'REFUSED'}.**")
        A("")

    return "\n".join(out)


# ---------------------------------------------------------------------------
def interpret_ran(cell_stats: Dict[str, float],
                  tenant_stats: Dict[str, Dict[str, float]],
                  cfg: Dict) -> str:
    """Plain-language description of the plant the run actually produced."""
    out: List[str] = []
    A = out.append
    A("# The RAN the experiment actually ran on")
    A("")
    ran = cfg.get("ran", {})
    bw = ran.get("n_prb", 0) * ran.get("prb_bandwidth_hz", 0) / 1e6
    A(f"A single three-sector-equivalent cell at "
      f"{ran.get('carrier_ghz', 3.5)} GHz with {ran.get('n_prb')} PRBs "
      f"({bw:.1f} MHz), six interfering neighbours at "
      f"{(ran.get('channel', {}) or {}).get('isd_m', 500)} m inter-site "
      f"distance. One slot is {ran.get('slot_ms')} ms and one arbitration "
      f"epoch is {ran.get('pre_slots', 0) + ran.get('post_slots', 0)} slots "
      f"({(ran.get('pre_slots', 0) + ran.get('post_slots', 0)) * ran.get('slot_ms', 0):.0f} ms), "
      f"which is inside the near-real-time RIC control loop.")
    A("")

    util = cell_stats.get("prb_util_pct", np.nan)
    A("## Was the cell actually contended?")
    A("")
    A(f"Mean PRB utilisation was {_f(util, 1)}%.")
    if util < 45:
        A("**That is too low for the results to be about arbitration.** "
          "Below roughly half utilisation a tenant can usually be given "
          "what it asks for without taking it from anyone, so co-authorised "
          "claims stop competing and the envelope constraint is vacuous. "
          "Raise the offered load before drawing conclusions.")
    elif util > 96:
        A("**That is saturated.** At full utilisation every allocation is "
          "purely zero-sum and queue dynamics dominate, which exaggerates "
          "the value of any control that moves capacity around. Reduce the "
          "offered load.")
    else:
        A("That is the regime the benchmark needs: enough contention that "
          "granting one tenant's claim costs another, without the cell "
          "being so saturated that queueing dominates every effect.")
    A("")

    A("## Per-tenant operating point")
    A("")
    A("| tenant | SINR (dB) | CQI | MCS | SE (b/s/Hz) | PRBs | "
      "offered (Mb/s) | delivered (Mb/s/UE) | delay p50 (ms) | BLER |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for t, s in tenant_stats.items():
        A(f"| {t} | {_f(s.get('sinr_db'), 1)} | {_f(s.get('cqi'), 1)} | "
          f"{_f(s.get('mcs'), 1)} | {_f(s.get('spectral_efficiency'), 2)} | "
          f"{_f(s.get('prb_alloc'), 1)} | "
          f"{_f(s.get('offered_slice_mbps'), 2)} | "
          f"{_f(s.get('throughput_mbps'), 2)} | "
          f"{_f(s.get('delay_p50_ms'), 1)} | {_f(s.get('bler'), 3)} |")
    A("")

    sinrs = [s.get("sinr_db", np.nan) for s in tenant_stats.values()]
    if len(sinrs) > 1 and np.isfinite(sinrs).all():
        spread = float(np.nanmax(sinrs) - np.nanmin(sinrs))
        A(f"The spread in mean SINR across tenants is {spread:.1f} dB.")
        if spread < 4:
            A("**The tenants are radio-equivalent.** A coverage-limited "
              "versus capacity-limited distinction is what makes the "
              "channel axis of the regime space worth anything, and it is "
              "not present here. Separate the tenants' placements.")
        else:
            A("That is a genuine coverage-limited versus capacity-limited "
              "split, which is what the channel axis of the regime space "
              "is there to exploit.")
    A("")

    A("## What this simulator does not model")
    A("")
    A("Stated explicitly so that no figure is read as evidence about it: "
      "handover and mobility robustness, uplink, per-subband CQI "
      "reporting, MIMO rank adaptation, carrier aggregation, TCP or any "
      "transport-layer reaction, core-network or transport delay, and "
      "energy consumption in watts. Transmit power appears only as a "
      "control and a constraint, never as a modelled energy cost.")
    return "\n".join(out)


# ---------------------------------------------------------------------------
def interpret_drift(ratios: Dict[str, float], flips: int,
                    s_early: Dict, s_late: Dict,
                    threshold: float) -> str:
    out: List[str] = []
    A = out.append
    A("# Did the plant actually go stale?")
    A("")
    A("True local slopes measured twice on clones of the same trajectory, "
      "by paired finite differences with common random numbers, once in "
      "the calibration window and once late in the run. Only the elapsed "
      "plant evolution differs between the two measurements.")
    A("")
    if not ratios:
        A("**No slopes could be compared.** Nothing here supports a "
          "staleness claim.")
        return "\n".join(out)

    worst = sorted(ratios.items(), key=lambda kv: -kv[1])[:8]
    A("| coefficient | early | late | ratio | reversed? |")
    A("|---|---|---|---|---|")
    for k, r in worst:
        prm, _, iid = k.partition("->")
        e = s_early.get((prm, iid), np.nan)
        l = s_late.get((prm, iid), np.nan)
        rev = "**YES**" if (np.isfinite(e) and np.isfinite(l)
                            and e * l < 0) else "no"
        A(f"| `{k}` | {_f(e, 5)} | {_f(l, 5)} | {_f(r, 2)} | {rev} |")
    A("")
    if flips:
        A(f"**{flips} coefficient(s) reversed sign.** This is the strongest "
          f"form of staleness available: the frozen table does not merely "
          f"mis-estimate the size of an effect, it recommends the opposite "
          f"action, and it does so with the small standard error the "
          f"offline sweep legitimately measured. A controller that trusts "
          f"its calibration will act confidently and wrongly.")
    else:
        best = max(ratios.values())
        A(f"No sign reversals. The largest magnitude change is "
          f"{_f(best, 2)}x against a required {_f(threshold, 2)}x. "
          + ("That clears the bar, but a pure magnitude change is a weaker "
             "test than a reversal: a robust arbiter that widens its "
             "uncertainty can often survive it without adapting at all."
             if best >= threshold else
             "**That does not clear the bar.** An online estimator is "
             "being asked to beat a table that is still approximately "
             "right, and the honest expectation is that it will lose, "
             "because its in-run standard error is wider than the offline "
             "sweep's."))
    return "\n".join(out)
