"""Scenario-validity gates.

A benchmark result only means something if the benchmark could have come
out the other way.  These gates are run BEFORE the expensive experiments
and they REFUSE a scenario that cannot support the claim, rather than
letting it through and leaving the reader to discover the problem.

The gates encode, as executable checks, the failure modes this line of
work has actually hit:

  G1  workload identity      every method must see the same proposal
                             stream, or the comparison is not a comparison
  G2  arbitration winnable   at least one epoch must exist where the
                             choice of portfolio changes the outcome
  G3  drift materiality      the true slopes must actually move, by more
                             than the offline calibration's own claimed
                             precision -- otherwise an online estimator is
                             being asked to beat a correct table
  G4  regime disagreement    the per-tenant and cell-average labels must
                             disagree often enough for the label to matter
  G7  CONTROL GAIN           the best controller must beat all-reject.
                             This is the gate that a previous version of
                             this work failed: doing nothing beat every
                             controller, because the plant was stationary
                             and writing was free.
  G8  conflict gain          the best controller must beat all-accept, or
                             there is no conflict worth mediating
  G9  agentic gain           the agentic method must beat the frozen one,
                             or the scenario does not exercise staleness

G7 and G8 together are the non-degeneracy requirement: the answer must
be neither "always act" nor "never act".
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
@dataclass
class GateResult:
    name: str
    passed: bool
    value: float
    threshold: float
    detail: str = ""
    critical: bool = True

    def line(self) -> str:
        mark = "PASS" if self.passed else ("FAIL" if self.critical else "warn")
        return (f"  [{mark}] {self.name:<24} "
                f"value={self.value:+.4f} threshold={self.threshold:+.4f}"
                + (f"  {self.detail}" if self.detail else ""))


@dataclass
class GateReport:
    scenario: str
    results: List[GateResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results if r.critical)

    def add(self, r: GateResult) -> None:
        self.results.append(r)

    def text(self) -> str:
        head = (f"Scenario validity gates: {self.scenario}\n"
                + "=" * 66)
        body = "\n".join(r.line() for r in self.results)
        verdict = ("\nVERDICT: scenario ACCEPTED"
                   if self.passed else
                   "\nVERDICT: scenario REFUSED -- do not run the benchmark "
                   "on it.\n         A scenario that fails a critical gate "
                   "cannot support the claim,\n         and running it "
                   "anyway produces a number that looks like\n         "
                   "evidence and is not.")
        return f"{head}\n{body}\n{verdict}"

    def as_dict(self) -> Dict:
        return {"scenario": self.scenario, "passed": self.passed,
                "gates": [{"name": r.name, "passed": r.passed,
                           "value": r.value, "threshold": r.threshold,
                           "critical": r.critical, "detail": r.detail}
                          for r in self.results]}

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.as_dict(), fh, indent=2)


# ---------------------------------------------------------------------------
def check_workload_identity(runs: Dict[str, object]) -> GateResult:
    """G1: every DELIBERATIVE method saw the same amount of work.

    The degenerate baselines are excluded on purpose.  A controller that
    writes 1.6 controls per epoch drives the plant to a completely
    different operating point from one that writes nothing, and the xApps
    then ask for different things -- that divergence is the *consequence*
    of the policy, not evidence of an unfair experiment.  What this gate
    is for is catching an accidental asymmetry between controllers that
    are supposed to be facing the same problem.
    """
    counts = {}
    for name, ex in runs.items():
        if name in ("all-accept", "all-reject"):
            continue
        counts[name] = [r.n_proposed for r in ex.metrics.records]
    if len(counts) < 2:
        return GateResult("G1 workload identity", True, 0.0, 0.0,
                          "only one method", critical=False)
    n = min(len(v) for v in counts.values())
    ref_name = sorted(counts)[0]
    ref = counts[ref_name][:n]
    worst, worst_name = 0.0, ""
    for name, v in counts.items():
        d = float(np.mean(np.abs(np.array(v[:n]) - np.array(ref))))
        if d > worst:
            worst, worst_name = d, name
    # A controller changes the controls, which changes what the xApps ask
    # for next, so the streams cannot be bit-identical.  What must hold is
    # that no method is systematically offered MORE to work with.
    return GateResult("G1 workload identity", worst < 1.5, worst, 1.5,
                      f"largest mean gap vs {ref_name}: {worst_name}")


def check_arbitration_winnable(runs: Dict[str, object]) -> GateResult:
    """G2: portfolio choice must actually change the outcome somewhere."""
    spread = []
    for ex in runs.values():
        for r in ex.metrics.records:
            if np.isfinite(r.best_utility):
                spread.append(abs(r.best_utility - r.utility))
    if not spread:
        # fall back to candidate-count evidence
        cands = [r.candidates for ex in runs.values()
                 for r in ex.metrics.records]
        v = float(np.mean(cands)) if cands else 0.0
        return GateResult("G2 arbitration winnable", v > 2.0, v, 2.0,
                          "mean candidate count")
    v = float(np.mean(spread))
    return GateResult("G2 arbitration winnable", v > 1e-6, v, 1e-6,
                      "mean |best - chosen| utility")


def check_drift_materiality(ratios: Dict[str, float],
                            threshold: float,
                            sign_flips: int = 0) -> GateResult:
    """G3: the true slopes must move by more than the prior's precision.

    ``ratios`` maps a label to |slope_late| / |slope_early|.  A SIGN FLIP
    counts automatically as material regardless of magnitude: a table that
    recommends the opposite action is stale in the strongest sense
    available, even if the magnitude happens to be similar.
    """
    vals = [v for v in ratios.values() if np.isfinite(v)]
    best = max(vals) if vals else 0.0
    passed = bool(sign_flips > 0 or best >= threshold)
    return GateResult("G3 drift materiality", passed, best, threshold,
                      f"{sign_flips} sign flip(s); max |late/early| ratio")


def check_regime_disagreement(runs: Dict[str, object],
                              threshold: float) -> GateResult:
    """G4: per-tenant and cell-average labels must disagree often enough."""
    dis = []
    for ex in runs.values():
        for r in ex.metrics.records:
            labs = list(r.regimes.values())
            if len(labs) > 1:
                dis.append(0.0 if len(set(labs)) == 1 else 1.0)
    v = float(np.mean(dis)) if dis else 0.0
    return GateResult("G4 regime disagreement", v >= threshold, v, threshold,
                      "fraction of epochs with >1 distinct label")


def check_control_gain(summaries: Dict[str, Dict], threshold: float,
                       controllers: Sequence[str]) -> GateResult:
    """G7: the best controller must beat doing nothing.

    THE gate.  A plant where frozen controls are fine is a plant where no
    controller is needed, and every result on it is an artefact.
    """
    ar = summaries.get("all-reject", {}).get("wIF")
    best = max((summaries[m]["wIF"] for m in controllers
                if m in summaries and "wIF" in summaries[m]), default=None)
    if ar is None or best is None:
        return GateResult("G7 control gain", False, float("nan"), threshold,
                          "all-reject or controller missing")
    v = best - ar
    return GateResult("G7 control gain", v >= threshold, v, threshold,
                      f"best controller {best:.4f} vs all-reject {ar:.4f}")


def check_conflict_gain(summaries: Dict[str, Dict], threshold: float,
                        controllers: Sequence[str]) -> GateResult:
    """G8: the best controller must beat accepting everything."""
    aa = summaries.get("all-accept", {}).get("wIF")
    best = max((summaries[m]["wIF"] for m in controllers
                if m in summaries and "wIF" in summaries[m]), default=None)
    if aa is None or best is None:
        return GateResult("G8 conflict gain", False, float("nan"), threshold,
                          "all-accept or controller missing")
    v = best - aa
    return GateResult("G8 conflict gain", v >= threshold, v, threshold,
                      f"best controller {best:.4f} vs all-accept {aa:.4f}")


def check_agentic_gain(summaries: Dict[str, Dict],
                       threshold: float) -> GateResult:
    """G9: the agentic method must beat the frozen one on this scenario."""
    ra = summaries.get("intact-ra", {}).get("wIF")
    ag = summaries.get("intact-ra-agentic", {}).get("wIF")
    if ra is None or ag is None:
        return GateResult("G9 agentic gain", True, float("nan"), threshold,
                          "not both methods present", critical=False)
    v = ag - ra
    return GateResult("G9 agentic gain", v >= threshold, v, threshold,
                      f"agentic {ag:.4f} vs INTACT-RA {ra:.4f}")


def check_safety_separation(summaries: Dict[str, Dict]) -> GateResult:
    """A deliberative controller must be safer than accepting everything."""
    aa = summaries.get("all-accept", {}).get("safety_crossings")
    ra = summaries.get("intact-ra", {}).get("safety_crossings")
    if aa is None or ra is None:
        return GateResult("G10 safety separation", True, float("nan"), 1.0,
                          "methods missing", critical=False)
    v = float(aa) / max(float(ra), 1.0)
    return GateResult("G10 safety separation", v >= 2.0, v, 2.0,
                      f"all-accept {int(aa)} vs INTACT-RA {int(ra)} crossings")


# ---------------------------------------------------------------------------
def measure_slope_drift(cfg: Dict, registry, early_epoch: int = 20,
                        late_epoch: int = 600, params: Sequence[str] = (),
                        reps: int = 3, log=None):
    """Measure true local slopes early and late, on RAN clones.

    Paired finite differences with common random numbers, run on clones of
    the SAME trajectory at two points in time, so the only thing that
    differs between the two measurements is how far the plant has moved.
    """
    from .arbiter.margins import margin
    from .ran.simulator import RealisticRAN
    from .config import epoch_slots

    pre, post = epoch_slots(cfg)
    claims = registry.claims
    params = list(params) or sorted({c.param for c in claims.values()})
    dom = {c.param: c.domain for c in claims.values()}
    step = {c.param: c.max_step_frac for c in claims.values()}

    ran = RealisticRAN(cfg)
    snaps = {}
    for ep in range(max(late_epoch, early_epoch) + 1):
        ran.epoch = ep
        ran.step(pre + post, record=False)
        if ep in (early_epoch, late_epoch):
            snaps[ep] = ran.clone()

    def slopes_at(base):
        out = {}
        cur = base.current_controls()
        for prm in params:
            lo, hi = dom[prm]
            c0 = float(cur.get(prm, 0.5 * (lo + hi)))
            half = float(step.get(prm, 0.2)) * (hi - lo)
            a, b = max(lo, c0 - half), min(hi, c0 + half)
            if b - a < 1e-9:
                continue
            vals = {}
            for v in (a, b):
                acc = {i: [] for i in registry.intents}
                for r in range(reps):
                    pb = base.clone()
                    # reseed(), not a bare rng assignment: the sub-models hold
                    # their own reference, and without this every replicate
                    # replayed the same future, so the reported drift was
                    # measured on a single noise realisation per point.
                    pb.reseed(4242 + 31 * r)
                    pb.apply(prm, v)
                    pb._reconfig_left = {t: 0 for t in pb.slices}
                    pb._cell_reconfig_left = 0
                    pb.step(24, record=False)
                    k = pb.step(32, record=False)
                    for i, it in registry.intents.items():
                        g = margin(it, k)
                        if g is not None:
                            acc[i].append(g)
                vals[v] = {i: float(np.mean(x)) for i, x in acc.items() if x}
            for i in vals[a]:
                if i in vals[b]:
                    out[(prm, i)] = (vals[b][i] - vals[a][i]) / (b - a)
        return out

    s_early = slopes_at(snaps[early_epoch])
    s_late = slopes_at(snaps[late_epoch])
    ratios, flips = {}, 0
    for k in sorted(set(s_early) & set(s_late)):
        e, l = s_early[k], s_late[k]
        if abs(e) < 1e-6:
            continue
        ratios[f"{k[0]}->{k[1]}"] = abs(l) / abs(e)
        if e * l < 0 and abs(l) > 1e-4 and abs(e) > 1e-4:
            flips += 1
            if log:
                log(f"    SIGN FLIP  d(g_{k[1]})/d({k[0]}): "
                    f"{e:+.5f} -> {l:+.5f}")
    return ratios, flips, s_early, s_late


# ---------------------------------------------------------------------------
def run_gates(cfg: Dict, scenario: str, summaries: Dict[str, Dict],
              runs: Optional[Dict[str, object]] = None,
              drift: Optional[tuple] = None) -> GateReport:
    """Evaluate every gate and produce the report."""
    g = (cfg.get("gates", {}) or {})
    rep = GateReport(scenario)
    controllers = [m for m in summaries
                   if m not in ("all-accept", "all-reject", "oracle")]

    if runs:
        rep.add(check_workload_identity(runs))
        rep.add(check_arbitration_winnable(runs))
        rep.add(check_regime_disagreement(
            runs, float(g.get("regime_disagreement_min", 0.20))))
    if drift is not None:
        ratios, flips = drift
        rep.add(check_drift_materiality(
            ratios, float(g.get("drift_materiality_min", 2.0)), flips))
    rep.add(check_control_gain(summaries,
                               float(g.get("control_gain_min", 0.030)),
                               controllers))
    rep.add(check_conflict_gain(summaries,
                                float(g.get("conflict_gain_min", 0.030)),
                                controllers))
    rep.add(check_agentic_gain(summaries,
                               float(g.get("agentic_gain_min", 0.010))))
    rep.add(check_safety_separation(summaries))
    return rep
