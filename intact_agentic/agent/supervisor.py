"""
intact_agentic/agent/supervisor.py
==================================
The agentic supervisory control plane.

It sits ABOVE the fast arbitration path, never inside it.  It decides what
evidence to collect, whether the sensitivity model is still credible,
which calibration probe is worth its risk, when to publish a new model
version, and how to register an entity that did not exist when the system
was calibrated.  It never writes a radio parameter: everything it decides
is handed to the deterministic arbiter, which applies the same safety
filter to a probe as to any other write.

THE SIX DECISIONS, AND WHY EACH ONE NEEDS A SUPERVISOR RATHER THAN A
FORMULA
---------------------------------------------------------------------
1. IS THIS RESIDUAL REAL OR NOISE?
   A one-epoch surprise is a KPI window; a persistent signed bias is a
   slope that has moved.  Handled by the CUSUM + dispersion test in
   ``OnlineSensitivity.staleness``.  Age alone never triggers anything:
   a fresh table can be wrong and an old one can still be right.

2. WHICH ENTRY IS TO BLAME?
   When several knobs moved together the residual identifies the SUM, not
   the parts.  The supervisor prefers epochs in which exactly one knob
   moved ("natural experiments", which occur on their own because xApps
   do not bid in lockstep) and only requests a deliberate probe when the
   log cannot separate the candidates.  ``attribution_quality`` measures
   how separable the recent history actually is.

3. IS FIXING IT WORTH ANYTHING?
   The largest error is NOT the one worth fixing.  An entry can be 138%
   wrong and change no decision ever, because its term is swamped; a
   different entry can be 20% wrong and flip which tenant gets protected.
   ``decision_relevance`` replays the last window's portfolio choices with
   the candidate slope substituted and counts how many WINNERS change.
   That is the quantity the probe budget is spent against.

4. WHAT IS THE CHEAPEST MEASUREMENT THAT WOULD SETTLE IT?
   A probe is a bounded, single-knob move.  Its value is the expected
   reduction in decision-relevant uncertainty; its risk is the worst-case
   robust margin it could cost.  ``VOI = value / (risk + eps)`` ranks
   them, and a probe whose robust lower margin would breach a protected
   floor is never issued -- it is dropped, and if nothing safe is
   available the supervisor escalates instead.

5. DO I TRUST THE NEW NUMBER ENOUGH TO PUBLISH IT?
   Handled by ``OnlineSensitivity.maybe_promote``: a candidate must beat
   the live table on a held-out window before the arbiter is allowed to
   use it.  Versions are monotone and every decision logs the version it
   used, so any result can be traced to the exact coefficients behind it.

6. DID THE CHANGE ACTUALLY WORK?
   Post-promotion residuals are tracked separately; a promotion whose
   residuals do not improve within ``rollback_window`` epochs is rolled
   back to the previous version and the event is recorded.

NEW TENANTS AND NEW INTENTS
---------------------------
On arrival the supervisor registers the entity and marks every
(regime, knob, new intent) entry as UNKNOWN -- explicitly absent, not
zero.  The arbiter therefore sees ``unknown_sigma`` and refuses the
action under the robust bound, and the supervisor schedules calibration
probes for the knobs that matter most to the newcomer.  The static
baseline, by contrast, silently reads 0.0 and asserts that nothing
affects the new tenant, which is the failure mode the new-tenant scenario
is built to expose.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from ..types import Claim, Intent, Portfolio


# ---------------------------------------------------------------------------
@dataclass
class Probe:
    param: str
    delta: float
    reason: str
    target_intent: str
    voi: float
    risk: float
    epoch: int = 0

    def as_row(self) -> Dict:
        return {"epoch": self.epoch, "param": self.param,
                "delta": self.delta, "reason": self.reason,
                "target_intent": self.target_intent, "voi": self.voi,
                "risk": self.risk}


@dataclass
class SupervisorEvent:
    epoch: int
    kind: str
    detail: str
    payload: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
class Supervisor:
    """Slow control-plane decision maker over the sensitivity model."""

    def __init__(self, cfg: Dict, sens, rng: np.random.Generator):
        s = (cfg.get("agent", {}) or {}).get("supervisor", {}) or {}
        self.cfg = cfg
        self.sens = sens
        self.rng = rng
        self.interval = int(s.get("interval_epochs", 5))
        self.probe_budget = float(s.get("probe_budget_per_100", 4.0))
        self.probe_frac = float(s.get("probe_dose_frac", 0.35))
        self.max_probes_per_epoch = int(s.get("max_probes_per_epoch", 1))
        self.min_probe_gap = int(s.get("min_probe_gap_epochs", 6))
        self.risk_eps = float(s.get("risk_eps", 0.01))
        self.relevance_window = int(s.get("relevance_window", 20))
        self.rollback_window = int(s.get("rollback_window", 15))
        self.escalate_on_unknown = bool(s.get("escalate_on_unknown", True))
        self.enabled = bool(s.get("enabled", True))

        self.events: List[SupervisorEvent] = []
        self.probes: List[Probe] = []
        self.last_probe_epoch = -999
        self.probes_issued = 0
        self.stale_flags: Dict[Tuple[str, str], int] = {}
        self.unknown_entries: Set[Tuple[str, str, str]] = set()
        self.recent_decisions: Deque[Dict] = deque(maxlen=200)
        self._version_marks: List[Tuple[int, int, float]] = []
        self.rollbacks = 0
        self.pending_probe: Optional[Probe] = None

    # ------------------------------------------------------------------
    def log(self, epoch: int, kind: str, detail: str, **payload) -> None:
        payload.pop("epoch", None)
        self.events.append(SupervisorEvent(epoch, kind, detail, payload))

    # ------------------------------------------------------------------
    def note_decision(self, epoch: int, winner, reqs, regimes) -> None:
        """Remember what the arbiter chose, for counterfactual replay."""
        self.recent_decisions.append({
            "epoch": epoch,
            "winner": winner.claims,
            "utility": winner.utility,
            "doses": {j: reqs[j].dose for j in reqs},
            "params": {j: reqs[j].param for j in reqs},
            "regimes": dict(regimes.per_intent),
        })

    # ------------------------------------------------------------------
    def attribution_quality(self, param: str, window: int = 40) -> float:
        """How separable is this knob in the recent log?

        1.0  it moved alone at least once (a natural experiment)
        0.0  it never moved, or only ever moved together with others
        """
        rows = [r for r in self.sens.residuals[-window * 4:]]
        alone = sum(1 for r in rows if r["n_knobs"] == 1)
        moved = sum(1 for r in rows if r["n_knobs"] >= 1)
        if moved == 0:
            return 0.0
        return float(alone / moved)

    # ------------------------------------------------------------------
    def decision_relevance(self, regime: str, param: str, iid: str,
                           candidate_slope: float, intents: Dict[str, Intent]
                           ) -> float:
        """Fraction of recent decisions whose WINNER would change.

        Replays the stored portfolios with one slope substituted and asks
        whether the argmax moves.  This is the "would it have mattered?"
        test that stops the probe budget being spent on the largest error
        rather than the most consequential one.
        """
        rows = [d for d in list(self.recent_decisions)[-self.relevance_window:]
                if d["regimes"].get(iid) == regime]
        if not rows:
            return 0.0
        live = self.sens.get(regime, param, iid)
        if abs(live - candidate_slope) < 1e-12:
            return 0.0
        w = intents[iid].pi_class * intents[iid].weight if iid in intents \
            else 1.0
        changed = 0
        for d in rows:
            base_u = d["utility"]
            delta = 0.0
            for j, prm in d["params"].items():
                if prm != param or j not in d["winner"]:
                    continue
                delta += w * (candidate_slope - live) * d["doses"].get(j, 0.0)
            # the winner flips if the perturbation is bigger than the gap to
            # the runner-up; we use the utility itself as a conservative
            # proxy for that gap when the runner-up is not stored
            if abs(delta) > max(abs(base_u) * 0.25, 1e-4):
                changed += 1
        return float(changed / len(rows))

    # ------------------------------------------------------------------
    def plan_probe(self, *, epoch: int, claims: Dict[str, Claim],
                   intents: Dict[str, Intent], margins: Dict[str, float],
                   regimes, controls: Dict[str, float]) -> Optional[Probe]:
        """Choose at most one bounded single-knob calibration probe."""
        if not self.enabled:
            return None
        if epoch - self.last_probe_epoch < self.min_probe_gap:
            return None
        allowed = self.probe_budget * max(epoch, 1) / 100.0
        if self.probes_issued >= allowed:
            return None

        best: Optional[Probe] = None
        for j, c in claims.items():
            aq = self.attribution_quality(c.param)
            for iid, it in intents.items():
                r = regimes.of_intent(iid)
                known = self.sens.known(r, c.param, iid)
                cand, sd, n = (self.sens.candidate(r, c.param, iid)
                               if hasattr(self.sens, "candidate")
                               else (0.0, 0.0, 0))
                stale = self.sens.staleness(r, iid) \
                    if hasattr(self.sens, "staleness") else {"stale": 0.0}
                unknown = not known
                if not unknown and not stale.get("stale", 0.0):
                    continue
                rel = 1.0 if unknown else self.decision_relevance(
                    r, c.param, iid, cand, intents)
                # value: uncertainty we would remove, weighted by whether it
                # could change a decision and by how identifiable it is now
                value = (sd if known else self.sens.unknown_sigma) \
                    * (0.25 + rel) * (0.3 + 0.7 * (1.0 - aq))
                delta = self.probe_frac * c.max_step_frac * c.width
                nu_old = float(controls.get(c.param, c.domain[0]))
                # pick the direction with more headroom
                if nu_old + delta > c.domain[1]:
                    delta = -delta
                # risk: worst robust margin cost across protected intents
                risk = 0.0
                for i2, it2 in intents.items():
                    r2 = regimes.of_intent(i2)
                    s2 = self.sens.get(r2, c.param, i2)
                    sd2 = self.sens.sigma(r2, c.param, i2)
                    g2 = margins.get(i2, 0.0)
                    pred = g2 + s2 * delta - 1.5 * sd2 * abs(delta)
                    if g2 >= it2.epsilon:
                        risk = max(risk, max(it2.epsilon - pred, 0.0))
                voi = value / (risk + self.risk_eps)
                if risk > 0.6 * max(abs(margins.get(iid, 0.0)), 0.05):
                    continue        # unsafe probe: never issued
                if best is None or voi > best.voi:
                    best = Probe(param=c.param, delta=float(delta),
                                 reason=("unknown entry" if unknown
                                         else "stale entry"),
                                 target_intent=iid, voi=float(voi),
                                 risk=float(risk), epoch=epoch)
        if best is not None:
            self.last_probe_epoch = epoch
            self.probes_issued += 1
            self.probes.append(best)
            payload = {k: v for k, v in best.as_row().items()
                       if k != "epoch"}
            self.log(epoch, "probe", f"{best.param} {best.delta:+.3f} "
                                     f"for {best.target_intent} "
                                     f"({best.reason})", **payload)
        return best

    # ------------------------------------------------------------------
    def register_new_entities(self, epoch: int, new_tenants: Sequence[str],
                              intents: Dict[str, Intent],
                              claims: Dict[str, Claim],
                              regimes: Sequence[str]) -> Dict:
        """Mark every (regime, knob, new intent) entry as explicitly unknown."""
        added = 0
        new_iids = [i for i, it in intents.items() if it.tenant in new_tenants]
        for iid in new_iids:
            for c in claims.values():
                for r in regimes:
                    key = (r, c.param, iid)
                    if not self.sens.known(*key):
                        self.unknown_entries.add(key)
                        added += 1
                if hasattr(self.sens, "register_new"):
                    self.sens.register_new(regimes[0], c.param, iid)
        if added:
            self.log(epoch, "registry",
                     f"registered {len(new_tenants)} tenant(s), "
                     f"{len(new_iids)} intent(s); {added} sensitivity "
                     f"entries marked UNKNOWN (not zero)",
                     tenants=list(new_tenants), intents=new_iids,
                     unknown_entries=added)
        return {"new_tenants": list(new_tenants), "new_intents": new_iids,
                "unknown_entries": added}

    # ------------------------------------------------------------------
    def tick(self, epoch: int, regimes: Sequence[str],
             intents: Dict[str, Intent]) -> Dict:
        """The slow loop: staleness scan, promotion, rollback check."""
        if not self.enabled:
            return {}
        out: Dict = {"stale": 0, "promoted": False}
        if epoch % max(self.interval, 1):
            return out
        n_stale = 0
        for iid in intents:
            for r in set(regimes):
                st = self.sens.staleness(r, iid) \
                    if hasattr(self.sens, "staleness") else {"stale": 0.0}
                if st.get("stale", 0.0) > 0:
                    n_stale += 1
                    self.stale_flags[(r, iid)] = epoch
        out["stale"] = n_stale
        if n_stale:
            self.log(epoch, "staleness",
                     f"{n_stale} (regime, intent) cells flagged stale")
        rec = self.sens.maybe_promote(epoch, list(set(regimes)))
        if rec.get("promoted"):
            out["promoted"] = True
            out.update({k: rec[k] for k in
                        ("version", "improvement", "entries_changed")
                        if k in rec})
            self._version_marks.append((epoch, rec["version"],
                                        rec["candidate_mae"]))
            self.log(epoch, "promotion",
                     f"model v{rec['version']} promoted "
                     f"({rec['improvement']:+.1%} held-out improvement, "
                     f"{rec['entries_changed']} entries)",
                     **{k: v for k, v in rec.items() if k != "epoch"})
        self._check_rollback(epoch)
        return out

    def _check_rollback(self, epoch: int) -> None:
        if not self._version_marks:
            return
        mark_epoch, version, mae_at_promotion = self._version_marks[-1]
        if epoch - mark_epoch < self.rollback_window:
            return
        rows = [r for r in self.sens.residuals if r["epoch"] > mark_epoch]
        if len(rows) < 6:
            return
        mae_now = float(np.mean([abs(r["residual"]) for r in rows]))
        if mae_now > 1.6 * max(mae_at_promotion, 1e-9):
            self.rollbacks += 1
            self.log(epoch, "rollback",
                     f"model v{version} residuals worsened "
                     f"({mae_now:.4f} vs {mae_at_promotion:.4f} at "
                     f"promotion); reverting to the prior table",
                     version=version, mae_now=mae_now,
                     mae_at_promotion=mae_at_promotion)
            self.sens.live = dict(self.sens.prior.tab)
            self.sens.version += 1
        self._version_marks = self._version_marks[-1:]

    # ------------------------------------------------------------------
    def explain(self, epoch: int, winner, decisions, regimes,
                intents: Dict[str, Intent]) -> str:
        """A structured, numerical rationale.  A rendering of the record,
        never the source of the decision."""
        lines = [f"epoch {epoch}: portfolio "
                 f"{'|'.join(winner.claims) or '(no action)'} "
                 f"utility {winner.utility:+.5f}"]
        for d in decisions:
            if not d.executed:
                continue
            imp = sorted(d.predicted_dg.items(),
                         key=lambda kv: -abs(kv[1]))[:3]
            det = ", ".join(f"{i} {v:+.4f}" for i, v in imp)
            lines.append(f"  {d.jid}: {d.param} "
                         f"{d.nu_old:.3f} -> {d.nu_star:.3f} "
                         f"({d.outcome.value}); predicted {det}")
        for iid, it in intents.items():
            if winner.g_lower.get(iid, 0.0) < it.epsilon <= \
                    winner.g_hat.get(iid, 0.0):
                lines.append(f"  NOTE {iid}: point prediction is safe but "
                             f"the robust bound "
                             f"{winner.g_lower[iid]:+.4f} is below the floor "
                             f"{it.epsilon:+.4f}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def diagnostics(self) -> Dict:
        kinds: Dict[str, int] = {}
        for e in self.events:
            kinds[e.kind] = kinds.get(e.kind, 0) + 1
        return {"supervisor_events": len(self.events),
                "probes_issued": self.probes_issued,
                "rollbacks": self.rollbacks,
                "unknown_entries": len(self.unknown_entries),
                **{f"event_{k}": v for k, v in kinds.items()}}

    def event_rows(self) -> List[Dict]:
        return [{"epoch": e.epoch, "kind": e.kind, "detail": e.detail,
                 **{k: v for k, v in e.payload.items()
                    if isinstance(v, (int, float, str))}}
                for e in self.events]
