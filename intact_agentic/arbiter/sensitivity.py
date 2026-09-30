"""
intact_agentic/arbiter/sensitivity.py
=====================================
The cross-sensitivity model  s_{i,p}(z) = d g_i / d nu_p  evaluated in the
operating context z.

    "If I move this knob by one unit RIGHT NOW, how much does this
     intent's margin move?"

THIS IS THE LOAD-BEARING OBJECT OF THE WHOLE PAPER
--------------------------------------------------
The arbiter's safety check asks whether the PREDICTED post-action margin
stays above the floor.  It therefore runs on the sensitivity table, not on
the radio.  If a slope is optimistic the check passes and the contract
still breaks -- and nobody finds out, because nothing in the original
design ever compares the prediction to the outcome.  A stale table is not
a tuning inconvenience, it is a safety defect.

TWO IMPLEMENTATIONS, ONE INTERFACE
----------------------------------
``StaticSensitivity``   the frozen INTACT-RA behaviour: a table measured
                        once, offline, by :func:`offline_sweep`, indexed
                        by (regime, param, intent) and never updated.
                        A missing entry returns 0.0 with zero uncertainty,
                        which is the documented failure mode -- "unknown"
                        is silently asserted to be "no effect".

``OnlineSensitivity``   the agentic behaviour.  Same offline table as the
                        PRIOR, then:
                          * joint recursive least squares per
                            (regime, intent) over the executed dose
                            vector, so simultaneous writes are separated
                            when the design has enough variation and are
                            honestly reported as unidentified when it
                            does not;
                          * a ridge pull toward the prior, so one noisy
                            KPI window cannot move a safety-critical
                            slope;
                          * per-entry predictive uncertainty sigma from
                            the RLS covariance, used by the robust safety
                            bound  g_lower = g_hat - beta * sigma;
                          * MISSING IS NOT ZERO.  An unknown entry returns
                            slope 0 with sigma = ``unknown_sigma``, which
                            makes the robust bound refuse the action
                            rather than wave it through;
                          * a residual log, a multi-signal staleness
                            detector, and a promotion gate that validates
                            a candidate update on held-out epochs before
                            the arbiter is allowed to use it.

WHY DRL IS NOT THE SLOPE LEARNER
--------------------------------
A policy gradient estimates "which action is good", not "what is the
partial derivative of this KPI with respect to this knob".  Using it as
the slope learner would put an unvalidated, non-interpretable number
inside the safety constraint.  The DRL policy in ``agent/`` proposes a
short list of candidate portfolios; the numbers that decide safety come
from this file, are versioned, and can be printed.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..types import Claim, Intent
from .margins import margin

Key = Tuple[str, str, str]      # (regime, param, intent)


# ---------------------------------------------------------------------------
@dataclass
class Entry:
    slope: float = 0.0
    sigma: float = 0.0
    n: int = 0
    source: str = "missing"     # offline | online | probe | prior | missing
    version: int = 0
    updated_epoch: int = 0


class SensitivityBase:
    """Common interface.  The arbiter only ever calls these four methods."""

    unknown_sigma = 0.05

    def get(self, regime: str, param: str, iid: str) -> float:
        raise NotImplementedError

    def sigma(self, regime: str, param: str, iid: str) -> float:
        raise NotImplementedError

    def known(self, regime: str, param: str, iid: str) -> bool:
        raise NotImplementedError

    def affected(self, regime: str, param: str,
                 iids: Iterable[str]) -> List[str]:
        raise NotImplementedError

    # ---- optional hooks (no-ops for the static model) -----------------
    def observe(self, *a, **kw) -> None:
        return None

    def maybe_promote(self, *a, **kw) -> Dict:
        return {}

    def diagnostics(self) -> Dict:
        return {}


# ---------------------------------------------------------------------------
class StaticSensitivity(SensitivityBase):
    """The frozen offline table.  INTACT-RA's model of the world."""

    def __init__(self, cfg: Dict, table: Optional[Dict[Key, Entry]] = None):
        self.cfg = cfg
        a = cfg.get("arbiter", {}) or {}
        self.sigma_min = float(a.get("sigma_min", 5e-4))
        self.tab: Dict[Key, Entry] = dict(table or {})

    def get(self, regime: str, param: str, iid: str) -> float:
        e = self.tab.get((regime, param, iid))
        if e is None:
            return 0.0
        return e.slope if abs(e.slope) >= self.sigma_min else 0.0

    def sigma(self, regime: str, param: str, iid: str) -> float:
        e = self.tab.get((regime, param, iid))
        # The static model asserts certainty about entries it has, and --
        # this is the defect the paper measures -- also about the entries
        # it does NOT have.
        return float(e.sigma) if e is not None else 0.0

    def known(self, regime: str, param: str, iid: str) -> bool:
        return (regime, param, iid) in self.tab

    def affected(self, regime: str, param: str,
                 iids: Iterable[str]) -> List[str]:
        return [i for i in iids
                if abs(self.get(regime, param, i)) >= self.sigma_min]

    # ---- persistence --------------------------------------------------
    def to_json(self) -> Dict:
        return {f"{k[0]}|{k[1]}|{k[2]}":
                {"slope": e.slope, "sigma": e.sigma, "n": e.n,
                 "source": e.source} for k, e in self.tab.items()}

    @classmethod
    def from_json(cls, cfg: Dict, blob: Dict) -> "StaticSensitivity":
        tab: Dict[Key, Entry] = {}
        for k, v in blob.items():
            r, p, i = k.split("|")
            tab[(r, p, i)] = Entry(slope=float(v["slope"]),
                                   sigma=float(v.get("sigma", 0.0)),
                                   n=int(v.get("n", 0)),
                                   source=str(v.get("source", "offline")))
        return cls(cfg, tab)

    def save(self, path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, cfg: Dict, path) -> "StaticSensitivity":
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_json(cfg, blob)


# ---------------------------------------------------------------------------
class OnlineSensitivity(SensitivityBase):
    """Offline-initialised, residual-driven, versioned sensitivity model."""

    def __init__(self, cfg: Dict, prior: StaticSensitivity,
                 params: Sequence[str], intents: Sequence[str]):
        self.cfg = cfg
        o = (cfg.get("agent", {}) or {}).get("sensitivity", {}) or {}
        a = cfg.get("arbiter", {}) or {}
        self.sigma_min = float(a.get("sigma_min", 5e-4))
        self.unknown_sigma = float(o.get("unknown_sigma", 0.05))
        self.backoff_sigma_mult = float(o.get("backoff_sigma_mult", 1.6))
        self.backoff_sigma_add = float(o.get("backoff_sigma_add", 0.004))
        self.n_backoffs = 0
        self.lam = float(o.get("forgetting", 0.985))       # RLS forgetting
        self.ridge = float(o.get("prior_ridge", 12.0))     # pull to prior
        self.p0 = float(o.get("p0", 1.0))
        self.max_sigma = float(o.get("max_sigma", 0.25))
        self.promote_min_obs = int(o.get("promote_min_obs", 6))
        self.promote_improve = float(o.get("promote_min_improvement", 0.05))
        self.holdout = int(o.get("holdout_epochs", 6))
        self.stale_resid = float(o.get("stale_resid_mult", 2.5))
        self.stale_window = int(o.get("stale_window", 12))
        self.dose_scale = dict(o.get("dose_scale", {}) or {})

        self.prior = prior
        self.params = list(params)
        self.intents = list(intents)
        self._pidx = {p: k for k, p in enumerate(self.params)}

        # per (regime, intent):  theta (len P), P matrix (P x P), counts
        self._theta: Dict[Tuple[str, str], np.ndarray] = {}
        self._P: Dict[Tuple[str, str], np.ndarray] = {}
        self._n: Dict[Tuple[str, str], int] = {}
        self._excite: Dict[Tuple[str, str], np.ndarray] = {}
        # promoted (live) table the arbiter reads
        self.live: Dict[Key, Entry] = dict(prior.tab)
        self.version = 1
        self.residuals: List[Dict] = []
        self._holdout_rows: List[Dict] = []
        self._stale_flags: Dict[str, Dict] = {}
        self.promotions: List[Dict] = []
        self.n_updates = 0
        self.n_unknown_blocks = 0
        self.n_psd_repairs = 0

    # ------------------------------------------------------------------
    def _slot(self, regime: str, iid: str):
        key = (regime, iid)
        if key not in self._theta:
            th = np.zeros(len(self.params))
            par = self._parent(regime)
            for p in self.params:
                e = self.prior.tab.get((regime, p, iid))
                if e is None and par is not None:
                    e = self.prior.tab.get((par, p, iid))
                # divided by the dose scale: theta lives in scaled-dose
                # units (see candidate()), the prior table does not
                th[self._pidx[p]] = (e.slope / self._scale(p)) if e else 0.0
            self._theta[key] = th
            self._P[key] = np.eye(len(self.params)) * self.p0
            self._n[key] = 0
            self._excite[key] = np.zeros(len(self.params))
        return key

    def _scale(self, param: str) -> float:
        return float(self.dose_scale.get(param, 1.0))

    # ------------------------------------------------------------------
    @staticmethod
    def _parent(regime: str) -> Optional[str]:
        """The coarser regime a refined label backs off to.

        The channel axis is a REFINEMENT of the load axis: "L1C1" means
        "load band 1, and additionally coverage-limited".  A refined cell
        with no evidence yet is not the same epistemic state as a cell we
        have never had any information about -- we still know what the
        load-indexed calibration said.  Treating the two identically makes
        the agent refuse every action the moment the channel axis switches
        on, which is a failure of bookkeeping, not honest caution.
        """
        if len(regime) >= 4 and regime[-2] == "C" and regime[-1] != "0":
            return regime[:-1] + "0"
        return None

    def _lookup(self, regime: str, param: str, iid: str):
        """(entry, backed_off).  Falls back to the parent regime's prior."""
        e = self.live.get((regime, param, iid))
        if e is not None:
            return e, False
        par = self._parent(regime)
        if par is not None:
            e = self.live.get((par, param, iid))
            if e is not None:
                return e, True
            e = self.prior.tab.get((par, param, iid))
            if e is not None:
                return e, True
        return None, False

    def get(self, regime: str, param: str, iid: str) -> float:
        e, _ = self._lookup(regime, param, iid)
        if e is None:
            return 0.0
        return e.slope if abs(e.slope) >= self.sigma_min else 0.0

    def sigma(self, regime: str, param: str, iid: str) -> float:
        e, backed_off = self._lookup(regime, param, iid)
        if e is None:
            # MISSING IS NOT ZERO.  Report the full unknown uncertainty so
            # the robust bound g_hat - beta*sigma refuses the action and
            # the supervisor is asked for a calibration probe instead.
            self.n_unknown_blocks += 1
            return self.unknown_sigma
        s = float(min(e.sigma, self.max_sigma))
        if backed_off:
            # usable, but explicitly less trustworthy than a slope measured
            # in this regime: the refinement itself is untested here
            self.n_backoffs += 1
            s = float(min(s * self.backoff_sigma_mult + self.backoff_sigma_add,
                          self.max_sigma))
        return s

    def known(self, regime: str, param: str, iid: str) -> bool:
        e, _ = self._lookup(regime, param, iid)
        return e is not None

    def affected(self, regime: str, param: str,
                 iids: Iterable[str]) -> List[str]:
        out = []
        for i in iids:
            if abs(self.get(regime, param, i)) >= self.sigma_min:
                out.append(i)
            elif not self.known(regime, param, i):
                out.append(i)      # unknown entries are implicated, not ignored
        return out

    # ------------------------------------------------------------------
    def observe(self, epoch: int, regime_of_intent: Dict[str, str],
                doses: Dict[str, float], g_before: Dict[str, float],
                g_after: Dict[str, float], predicted: Dict[str, float],
                probe: bool = False) -> None:
        """One epoch of evidence: what we wrote and what actually happened.

        ``doses`` maps parameter -> executed delta (nu_star - nu_old).  It
        is deliberately the EXECUTED delta, not the requested one: a write
        that was overridden to a smaller value carries the information of
        the smaller value.
        """
        if not doses:
            # Still record the residual -- a zero-dose epoch grades the
            # model's claim that nothing should have changed, which is a
            # genuine test of the drift detector.
            pass
        x_full = np.zeros(len(self.params))
        for p, d in doses.items():
            if p in self._pidx:
                x_full[self._pidx[p]] = float(d) * self._scale(p)

        for iid, r in regime_of_intent.items():
            if iid not in g_after or iid not in g_before:
                continue
            y = float(g_after[iid] - g_before[iid])
            pred = float(predicted.get(iid, 0.0))
            resid = y - pred
            row = {"epoch": epoch, "intent": iid, "regime": r,
                   "observed_dg": y, "predicted_dg": pred,
                   "residual": resid, "probe": int(probe),
                   "n_knobs": int(np.count_nonzero(x_full)),
                   "dose_norm": float(np.linalg.norm(x_full))}
            self.residuals.append(row)
            self._holdout_rows.append({**row, "x": x_full.copy()})

            if np.allclose(x_full, 0.0):
                continue
            # An observation made in a REFINED regime is also an
            # observation in its parent: "load band 1 and coverage-limited"
            # is a subset of "load band 1".  Updating both keeps the coarse
            # cell usable as a back-off target instead of letting it go
            # stale the moment the channel axis switches on, and it roughly
            # doubles the evidence per cell in a regime space this sparse.
            keys = [self._slot(r, iid)]
            par = self._parent(r)
            if par is not None:
                keys.append(self._slot(par, iid))
            for key in keys:
                self._rls_update(key, x_full, y)
            self.n_updates += 1
            continue

            key = self._slot(r, iid)
            P = self._P[key]
            th = self._theta[key]
            x = x_full
            Px = P @ x
            denom = self.lam + float(x @ Px)
            k = Px / max(denom, 1e-12)
            th_new = th + k * (y - float(x @ th))
            # ridge pull toward the offline prior: a safety-critical slope
            # should need repeated, consistent evidence to move far.
            prior_vec = np.array([
                (self.prior.tab.get((r, p, iid)).slope
                 if self.prior.tab.get((r, p, iid)) else 0.0)
                for p in self.params])
            w = 1.0 / (1.0 + self.ridge / max(self._n[key] + 1, 1))
            th_new = w * th_new + (1.0 - w) * prior_vec
            self._theta[key] = th_new
            self._P[key] = (P - np.outer(k, Px)) / self.lam
            self._n[key] += 1
            self._excite[key] += np.abs(x)
            self.n_updates += 1

    # ------------------------------------------------------------------
    def _rls_update(self, key, x: np.ndarray, y: float) -> None:
        """One recursive-least-squares step with a ridge pull to the prior."""
        P = self._P[key]
        th = self._theta[key]
        Px = P @ x
        denom = self.lam + float(x @ Px)
        k = Px / max(denom, 1e-12)
        th_new = th + k * (y - float(x @ th))
        rg, iid = key
        prior_vec = np.array([
            (self.prior.tab.get((rg, p, iid)).slope / self._scale(p)
             if self.prior.tab.get((rg, p, iid)) else 0.0)
            for p in self.params])
        # a safety-critical slope should need repeated, consistent evidence
        # before it moves far from the offline calibration
        w = 1.0 / (1.0 + self.ridge / max(self._n[key] + 1, 1))
        th_new = w * th_new + (1.0 - w) * prior_vec
        self._theta[key] = th_new
        Pn = (P - np.outer(k, Px)) / self.lam
        # The Riccati update is symmetric in exact arithmetic but not in
        # floating point, and the asymmetry compounds over thousands of
        # updates with forgetting (lam < 1 inflates it every step).  Once P
        # loses positive semi-definiteness the predictive variance can go
        # negative and sigma becomes a NaN, which silently disables the
        # robust safety bound rather than raising.  Symmetrise every step
        # and project back onto the PSD cone if an eigenvalue has gone
        # negative; both are cheap at this dimension.
        Pn = 0.5 * (Pn + Pn.T)
        dmin = float(np.min(np.diag(Pn)))
        if dmin <= 0.0 or not np.all(np.isfinite(Pn)):
            ev, V = np.linalg.eigh(np.nan_to_num(Pn))
            Pn = (V * np.clip(ev, 1e-12, None)) @ V.T
            self.n_psd_repairs += 1
        self._P[key] = Pn
        self._n[key] += 1
        self._excite[key] += np.abs(x)

    # ------------------------------------------------------------------
    def candidate(self, regime: str, param: str, iid: str
                  ) -> Tuple[float, float, int]:
        key = (regime, iid)
        if key not in self._theta or param not in self._pidx:
            e = self.prior.tab.get((regime, param, iid))
            return ((e.slope, e.sigma, e.n) if e else (0.0, self.unknown_sigma,
                                                       0))
        j = self._pidx[param]
        th = self._theta[key][j]
        var = max(float(self._P[key][j, j]), 1e-12)
        exc = float(self._excite[key][j])
        infl = 1.0 + 3.0 / (1.0 + exc)
        sd = math.sqrt(var) * infl
        # UNITS.  The regression is fitted on SCALED doses, x_j = d_j * s_j,
        # purely to keep the design matrix well conditioned when knobs have
        # very different natural step sizes.  theta_j is therefore the slope
        # per scaled dose, while every consumer (the arbiter's predict(),
        # the residual log, the promotion gate) applies the slope to a RAW
        # dose.  Returning theta unconverted understates any knob with
        # s_j > 1 by exactly that factor -- for schedw_* with s = 10 the
        # published slope was one tenth of the truth, so the arbiter
        # systematically believed scheduler steering did almost nothing.
        s = self._scale(param)
        return float(th * s), float(min(sd * s, self.max_sigma)), \
            int(self._n[key])

    # ------------------------------------------------------------------
    def staleness(self, regime: str, iid: str) -> Dict[str, float]:
        """Multi-signal staleness test.

        A model is stale when its residuals stop looking like noise.  Age
        alone is never sufficient: a fresh table can be wrong and an old
        one can still be right, so ``age`` only contributes when residual
        evidence agrees.
        """
        # Only rows where a knob actually moved carry information about a
        # SLOPE.  Zero-dose rows measure how much the plant wanders on its
        # own, which is real but is not evidence that a coefficient is
        # wrong -- mixing the two buries the signal in plant noise.
        rows = [r for r in self.residuals[-600:]
                if r["intent"] == iid and r["regime"] == regime
                and r.get("n_knobs", 0) > 0]
        if len(rows) < 4:
            return {"stale": 0.0, "mean_abs_resid": 0.0, "cusum": 0.0,
                    "n": len(rows)}
        recent = rows[-self.stale_window:]
        older = rows[:-self.stale_window] or rows
        mar = float(np.mean([abs(r["residual"]) for r in recent]))
        base = float(np.mean([abs(r["residual"]) for r in older])) + 1e-9
        # one-sided CUSUM on the signed residual: a persistent bias means
        # the slope has moved, whereas symmetric noise cancels out
        s = 0.0
        peak = 0.0
        sd = float(np.std([r["residual"] for r in older])) + 1e-9
        for r in recent:
            s = max(0.0, s + r["residual"] / sd - 0.5)
            peak = max(peak, s)
        stale = 1.0 if (mar > self.stale_resid * base or peak > 4.0) else 0.0
        return {"stale": stale, "mean_abs_resid": mar, "baseline": base,
                "cusum": peak, "n": len(rows)}

    # ------------------------------------------------------------------
    def maybe_promote(self, epoch: int, regimes: Sequence[str],
                      force: bool = False) -> Dict:
        """Validate candidate slopes on held-out rows, then publish.

        The candidate model replaces the live table only if it predicts the
        held-out window better than the live table does, by at least
        ``promote_min_improvement`` in mean absolute residual.  This is the
        gate that stops one unusual traffic burst from rewriting a
        safety-critical coefficient.
        """
        # The held-out window must be counted in INFORMATIVE rows -- rows
        # where something was actually written.  A zero-dose row grades
        # nothing about a slope: both the live and the candidate table
        # predict zero change and both are right, so including them makes
        # the two errors identical (and, when the window is all zero-dose,
        # identically zero) and the improvement test can never fire.
        informative = [r for r in self._holdout_rows if np.any(r["x"])]
        rows = informative[-max(self.holdout * 6, 24):]
        if len(rows) < self.promote_min_obs and not force:
            return {"promoted": False, "reason": "insufficient evidence",
                    "informative_rows": len(informative)}

        def err(table_get) -> float:
            tot, k = 0.0, 0
            for r in rows:
                x = r["x"]
                if not np.any(x):
                    continue
                pred = 0.0
                for p, j in self._pidx.items():
                    if abs(x[j]) > 0:
                        pred += table_get(r["regime"], p, r["intent"]) * x[j]
                tot += abs(r["observed_dg"] - pred)
                k += 1
            return tot / max(k, 1)

        live_err = err(lambda rg, p, i: (self.live.get((rg, p, i)).slope
                                         if self.live.get((rg, p, i)) else 0.0))
        cand_err = err(lambda rg, p, i: self.candidate(rg, p, i)[0])
        improved = (live_err - cand_err) / max(live_err, 1e-9)
        if not force and improved < self.promote_improve:
            return {"promoted": False, "reason": "no improvement",
                    "live_mae": live_err, "candidate_mae": cand_err,
                    "improvement": improved}

        # The published uncertainty is the HELD-OUT residual spread, not the
        # RLS covariance diagonal.  The covariance is a statement about how
        # well-conditioned the regression is; it starts at the prior scale
        # p0 and shrinks only where a knob has actually been excited.
        # Publishing it as sigma creates a deadlock: a freshly promoted
        # entry carries a huge sigma, the robust bound then refuses every
        # write, no writes means no excitation, and the entry never
        # improves.  The held-out residual spread is the quantity the
        # promotion gate just validated against, so it is both the honest
        # and the self-consistent choice.
        resid_sd = float(np.std([r["observed_dg"] - sum(
            self.candidate(r["regime"], p, r["intent"])[0] * r["x"][j]
            for p, j in self._pidx.items() if abs(r["x"][j]) > 0)
            for r in rows if np.any(r["x"])])) if rows else self.max_sigma

        n_changed = 0
        for (rg, iid), th in self._theta.items():
            if rg not in regimes:
                continue
            for p, j in self._pidx.items():
                slope, sd, n = self.candidate(rg, p, iid)
                prior_e = self.prior.tab.get((rg, p, iid))
                floor = prior_e.sigma if prior_e else self.unknown_sigma
                # sigma is the standard error of the SLOPE, not the spread
                # of the residuals.  The residual spread contains the
                # plant's own irreducible noise, which no amount of
                # evidence removes and which is not uncertainty about the
                # coefficient; publishing it directly makes a freshly
                # learned entry look an order of magnitude less certain
                # than the offline calibration it just beat, and the
                # arbiter then refuses to use what the agent has learned.
                # The textbook relation se = s / sqrt(n) is the right one.
                se = resid_sd / max(np.sqrt(max(n, 1)), 1.0)
                sd = float(np.clip(min(sd, se), self.sigma_min,
                                   max(floor * 3.0, self.max_sigma)))
                old = self.live.get((rg, p, iid))
                if old is None or abs(old.slope - slope) > 1e-9:
                    n_changed += 1
                self.live[(rg, p, iid)] = Entry(
                    slope=slope, sigma=sd, n=n, source="online",
                    version=self.version + 1, updated_epoch=epoch)
        self.version += 1
        rec = {"promoted": True, "epoch": epoch, "version": self.version,
               "live_mae": live_err, "candidate_mae": cand_err,
               "improvement": improved, "entries_changed": n_changed}
        self.promotions.append(rec)
        self._holdout_rows = self._holdout_rows[-self.holdout:]
        return rec

    # ------------------------------------------------------------------
    def register_new(self, regime: str, param: str, iid: str) -> None:
        """Declare an entry that does not exist yet (new tenant / intent).

        It is NOT created with slope zero.  It is left absent, so
        :meth:`sigma` reports ``unknown_sigma`` and the arbiter is forced
        to abstain, restrict, or ask for calibration.
        """
        if param not in self._pidx:
            self._pidx[param] = len(self.params)
            self.params.append(param)
            for k in list(self._theta):
                self._theta[k] = np.append(self._theta[k], 0.0)
                P = self._P[k]
                n = P.shape[0] + 1
                newP = np.eye(n) * self.p0
                newP[:n - 1, :n - 1] = P
                self._P[k] = newP
                self._excite[k] = np.append(self._excite[k], 0.0)
        if iid not in self.intents:
            self.intents.append(iid)

    # ------------------------------------------------------------------
    def diagnostics(self) -> Dict:
        res = [r["residual"] for r in self.residuals] or [0.0]
        return {
            "version": self.version,
            "n_updates": self.n_updates,
            "n_promotions": len(self.promotions),
            "n_unknown_blocks": self.n_unknown_blocks,
                "n_backoffs": self.n_backoffs,
                "n_psd_repairs": self.n_psd_repairs,
            "mae_residual": float(np.mean(np.abs(res))),
            "live_entries": len(self.live),
        }


# ---------------------------------------------------------------------------
def offline_sweep(ran, cfg: Dict, intents: Dict[str, Intent],
                  claims: Dict[str, Claim], regimes: Sequence[str],
                  log=None, per_tenant: bool = False) -> StaticSensitivity:
    """Phase 1: the offline parameter sweep that builds the prior table.

    This reproduces the INTACT-RA calibration procedure exactly:

      * force the cell into each declared load regime;
      * freeze every other knob;
      * step the parameter across its TRUST REGION around the operating
        point (a local derivative should be measured locally);
      * reset to a common random seed for every value, so knob value is
        not confounded with queue history or slot index;
      * let it settle, then measure;
      * repeat over replicates and fit OLS with a standard error.

    The sweep runs with mobility DISABLED (``mode: static``) because that
    is what an offline calibration in a lab or digital twin can actually
    do.  That is precisely why the resulting table cannot anticipate a
    later edge-user drift -- the limitation is structural, not an
    implementation shortcut.
    """
    sc = cfg["calibration"]
    tab: Dict[Key, Entry] = {}
    params = sorted({c.param for c in claims.values()})
    base_controls = dict(ran.current_controls())
    domains = {c.param: c.domain for c in claims.values()}
    steps = {c.param: c.max_step_frac for c in claims.values()}

    level_of = {r: lv for r, lv in zip(regimes,
                                       sc.get("sweep_load_scales", [1.0]))}
    # PER-TENANT variant (an ablation; the original scales every tenant
    # together).  INTACT-RA reads intent i's slope at i's OWN tenant's load
    # band, so the cell for band b is measured here while ONLY that tenant
    # is at band b and every other tenant is at nominal load -- a state the
    # uniform sweep never visits.  Only that tenant's intents are recorded.
    focus_list = (sorted(ran.traffic.base_mbps) if per_tenant else [None])
    for regime, focus in [(r, f) for r in regimes for f in focus_list]:
        scale = float(level_of.get(regime, 1.0))
        for prm in params:
            lo, hi = domains[prm]
            cur = float(base_controls.get(prm, 0.5 * (lo + hi)))
            half = float(steps.get(prm, 0.25)) * (hi - lo)
            glo, ghi = max(lo, cur - half), min(hi, cur + half)
            if ghi - glo < 1e-9:
                glo, ghi = lo, hi
            grid = np.linspace(glo, ghi, int(sc.get("sweep_points", 5)))
            xs: List[float] = []
            ys: Dict[str, List[float]] = {i: [] for i in intents}
            for rep in range(int(sc.get("replicates", 3))):
                for val in grid:
                    probe = ran.clone()
                    probe.reset(int(sc.get("base_seed", 101)) + 97 * rep)
                    # force the regime by scaling every UE's baseline rate
                    for tid in probe.traffic.base_mbps:
                        if focus is None or tid == focus:
                            probe.traffic.base_mbps[tid] = (
                                probe.traffic.base_mbps[tid] * scale)
                        probe.traffic.profiles[tid].levels = (1.0,)
                    probe.mobility.mode = "static"
                    for k, v in base_controls.items():
                        probe.apply(k, v)
                    probe._reconfig_left = {t: 0 for t in probe.slices}
                    probe._cell_reconfig_left = 0
                    probe.step(int(sc.get("settle_slots", 24)), record=False)
                    probe.apply(prm, float(val))
                    probe._reconfig_left = {t: 0 for t in probe.slices}
                    probe._cell_reconfig_left = 0
                    probe.step(int(sc.get("settle_slots", 24)), record=False)
                    kpm = probe.step(int(sc.get("measure_slots", 24)),
                                     record=False)
                    xs.append(float(val))
                    for iid, it in intents.items():
                        g = margin(it, kpm)
                        ys[iid].append(np.nan if g is None else g)
            X = np.asarray(xs, dtype=float)
            if X.size < 3 or np.std(X) < 1e-12:
                continue
            A = np.column_stack([np.ones_like(X), X])
            for iid in intents:
                if focus is not None and intents[iid].tenant != focus:
                    continue
                Y = np.asarray(ys[iid], dtype=float)
                ok = np.isfinite(Y)
                if ok.sum() < 3 or np.std(X[ok]) < 1e-12:
                    continue
                coef, *_ = np.linalg.lstsq(A[ok], Y[ok], rcond=None)
                resid = Y[ok] - A[ok] @ coef
                dof = max(int(ok.sum()) - 2, 1)
                s2 = float(resid @ resid / dof)
                se = float(np.sqrt(max(s2 * np.linalg.pinv(
                    A[ok].T @ A[ok])[1, 1], 0.0)))
                tab[(regime, prm, iid)] = Entry(
                    slope=float(coef[1]), sigma=se, n=int(ok.sum()),
                    source="offline")
                if log:
                    log(f"sweep {regime:>6} | {prm:<14} -> {iid:<4} "
                        f"s={coef[1]:+.6f} se={se:.6f}")
    out = StaticSensitivity(cfg, tab)
    if log:
        log(f"offline sweep complete: {len(tab)} (regime,param,intent) entries")
    return out


class ScrambledSensitivity(SensitivityBase):
    """A deliberately WRONG table with the right summary statistics.

    Diagnostic control, never a method.  It keeps the offline table's
    distribution of slope magnitudes and standard errors but permutes
    which (parameter, intent) pair each value belongs to, deterministically
    per seed.  Any controller reading it therefore has perfectly calibrated
    *aggregate* beliefs and no *specific* knowledge whatsoever.

    Its purpose is to answer the question no amount of estimator tuning can
    answer from inside: does knowing the actual slopes matter in this
    plant?  If a controller scores the same on scrambled slopes as on true
    ones, then the benchmark is not measuring sensitivity knowledge, and
    any reported win for a better estimator is an artefact of something
    else -- usually of how often it happens to write.
    """

    def __init__(self, cfg: Dict, prior: "StaticSensitivity", seed: int = 0):
        self.cfg = cfg
        self.sigma_min = float((cfg.get("arbiter", {}) or {})
                               .get("sigma_min", 5e-4))
        rng = np.random.default_rng(int(seed) + 991)
        keys = sorted(prior.tab.keys())
        vals = [prior.tab[k] for k in keys]
        perm = rng.permutation(len(vals))
        self.tab = {k: vals[perm[i]] for i, k in enumerate(keys)}

    def get(self, regime: str, param: str, iid: str) -> float:
        e = self.tab.get((regime, param, iid))
        return 0.0 if e is None else (e.slope if abs(e.slope)
                                      >= self.sigma_min else 0.0)

    def sigma(self, regime: str, param: str, iid: str) -> float:
        e = self.tab.get((regime, param, iid))
        return 0.0 if e is None else float(e.sigma)

    def known(self, regime: str, param: str, iid: str) -> bool:
        return (regime, param, iid) in self.tab

    def affected(self, regime: str, param: str, iids):
        return [i for i in iids if abs(self.get(regime, param, i))
                >= self.sigma_min]

    def diagnostics(self) -> Dict:
        return {"scrambled_entries": len(self.tab)}


# ===========================================================================
class KalmanSensitivity(OnlineSensitivity):
    """Random-walk Kalman tracker of cross-sensitivities.  [INTACT-RA-Agentic]

    Replaces only the ESTIMATION CORE of OnlineSensitivity; every interface
    the arbiter and supervisor rely on is inherited, so no other component
    changes.

    Why it exists.  The joint-RLS table published an uncertainty built from
    the residual spread of margin changes, which is dominated by plant noise
    (served ratio and delay measured over 8-slot windows of bursty traffic
    swing by ~1 per epoch), and divided it only by sqrt(n), not by the dose
    that produced it.  A per-PRB slope then looked +/-0.25 uncertain, the
    robust floor g_hat - beta*sigma rejected 2,326 of the portfolios it
    considered, and the agent never wrote.  That is a bookkeeping error, not
    honest caution.

    What this does instead, per (regime, intent), jointly over the knobs:
      state      theta (slopes), C (their POSTERIOR covariance, slope units)
      prior      theta0 = offline slope; C0 = diag(sigma0^2) with
                   sigma0 = max(offline s.e., kappa*|theta0|, sigma_floor)
                 -- the kappa term is STRUCTURAL uncertainty: an offline
                 slope is only valid near the operating point it was
                 measured at, and the plant has since moved.
      noise R_i  the variance of the margin change in epochs with NO write,
                 i.e. measured plant noise, not a guess.
      update     S = x'Cx + R ; K = Cx/S ; theta += K(y - x'theta) ;
                 C -= K x'C             (only ever NARROWS uncertainty)
      drift      C += Q each epoch, Q = diag((q*sigma0)^2): slopes may move.
      change     if the normalised innovation e^2/S stays large, C is
                 widened back toward 4*C0 so a real shift is learned fast.
    Published slope = theta_j, sigma = sqrt(C_jj).
    """

    def __init__(self, cfg, prior, params, intents):
        super().__init__(cfg, prior, params=params, intents=intents)
        k = (cfg.get("agent", {}) or {}).get("kalman", {}) or {}
        self.kappa = float(k.get("structural_kappa", 0.5))
        self.sig_floor = float(k.get("sigma_floor", 0.002))
        self.q = float(k.get("process_q", 0.05))
        self.chg_thresh = float(k.get("change_threshold", 4.0))
        self.chg_alpha = float(k.get("change_ewma", 0.2))
        self.noise_alpha = float(k.get("noise_ewma", 0.05))
        self.noise_floor = float(k.get("noise_floor", 1e-4))
        self._kth: Dict = {}
        self._kC: Dict = {}
        self._kC0: Dict = {}
        self._nu: Dict = {}
        self._noise: Dict[str, float] = {}
        self.n_changes = 0
        # physics-informed prior for own-tenant served-ratio slopes
        self.analytic_prior = bool(k.get("analytic_prior", True))
        self._knob_tenant: Dict[str, str] = {}
        self._intent_meta: Dict[str, tuple] = {}
        self._ctx = None
        self.n_analytic = 0

    def bind(self, claims, intents) -> None:
        """Which tenant each allocative PRB knob belongs to, and each
        intent's tenant, KPI and target -- structural facts the operator
        configured, needed to form the analytic prior."""
        for c in claims.values():
            if getattr(c, "resource", None) == "prb":
                self._knob_tenant[c.param] = c.tenant
        for iid, it in intents.items():
            self._intent_meta[iid] = (it.tenant, it.kpi, float(it.target))

    def set_context(self, ran, kpm) -> None:
        """Latest telemetry, used only when a new regime cell is created."""
        self._ctx = (ran, kpm)

    def _analytic_slope(self, regime: str, iid: str, param: str):
        """d g / d reserved-PRB for a served-ratio intent of the knob's own
        tenant: (Mb/s per PRB) / (offered Mb/s * target) while the
        reservation binds, ~0 while it does not.  The pressure band in the
        regime label sets how binding the tenant is (L0 no, L1 partly, L2
        yes).  Returns None where the formula does not apply."""
        if not (self.analytic_prior and self._ctx):
            return None
        meta = self._intent_meta.get(iid)
        if not meta or meta[1] != "served_ratio":
            return None
        tid = meta[0]
        if self._knob_tenant.get(param) != tid:
            return None
        ran, kpm = self._ctx
        row = kpm.get(tid)
        if not row:
            return None
        try:
            band = int(regime[1])
        except (ValueError, IndexError):
            return None
        w = {0: 0.0, 1: 0.5}.get(band, 1.0)
        se = max(float(row.get("spectral_efficiency", 0.1)), 1e-3)
        bler = min(max(float(row.get("bler", 0.0)), 0.0), 0.9)
        mbps_per_prb = se * (1.0 - bler) * float(ran.prb_hz) / 1e6
        offered = max(float(row.get("offered_slice_mbps", 0.0)), 1e-3)
        return w * mbps_per_prb / (offered * max(meta[2], 1e-6))

    # ---- prior -------------------------------------------------------------
    def _prior_entry(self, regime, p, iid):
        e = self.prior.tab.get((regime, p, iid))
        if e is None:
            par = self._parent(regime)
            if par is not None:
                e = self.prior.tab.get((par, p, iid))
        if e is None:
            vals = [v for (r, pp, ii), v in self.prior.tab.items()
                    if pp == p and ii == iid]
            if vals:
                s = float(np.median([v.slope for v in vals]))
                se = float(np.median([v.sigma for v in vals]))
                return s, se
            return 0.0, self.unknown_sigma
        return float(e.slope), float(e.sigma)

    def _kslot(self, regime, iid):
        key = (regime, iid)
        if key not in self._kth:
            th = np.zeros(len(self.params))
            s0 = np.zeros(len(self.params))
            for p, j in self._pidx.items():
                s, se = self._prior_entry(regime, p, iid)
                a = self._analytic_slope(regime, iid, p)
                if a is not None:
                    # the offline sweep scaled every tenant together, so it
                    # never observed ONE tenant binding while the others were
                    # light; there its slope is ~9x too small.  The analytic
                    # slope replaces it, with +/-kappa structural uncertainty.
                    s, se = a, 0.0
                    self.n_analytic += 1
                th[j] = s
                s0[j] = max(se, self.kappa * abs(s), self.sig_floor)
            self._kth[key] = th
            self._kC0[key] = np.diag(s0 ** 2)
            self._kC[key] = np.diag(s0 ** 2)
            self._nu[key] = 1.0
        return key

    # ---- queries ------------------------------------------------------------
    def get(self, regime, param, iid):
        if param not in self._pidx:
            return 0.0
        th = self._kth[self._kslot(regime, iid)][self._pidx[param]]
        return float(th) if abs(th) >= self.sigma_min else 0.0

    def sigma(self, regime, param, iid):
        if param not in self._pidx:
            self.n_unknown_blocks += 1
            return self.unknown_sigma
        key = self._kslot(regime, iid)
        j = self._pidx[param]
        return float(min(np.sqrt(max(self._kC[key][j, j], 0.0)),
                         self.max_sigma))

    def known(self, regime, param, iid):
        return param in self._pidx

    def candidate(self, regime, param, iid):
        return (self.get(regime, param, iid), self.sigma(regime, param, iid),
                int(self._n.get((regime, iid), 0)))

    def maybe_promote(self, epoch, regimes, force=False):
        # a Kalman posterior is validated by construction and always live
        return {"promoted": False, "reason": "kalman: estimates always live"}

    # ---- learning ------------------------------------------------------------
    def observe(self, epoch, regime_of_intent, doses, g_before, g_after,
                predicted, probe=False):
        x = np.zeros(len(self.params))
        for p, d in doses.items():
            if p in self._pidx:
                x[self._pidx[p]] = float(d)
        wrote = bool(np.any(np.abs(x) > 1e-12))
        for iid, r in regime_of_intent.items():
            if iid not in g_after or iid not in g_before:
                continue
            y = float(g_after[iid] - g_before[iid])
            if not wrote:
                # a margin change with no write is PLANT NOISE: exactly the
                # measurement-noise term the filter needs
                v = self._noise.get(iid, y * y)
                self._noise[iid] = (1 - self.noise_alpha) * v \
                    + self.noise_alpha * y * y
            key = self._kslot(r, iid)
            C = self._kC[key]
            C += np.diag((self.q * np.sqrt(np.diag(self._kC0[key]))) ** 2)
            if not wrote:
                self._kC[key] = C
                continue
            th = self._kth[key]
            R = max(self._noise.get(iid, 0.05), self.noise_floor)
            Cx = C @ x
            S = float(x @ Cx) + R
            e = y - float(x @ th)
            K = Cx / S
            self._kth[key] = th + K * e
            Cn = C - np.outer(K, Cx)
            Cn = 0.5 * (Cn + Cn.T)
            ev, V = np.linalg.eigh(Cn)
            Cn = (V * np.clip(ev, 1e-14, None)) @ V.T
            nu = (1 - self.chg_alpha) * self._nu[key] \
                + self.chg_alpha * (e * e / S)
            self._nu[key] = nu
            if nu > self.chg_thresh:
                # persistent surprise: the plant has moved; widen so the new
                # slope is learned quickly instead of averaged away
                Cn = Cn + 4.0 * self._kC0[key]
                self._nu[key] = 1.0
                self.n_changes += 1
            self._kC[key] = Cn
            self._n[key] = self._n.get(key, 0) + 1
            self.n_updates += 1
            self.residuals.append({
                "epoch": int(epoch), "regime": r, "intent": iid,
                "observed_dg": y, "predicted_dg": float(predicted.get(iid, 0)),
                "residual": e, "n_knobs": int(np.sum(np.abs(x) > 1e-12)),
                "probe": int(bool(probe)), "x": x.tolist()})
        if len(self.residuals) > 4000:
            del self.residuals[:1000]

    def twin_observe(self, obs, inflate: float = 1.0) -> int:
        """Fold digital-twin slope measurements in as direct, noisy
        observations of individual coefficients (H = e_j), with variance
        from the replicate spread times ``inflate`` (twin-model distrust)."""
        n = 0
        for regime, iid, param, z, var in obs:
            if param not in self._pidx:
                continue
            key = self._kslot(regime, iid)
            j = self._pidx[param]
            C = self._kC[key]
            S = float(C[j, j]) + max(var * inflate, 1e-10)
            K = C[:, j] / S
            self._kth[key] = self._kth[key] + K * (z - self._kth[key][j])
            Cn = C - np.outer(K, C[j, :])
            Cn = 0.5 * (Cn + Cn.T)
            ev, V = np.linalg.eigh(Cn)
            self._kC[key] = (V * np.clip(ev, 1e-14, None)) @ V.T
            n += 1
        self.n_twin_obs = getattr(self, "n_twin_obs", 0) + n
        return n

    def diagnostics(self):
        d = super().diagnostics()
        d.update({"kalman_twin_obs": getattr(self, "n_twin_obs", 0)})
        d.update({"kalman_cells": len(self._kth),
                  "kalman_analytic_priors": self.n_analytic,
                  "kalman_changes": self.n_changes,
                  "kalman_updates": self.n_updates})
        return d

    def state_dict(self):
        return {"th": {repr(k): v.tolist() for k, v in self._kth.items()},
                "C": {repr(k): v.tolist() for k, v in self._kC.items()}}
