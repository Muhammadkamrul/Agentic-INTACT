"""
intact_agentic/arbiter/core.py
==============================
The deterministic arbiter.  THE FINAL AUTHORITY, in every method.

No learned component in this package can bypass what happens here.  The
DRL policy may propose a short list of candidate portfolios and the online
model may supply better slopes and honest uncertainties, but the accept /
reduce / reject / abstain decision is taken by this file, from numbers
that are versioned and printable.  That separation is the reason an
agentic extension of a safety-critical arbiter is defensible at all.

THE EPOCH PROCEDURE
-------------------
 1. NORMALISE.  Every claim's request is mapped to (parameter, current
    value, requested value, feasible dose).  The feasible dose applies the
    legal grid, the one-epoch trust region, and the tenant envelope.
 2. CONTEXT.  Each intent is assigned its regime label (per tenant for
    INTACT-RA, learned for the agentic variant).
 3. CANDIDATES.  Either the full conflict-free enumeration, or the DRL
    top-K list, ALWAYS augmented with the deterministic greedy portfolio,
    the no-action portfolio, and any calibration probe the supervisor
    asked for.  The augmentation is what makes the policy a search
    accelerator rather than a second decision maker: if the policy
    proposes nothing useful, the deterministic fallbacks are still there.
 4. PREDICT.  dg_i(S) = sum_{j in S} s_{i,p(j)}(r_i) * dnu_j, and the
    robust lower bound  g_lower = g + dg - beta * sigma(S).
 5. FILTER.  Drop portfolios that (a) write one parameter twice (C1),
    (b) break a tenant envelope (C2), (c) would push a currently-safe
    protected intent below its floor under the robust bound, or (d) have
    an unknown high-impact effect.
 6. SELECT.  Highest utility among the survivors; ties broken
    deterministically by (fewest writes, then claim ids).
 7. EXECUTE.  Writes are issued most-valuable-first with a running ledger,
    so the second write sees the first write's predicted effect.

WHY A RUNNING LEDGER MATTERS
----------------------------
Handing every decision the same pre-epoch margin vector makes sequential
mediation behave as parallel mediation: two individually safe writes can
jointly breach a floor and nothing notices.  The ledger closes that.
"""
from __future__ import annotations

import dataclasses
import itertools
import math
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from ..types import (Claim, Decision, Intent, Kind, Outcome, Portfolio,
                     Scope, Tenant, portfolio_key)
from .regime import RegimeReport
from .sensitivity import SensitivityBase


# ---------------------------------------------------------------------------
@dataclass
class ClaimRequest:
    jid: str
    param: str
    nu_old: float
    nu_req: float
    nu_feasible: float
    clipped_by_trust: bool = False
    clipped_by_envelope: bool = False

    @property
    def dose(self) -> float:
        return float(self.nu_feasible - self.nu_old)


@dataclass
class PortfolioScore:
    claims: Portfolio
    utility: float
    dg: Dict[str, float]
    sigma: Dict[str, float]
    g_hat: Dict[str, float]
    g_lower: Dict[str, float]
    admissible: bool
    reject_reason: str = ""
    n_writes: int = 0
    unknown_intents: List[str] = field(default_factory=list)
    # requests as ADMITTED for this portfolio: raises may be resized so the
    # portfolio's own releases fund them (pool-conserving transfers)
    reqs: Dict[str, "ClaimRequest"] = field(default_factory=dict)

    def as_row(self) -> Dict:
        return {"portfolio": "|".join(self.claims) or "(none)",
                "utility": self.utility, "admissible": int(self.admissible),
                "reason": self.reject_reason, "n_writes": self.n_writes}


# ---------------------------------------------------------------------------
# ONE tolerance for the C2 test, shared by the arbiter's admission check and
# the write-time violation count.  They previously disagreed (1e-6 when
# admitting, 1e-9 when counting), so a portfolio the arbiter correctly
# admitted at exactly the envelope was then counted as a violation.
C2_TOL = 1e-6


def _norm_cdf(z: float) -> float:
    """Standard normal CDF via erf; avoids a scipy call in the hot path."""
    return 0.5 * (1.0 + math.erf(float(z) / math.sqrt(2.0)))


class Arbiter:
    """Deterministic portfolio arbitration with a robust safety bound."""

    def __init__(self, cfg: Dict, tenants: Dict[str, Tenant]):
        a = cfg.get("arbiter", {}) or {}
        self.cfg = cfg
        self.tenants = tenants
        self.beta = float(a.get("uncertainty_beta", 1.5))
        self.lambda_risk = float(a.get("lambda_risk", 0.5))
        self.lambda_churn = float(a.get("lambda_churn", 0.0))
        self.action_cost = float(a.get("action_cost", 0.0015))
        self.protect_only_safe = bool(a.get("protect_only_safe", True))
        self.max_enumerate = int(a.get("max_enumerate_claims", 14))
        self.abstain_on_unknown = bool(a.get("abstain_on_unknown", False))
        self.unknown_impact = float(a.get("unknown_impact_threshold", 0.02))
        self.greedy_fallback = bool(a.get("greedy_fallback", True))
        self.use_weights = bool(a.get("use_intent_weights", True))
        self.objective = str(a.get("objective", "linear"))
        self.c2_cell_pool = bool(a.get("c2_cell_pool", False))
        self.pool_transfers = bool(a.get("pool_transfers", False))
        self.prob_sigma_floor = float(a.get("prob_sigma_floor", 0.03))
        self.tie_break = float(a.get("tie_break", 0.05))
        # per-intent rolling |residual| of predicted vs observed margin
        # change, maintained by note_residuals() from the experiment loop
        self.resid_sd: Dict[str, float] = {}
        self._resid_hist: Dict[str, List[float]] = {}
        self.resid_window = int(a.get("resid_window", 60))
        # diagnostics
        self.last_candidates = 0
        self.last_latency_ms = 0.0
        self.last_scores: List[PortfolioScore] = []
        self.enumeration_truncated = 0

    # ------------------------------------------------------------------
    def note_residuals(self, predicted: Dict[str, float],
                       g_before: Dict[str, float],
                       g_after: Dict[str, float]) -> None:
        """Feed back realised prediction error, per intent.

        Only epochs in which something was written are informative about a
        SLOPE; a zero-dose epoch measures how much the plant wanders on its
        own, which is real but is not prediction error.
        """
        for iid, pred in predicted.items():
            if iid not in g_after or iid not in g_before:
                continue
            obs = g_after[iid] - g_before[iid]
            h = self._resid_hist.setdefault(iid, [])
            h.append(abs(obs - pred))
            if len(h) > self.resid_window:
                del h[0]
            if len(h) >= 8:
                self.resid_sd[iid] = float(np.mean(h))

    # ------------------------------------------------------------------
    # step 1: normalise requests
    # ------------------------------------------------------------------
    def normalise(self, proposals: Dict[str, float], claims: Dict[str, Claim],
                  controls: Dict[str, float], ran) -> Dict[str, ClaimRequest]:
        out: Dict[str, ClaimRequest] = {}
        for jid, nu_req in proposals.items():
            c = claims[jid]
            nu_old = float(controls.get(c.param, c.project(nu_req)))
            projected = c.project(nu_req)
            trusted = c.trust_clip(projected, nu_old)
            clipped_trust = abs(trusted - projected) > 1e-9
            clipped_env = False
            if c.kind == Kind.ALLOCATIVE and c.resource:
                env, used = ran.headroom(c.tenant, c.resource)
                other = used - float(controls.get(c.param, 0.0))
                room = env - other
                if trusted > room + 1e-9:
                    trusted = c.project(max(min(trusted, room), c.domain[0]))
                    clipped_env = True
            out[jid] = ClaimRequest(jid=jid, param=c.param, nu_old=nu_old,
                                    nu_req=float(nu_req),
                                    nu_feasible=float(trusted),
                                    clipped_by_trust=clipped_trust,
                                    clipped_by_envelope=clipped_env)
        return out

    # ------------------------------------------------------------------
    # step 3: candidate generation
    # ------------------------------------------------------------------
    @staticmethod
    def conflict_free(subset: Sequence[str], claims: Dict[str, Claim]
                      ) -> bool:
        """C1: at most one writer per parameter in one epoch."""
        seen: Set[str] = set()
        for j in subset:
            p = claims[j].param
            if p in seen:
                return False
            seen.add(p)
        return True

    def fit_requests(self, subset: Sequence[str], claims: Dict[str, Claim],
                     reqs: Dict[str, ClaimRequest],
                     controls: Dict[str, float], ran
                     ) -> Tuple[bool, Dict[str, ClaimRequest]]:
        """C2 with POOL-CONSERVING TRANSFERS.  Returns (feasible, requests).

        Without transfers a portfolio that overflows a tenant envelope or
        the cell pool is simply infeasible.  xApps size their requests from
        their own controllers, so a raise for an overloaded tenant and a
        release from a light one rarely match; with the pool full, the pair
        overflows and the arbiter could never move capacity between tenants.
        Under hard slicing that left every mediated method -- the oracle
        included -- admitting releases with no matching raise, stranding
        PRBs, and far below what re-partitioning achieves on the same plant.

        With ``pool_transfers`` the raises in a portfolio are SCALED DOWN to
        fit: first to each tenant's envelope, then to the pool, funded by the
        portfolio's own releases plus any idle room.  Resized raises are
        floored to the knob step so rounding can never overflow again.  A
        portfolio with no raise to shrink is still infeasible.
        """
        out = {j: reqs[j] for j in subset if j in reqs}
        alloc = [j for j in subset if claims[j].kind == Kind.ALLOCATIVE
                 and claims[j].resource]
        if not alloc:
            return True, out

        def shrink(ids, excess):
            ups = [j for j in ids if out[j].dose > 1e-9]
            tot = sum(out[j].dose for j in ups)
            if not self.pool_transfers or tot <= excess + 1e-12:
                return False
            f = (tot - excess) / tot
            for j in ups:
                rq, c = out[j], claims[j]
                step = float(getattr(c, "step", 0.0) or 0.0)
                d = rq.dose * f
                if step > 0:
                    d = math.floor(d / step + 1e-9) * step
                out[j] = dataclasses.replace(
                    rq, nu_feasible=float(rq.nu_old + d),
                    clipped_by_envelope=True)
            return True

        by_t: Dict[Tuple[str, str], List[str]] = {}
        for j in alloc:
            by_t.setdefault((claims[j].tenant, claims[j].resource), []).append(j)
        for (tid, res), ids in by_t.items():
            env, used = ran.headroom(tid, res)
            total = used + sum(out[j].dose for j in ids)
            if total > env + C2_TOL and not shrink(ids, total - env):
                return False, out
        if self.c2_cell_pool:
            committed = {t2: ran.headroom(t2, "prb")[1]
                         for t2 in getattr(ran, "slices", {})}
            total_all = sum(committed.values()) + sum(
                out[j].dose for j in alloc if claims[j].resource == "prb")
            pool = float(getattr(ran, "n_prb", float("inf")))
            if total_all > pool + C2_TOL and \
                    not shrink([j for j in alloc
                                if claims[j].resource == "prb"],
                               total_all - pool):
                return False, out
        return True, out

    def envelope_ok(self, subset: Sequence[str], claims: Dict[str, Claim],
                    reqs: Dict[str, ClaimRequest],
                    controls: Dict[str, float], ran) -> bool:
        """C2 feasibility, allowing pool-conserving transfers if enabled."""
        return self.fit_requests(subset, claims, reqs, controls, ran)[0]

    def _envelope_ok_strict(self, subset: Sequence[str],
                            claims: Dict[str, Claim],
                            reqs: Dict[str, ClaimRequest],
                            controls: Dict[str, float], ran) -> bool:
        """C2: a tenant's co-authorised allocative claims fit its envelope."""
        by_tenant: Dict[Tuple[str, str], float] = {}
        for j in subset:
            c = claims[j]
            if c.kind != Kind.ALLOCATIVE or not c.resource:
                continue
            key = (c.tenant, c.resource)
            if key not in by_tenant:
                env, used = ran.headroom(c.tenant, c.resource)
                by_tenant[key] = used
            by_tenant[key] += reqs[j].dose
        for (tid, res), total in by_tenant.items():
            env, _ = ran.headroom(tid, res)
            if total > env + C2_TOL:
                return False
        # FINITE RESOURCES.  The tenant envelopes are contractual and are
        # oversold relative to the cell, as operators do, so they alone do
        # not stop the reservations from summing past the physical pool.
        # With c2_cell_pool the total reserved PRB after this subset must
        # also fit the pool, so a claim that does not fit is REJECTED rather
        # than silently shaving every other tenant.
        if self.c2_cell_pool and by_tenant:
            total_all = 0.0
            for tid in getattr(ran, "slices", {}):
                key = (tid, "prb")
                if key in by_tenant:
                    total_all += by_tenant[key]
                else:
                    total_all += ran.headroom(tid, "prb")[1]
            if total_all > float(getattr(ran, "n_prb", float("inf"))) + C2_TOL:
                return False
        return True

    def enumerate_candidates(self, reqs: Dict[str, ClaimRequest],
                             claims: Dict[str, Claim]) -> List[Portfolio]:
        """Every conflict-free subset, with an explicit complexity guard.

        The number of subsets is 2^|J|.  For the claim counts a near-RT
        RIC actually sees (5-20) that is between 32 and a million.  The
        guard is what motivates the DRL proposer: above
        ``max_enumerate_claims`` the deterministic arbiter must fall back
        to a greedy search and loses optimality, and ``agent.csv`` records
        how often that happens.
        """
        jids = sorted(reqs)
        if len(jids) > self.max_enumerate:
            self.enumeration_truncated += 1
            return self._greedy_chain(reqs, claims)
        out: List[Portfolio] = []
        for r in range(len(jids) + 1):
            for sub in itertools.combinations(jids, r):
                if self.conflict_free(sub, claims):
                    out.append(sub)
        return out

    def _greedy_chain(self, reqs: Dict[str, ClaimRequest],
                      claims: Dict[str, Claim]) -> List[Portfolio]:
        """Nested greedy portfolios: (), (best), (best,2nd), ...

        Used when exhaustive enumeration is out of budget, and always
        added as a deterministic fallback next to the DRL proposals.
        """
        order = sorted(reqs, key=lambda j: (-abs(reqs[j].dose), j))
        chain: List[Portfolio] = [()]
        cur: List[str] = []
        for j in order:
            trial = cur + [j]
            if self.conflict_free(trial, claims):
                cur = trial
                chain.append(tuple(cur))
        return chain

    # ------------------------------------------------------------------
    # step 4: prediction
    # ------------------------------------------------------------------
    def predict(self, subset: Sequence[str], reqs: Dict[str, ClaimRequest],
                claims: Dict[str, Claim], intents: Dict[str, Intent],
                sens: SensitivityBase, regimes: RegimeReport
                ) -> Tuple[Dict[str, float], Dict[str, float], List[str]]:
        dg = {i: 0.0 for i in intents}
        var = {i: 0.0 for i in intents}
        unknown: List[str] = []
        for j in subset:
            c = claims[j]
            d = reqs[j].dose
            if abs(d) < 1e-12:
                continue
            for iid in intents:
                r = regimes.of_intent(iid)
                s = sens.get(r, c.param, iid)
                known = sens.known(r, c.param, iid)
                dg[iid] += s * d
                # Uncertainty is counted ONLY for intents the write is
                # predicted to affect -- as the original INTACT safety check
                # does (it iterates over sens.affected_intents(...), i.e.
                # |slope| >= sigma_min) -- or whose effect is UNKNOWN, since a
                # missing entry means "refuse", never "assume zero".  Counting
                # every pair let measurement noise on structurally isolated
                # pairs (raising T2's hard-sliced reservation cannot touch
                # T3) push a near-floor intent below it, so the robust floor
                # rejected raises that affect nobody, and even the oracle
                # left 34 of 106 PRBs stranded.
                if abs(s) >= getattr(sens, "sigma_min", 5e-4) or not known:
                    sd = sens.sigma(r, c.param, iid)
                    var[iid] += (sd * d) ** 2
                if not known and \
                        abs(d) * sens.unknown_sigma > self.unknown_impact:
                    unknown.append(iid)
        sig = {i: float(np.sqrt(v)) for i, v in var.items()}
        return dg, sig, sorted(set(unknown))

    # ------------------------------------------------------------------
    # steps 4-6: score, filter, select
    # ------------------------------------------------------------------
    def score(self, subset: Sequence[str], reqs, claims, intents, g_now,
              sens, regimes, controls, ran,
              protected: Optional[Set[str]] = None) -> PortfolioScore:
        c2_ok, preqs = self.fit_requests(subset, claims, reqs, controls, ran)
        dg, sig, unknown = self.predict(subset, {**reqs, **preqs}, claims,
                                        intents, sens, regimes)
        g_hat = {i: g_now.get(i, 0.0) + dg[i] for i in intents}
        g_low = {i: g_hat[i] - self.beta * sig[i] for i in intents}

        ok, reason = True, ""
        if not self.conflict_free(subset, claims):
            ok, reason = False, "C1 conflict"
        elif not c2_ok:
            ok, reason = False, "C2 envelope"
        else:
            prot = protected if protected is not None else set(intents)
            for iid in prot:
                it = intents[iid]
                if self.protect_only_safe and g_now.get(iid, 0.0) < it.epsilon:
                    # an already-breached intent is not protected by this
                    # rule; it is rescued by the objective instead
                    continue
                if g_low[iid] < it.epsilon - 1e-12:
                    ok, reason = False, f"safety floor {iid}"
                    break
            if ok and unknown and self.abstain_on_unknown:
                ok, reason = False, f"unknown effect {','.join(unknown[:3])}"

        # ---- objective -------------------------------------------------
        # The headline metric is weighted intent FULFILMENT: the weighted
        # fraction of intents with g >= 0.  That is a threshold, and a
        # linear sum of predicted margin gains is not a monotone surrogate
        # for it -- moving an already-comfortable intent from +0.50 to
        # +1.00 scores exactly as well as rescuing one from -0.01 to +0.49,
        # though only the second changes the metric.  Optimising the linear
        # surrogate therefore spends the actuation budget where it cannot
        # pay, which is why adding writes did not add fulfilment and why
        # even PERFECT slopes bought almost nothing over a frozen table.
        #
        # 'threshold' scores the expected CHANGE IN FULFILMENT instead:
        # P(g_i >= 0 after the writes) = Phi(g_hat_i / sigma_i).  This
        # prices uncertainty where it actually matters -- near the
        # threshold -- rather than as a flat penalty.  A small linear term
        # breaks ties, because the threshold objective is flat across the
        # many portfolios that flip no indicator, and a flat objective
        # makes the search arbitrary.
        #
        # This is a property of the ARBITER, not of any one method: every
        # controller in the registry uses whichever objective is
        # configured, so switching it cannot advantage one of them.
        w = {i: (intents[i].pi_class * intents[i].weight
                 if self.use_weights else 1.0) for i in intents}
        wsum = sum(w.values()) or 1.0
        if self.objective == "threshold":
            util = 0.0
            for i in intents:
                # The probit width is how far g must clear zero before the
                # arbiter believes the intent is really satisfied.  A single
                # global constant is wrong: it is a claim about prediction
                # error, and prediction error differs by an order of
                # magnitude between a smooth throughput intent and a
                # queueing-delay one.  Use each intent's own measured
                # residual spread, floored so a cold start is not
                # overconfident.  With a constant 0.03 the arbiter chased
                # threshold flips far inside its own noise and caused 209
                # safety crossings where the linear objective caused 21.
                s_i = max(sig[i], self.resid_sd.get(i, self.prob_sigma_floor),
                          self.prob_sigma_floor)
                p_after = _norm_cdf(g_hat[i] / s_i)
                p_before = 1.0 if g_now.get(i, 0.0) >= 0.0 else 0.0
                util += w[i] * (p_after - p_before)
            util /= wsum
            util += self.tie_break * sum(w[i] * dg[i]
                                         for i in intents) / wsum
        else:
            util = sum(w[i] * dg[i] for i in intents) / wsum
        util -= self.lambda_risk * sum(w[i] * sig[i]
                                       for i in intents) / wsum
        n_w = sum(1 for j in subset if abs(reqs[j].dose) > 1e-12)
        util -= self.action_cost * n_w
        if self.lambda_churn > 0:
            util -= self.lambda_churn * sum(
                abs(reqs[j].dose) / max(claims[j].width, 1e-9)
                for j in subset)
        return PortfolioScore(portfolio_key(subset), float(util), dg, sig,
                              g_hat, g_low, ok, reason, n_w, unknown,
                              reqs=preqs)

    # ------------------------------------------------------------------
    def decide(self, *, epoch: int, proposals: Dict[str, float],
               claims: Dict[str, Claim], intents: Dict[str, Intent],
               g_now: Dict[str, float], sens: SensitivityBase,
               regimes: RegimeReport, controls: Dict[str, float], ran,
               candidates: Optional[Sequence[Portfolio]] = None,
               protected: Optional[Set[str]] = None,
               force: bool = False
               ) -> Tuple[List[Decision], PortfolioScore, Dict]:
        """Run one arbitration epoch.  Returns (decisions, winner, info).

        ``force`` executes the single supplied candidate WITHOUT scoring it
        and without the no-action fallback.  It exists only for the
        degenerate baselines: "accept everything" and "rank by priority"
        are defined as controllers that do not deliberate, and scoring them
        would quietly turn them into the very thing they are meant to be a
        foil for.  No real method may set it.
        """
        t0 = time.perf_counter()
        reqs = self.normalise(proposals, claims, controls, ran)
        self.last_reqs = reqs

        if force:
            sub = portfolio_key(candidates[0]) if candidates else ()
            winner = self.score(sub, reqs, claims, intents, g_now, sens,
                                regimes, controls, ran, protected)
            winner.admissible = True
            winner.reject_reason = "forced (non-deliberative baseline)"
            decisions = self._make_decisions(epoch, winner, {**reqs, **winner.reqs}, claims,
                                             intents, g_now, sens, regimes)
            self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
            return decisions, winner, {
                "candidates": 1, "candidate_source": "forced",
                "feasible": 1, "latency_ms": self.last_latency_ms,
                "winner_utility": winner.utility,
                "winner_writes": winner.n_writes,
                "unknown_intents": len(winner.unknown_intents),
                "enumeration_truncated": False}

        if candidates is None:
            cands = self.enumerate_candidates(reqs, claims)
            source = "exhaustive" if len(reqs) <= self.max_enumerate \
                else "greedy"
        else:
            cands = [portfolio_key(c) for c in candidates]
            # ALWAYS add the deterministic fallbacks.  A learned proposer
            # that returns nothing useful must not be able to stop the
            # arbiter from acting.
            cands.append(())
            if self.greedy_fallback:
                cands.extend(self._greedy_chain(reqs, claims))
            cands = list(dict.fromkeys(cands))
            source = "topk"

        scored = [self.score(s, reqs, claims, intents, g_now, sens, regimes,
                             controls, ran, protected) for s in cands]
        self.last_scores = scored
        self.last_candidates = len(cands)
        feasible = [s for s in scored if s.admissible]
        if not feasible:
            # every candidate fails: take the no-action portfolio, which is
            # always feasible by construction, and record why.
            none = self.score((), reqs, claims, intents, g_now, sens,
                              regimes, controls, ran, protected)
            none.admissible = True
            none.reject_reason = "all candidates infeasible -> abstain"
            feasible = [none]
        winner = max(feasible, key=lambda s: (s.utility, -s.n_writes,
                                              tuple(s.claims)))

        decisions = self._make_decisions(epoch, winner, {**reqs, **winner.reqs}, claims, intents,
                                         g_now, sens, regimes)
        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
        info = {
            "candidates": len(cands),
            "candidate_source": source,
            "feasible": len(feasible),
            "latency_ms": self.last_latency_ms,
            "winner_utility": winner.utility,
            "winner_writes": winner.n_writes,
            "unknown_intents": len(winner.unknown_intents),
            "enumeration_truncated": self.enumeration_truncated,
        }
        return decisions, winner, info

    # ------------------------------------------------------------------
    def _make_decisions(self, epoch, winner, reqs, claims, intents, g_now,
                        sens, regimes) -> List[Decision]:
        """Turn the winning portfolio into ordered, audited write records.

        Executed most-valuable-first with a running margin ledger, so a
        later write sees the predicted effect of the earlier ones.  Every
        claim that was proposed but not selected gets an explicit REJECT
        record -- silence is not an audit trail.
        """
        chosen = set(winner.claims)
        # value of each selected claim, for the execution order
        val: Dict[str, float] = {}
        for j in chosen:
            c = claims[j]
            d = reqs[j].dose
            v = 0.0
            for iid, it in intents.items():
                r = regimes.of_intent(iid)
                v += it.pi_class * it.weight * sens.get(r, c.param, iid) * d
            val[j] = v
        order = sorted(chosen, key=lambda j: (-val.get(j, 0.0), j))

        ledger = dict(g_now)
        decisions: List[Decision] = []
        for j in order:
            c = claims[j]
            rq = reqs[j]
            implicated = []
            pred = {}
            sigmas = {}
            regs = {}
            for iid in intents:
                r = regimes.of_intent(iid)
                s = sens.get(r, c.param, iid)
                if abs(s) > 0 or not sens.known(r, c.param, iid):
                    implicated.append(iid)
                pred[iid] = s * rq.dose
                sigmas[iid] = sens.sigma(r, c.param, iid) * abs(rq.dose)
                regs[iid] = r
            reason = "selected"
            outcome = Outcome.ADMIT
            if rq.clipped_by_trust or rq.clipped_by_envelope:
                outcome = Outcome.OVERRIDE
                reason = ("trust region" if rq.clipped_by_trust
                          else "") + ("+envelope"
                                      if rq.clipped_by_envelope else "")
            for iid in intents:
                ledger[iid] = ledger.get(iid, 0.0) + pred[iid]
            decisions.append(Decision(
                jid=j, param=c.param, nu_old=rq.nu_old, nu_req=rq.nu_req,
                nu_star=rq.nu_feasible, outcome=outcome, reason=reason,
                implicated=implicated, predicted_dg=pred, sigma=sigmas,
                regime=regs, epoch=epoch))

        for j in sorted(set(reqs) - chosen):
            rq = reqs[j]
            decisions.append(Decision(
                jid=j, param=rq.param, nu_old=rq.nu_old, nu_req=rq.nu_req,
                nu_star=rq.nu_old, outcome=Outcome.REJECT,
                reason="not in winning portfolio", epoch=epoch))
        return decisions


# ---------------------------------------------------------------------------
def greedy_portfolio(reqs, claims, intents, g_now, sens, regimes,
                     weights: Optional[Dict[str, float]] = None) -> Portfolio:
    """Deterministic greedy selection used as a DRL fallback candidate."""
    w = weights or {i: intents[i].pi_class * intents[i].weight
                    for i in intents}
    gain = {}
    for j, rq in reqs.items():
        c = claims[j]
        g = 0.0
        for iid in intents:
            r = regimes.of_intent(iid)
            g += w.get(iid, 1.0) * sens.get(r, c.param, iid) * rq.dose
        gain[j] = g
    out: List[str] = []
    used: Set[str] = set()
    for j in sorted(gain, key=lambda k: (-gain[k], k)):
        if gain[j] <= 0:
            break
        if claims[j].param in used:
            continue
        out.append(j)
        used.add(claims[j].param)
    return portfolio_key(out)
