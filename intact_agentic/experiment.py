"""The epoch loop.

ONE loop runs every method.  What a method changes is only which objects
get plugged into four slots (sensitivity, regime, candidate source,
supervisor).  Everything else -- plant, xApps, proposal stream, C1/C2
tests, trust region, robust safety floor, execution order, telemetry --
is shared code on a shared random stream.  If a method wins here, it is
not winning because it was given a different experiment.

Epoch structure
---------------
    activate any tenant whose arrival epoch has come
    PRE  half   : ran.step(pre_slots)   -> kpm_before, g_before
    estimate the regime from kpm_before
    poll the xApps ONCE                 -> proposals
    supervisor may inject a probe as a synthetic claim
    arbiter.decide(...)                 -> decisions, winner
    execute the admitted writes
    POST half   : ran.step(post_slots)  -> kpm_after, g_after
    grade the model: sens.observe(predicted vs observed)
    train the context head and the policy
    supervisor.tick: staleness -> promotion -> rollback
    record, and checkpoint every N epochs

The PRE/POST split is load-bearing.  A single-half epoch cannot separate
"the margin moved because we wrote something" from "the margin moved
because the plant moved", and that separation is the entire basis of the
residual that grades the sensitivity model.
"""

from __future__ import annotations

import json
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .arbiter.core import Arbiter, greedy_portfolio
from .arbiter.margins import MarginTracker, margin
from .arbiter.oracle import OracleSensitivity
from .arbiter.regime import build_regime_estimator
from .arbiter.sensitivity import (OnlineSensitivity, StaticSensitivity,
                                  offline_sweep)
from .config import epoch_slots, scenario_name
from .methods import Method
from .ran.simulator import RealisticRAN
from .report.metrics import EpochRecord, MetricAccumulator
from .telemetry import Telemetry
from .types import Claim, Decision, Kind, Outcome, Registry, Scope, portfolio_key
from .arbiter.core import C2_TOL
from .xapps import build_xapps, collect_proposals


# ---------------------------------------------------------------------------
class _NullSensitivity(StaticSensitivity):
    """For the degenerate baselines, which consult no model at all."""

    def __init__(self, cfg):
        super().__init__(cfg, {})


# ---------------------------------------------------------------------------
class Experiment:
    """One (method, scenario, seed) run."""

    def __init__(self, cfg: Dict, method: Method, registry: Registry,
                 rundir: os.PathLike, seed: Optional[int] = None,
                 prior: Optional[StaticSensitivity] = None,
                 telemetry: bool = True, log=print):
        self.cfg = dict(cfg)
        self.method = method
        self.registry = registry
        self.rundir = Path(rundir)
        self.rundir.mkdir(parents=True, exist_ok=True)
        self.log = log or (lambda *a, **k: None)

        run = cfg.get("run", {}) or {}
        self.seed = int(seed if seed is not None else run.get("seed", 0))
        self.n_epochs = int(run.get("epochs", 400))
        self.checkpoint_every = int(run.get("checkpoint_every", 50))
        self.log_every = int(run.get("log_every", 25))
        self.probe_as_claim = bool(run.get("probe_as_claim", True))
        # Paired no-write counterfactual for CAUSAL metrics.  Costs one extra
        # post-half per epoch (about +50% wall time) and changes nothing
        # the controller sees: the clone is stepped after the decision and
        # discarded, and it draws from its OWN copy of the random stream.
        self.counterfactual = bool(run.get("counterfactual", True))
        # optional shadow sensitivity model for the decision-relevance test
        self.shadow_sens = None
        self.shadow_log: List[Dict] = []
        self.shadow_horizon = int(run.get("shadow_horizon_epochs", 4))
        o = cfg.get("oracle", {}) or {}
        self.rollout_top_m = int(o.get("rollout_top_m", 6))
        self.rollout_reps = int(o.get("rollout_reps", 2))
        self.rollout_horizon = int(o.get("rollout_horizon_epochs", 2))

        # a method that uses the channel axis must say so BEFORE the
        # regime estimator and the sensitivity table are built, because
        # the table is keyed by regime label
        self.cfg["arbiter"] = dict(self.cfg["arbiter"])
        self.cfg["arbiter"]["channel_aware_regime"] = bool(
            method.channel_aware)
        for dotted, val in (method.overrides or {}).items():
            sec, _, key = dotted.partition(".")
            self.cfg.setdefault(sec, {})
            self.cfg[sec] = dict(self.cfg[sec])
            # merge a nested dict instead of replacing it, so an override of
            # one setting cannot silently reset its siblings to code defaults
            if isinstance(val, dict) and isinstance(self.cfg[sec].get(key),
                                                    dict):
                self.cfg[sec][key] = {**self.cfg[sec][key], **val}
            else:
                self.cfg[sec][key] = val

        self.cfg["_method"] = method.name
        self.cfg["_run_id"] = f"{scenario_name(cfg)}__{method.name}__s{self.seed}"
        self.cfg["ran"] = dict(self.cfg["ran"])
        self.cfg["ran"]["seed"] = self.seed

        self.rng = np.random.default_rng(self.seed ^ 0x5EED)
        self.telemetry = Telemetry(self.rundir, self.cfg, enabled=telemetry)
        self.ran = RealisticRAN(self.cfg, telemetry=self.telemetry)
        self.xapps = build_xapps(self.cfg)
        self.arbiter = Arbiter(self.cfg, registry.tenants)
        self.tracker = MarginTracker(window=40)
        # ---- burn-in / evaluation separation -------------------------
        # Epochs [0, burn_in) are a learning phase and are NOT scored.  At
        # the boundary every method is handed the SAME plant state -- the
        # passive plant at that epoch, provisioned controls, identical
        # random draws -- so damage done while learning is discarded and
        # only what was LEARNED carries over.  Without this, one controller
        # releasing a reservation too early during warm-up (measured: pre-
        # change fulfilment 0.55 against ~0.76) decided a comparison that is
        # supposed to be about adapting to the later plant change.
        self.burn_in = int(run.get("burn_in_epochs", 0))
        self.reset_at_boundary = bool(run.get("reset_at_boundary", True))
        self.eval_mode = False
        self._boundary_done = self.burn_in <= 0
        self.metrics = MetricAccumulator(
            warmup_epochs=max(int(run.get("warmup_epochs", 0)), self.burn_in))

        self.prior = prior if prior is not None else StaticSensitivity(
            self.cfg, {})
        self.sens = self._build_sensitivity()
        self.head = self._build_head()
        self.regime_est = build_regime_estimator(
            self.cfg, method.regime, head=self.head)
        self.policy = self._build_policy()
        self.supervisor = self._build_supervisor()
        # digital-twin calibration loop (INTACT-RA-Agentic only)
        tw = (self.cfg.get("agent", {}) or {}).get("twin", {}) or {}
        self.twin = None
        if bool(tw.get("enabled", False)) and hasattr(self.sens,
                                                      "twin_observe"):
            from .agent.twin import DigitalTwin
            self.twin = DigitalTwin(self.cfg, registry.claims,
                                    registry.intents, self.seed)
        self.twin_every_eval = int(tw.get("every_eval", 1))
        self.twin_every_burn = int(tw.get("every_burn_in", 5))
        self.twin_inflate = float(tw.get("variance_inflation", 4.0))

        self.epoch = 0
        self.wall_s = 0.0
        self._sens_rows: List[Dict] = []
        self._new_tenants: List[str] = []
        self.pre_slots, self.post_slots = epoch_slots(self.cfg)

    # ------------------------------------------------------------------
    # component construction
    # ------------------------------------------------------------------
    def _build_sensitivity(self):
        s = self.method.sensitivity
        if s == "none":
            return _NullSensitivity(self.cfg)
        if s == "static":
            return self.prior
        if s == "static_pt":
            # the per-tenant offline sweep (ablation), cached separately
            from .config import scenario_name as _sn
            cache = Path("artifacts") / f"prior_pertenant_{_sn(self.cfg)}.json"
            return build_prior(self.cfg, self.registry, log=self.log,
                               cache=cache, per_tenant=True)
        if s == "online":
            return OnlineSensitivity(
                self.cfg, self.prior,
                params=sorted({c.param for c in self.registry.claims.values()}),
                intents=sorted(self.registry.intents))
        if s == "scrambled":
            from .arbiter.sensitivity import ScrambledSensitivity
            return ScrambledSensitivity(self.cfg, self.prior, seed=self.seed)
        if s == "kalman":
            from .arbiter.sensitivity import KalmanSensitivity
            k = KalmanSensitivity(
                self.cfg, self.prior,
                params=sorted({c.param for c in self.registry.claims.values()}),
                intents=sorted(self.registry.intents))
            k.bind(self.registry.claims, self.registry.intents)
            return k
        if s == "oracle":
            o = OracleSensitivity(self.cfg, self.registry.claims,
                                  self.registry.intents)
            o.attach_prior(self.prior)          # uncertainty parity
            return o
        raise ValueError(f"unknown sensitivity source {s!r}")

    def _build_head(self):
        if not self.method.learn_context:
            return None
        from .agent.context_head import ContextHead
        return ContextHead(self.cfg,
                           np.random.default_rng(self.seed ^ 0xC0DE))

    def _build_policy(self):
        if self.method.candidates != "drl":
            return None
        from .agent.candidate_policy import CandidatePolicy
        return CandidatePolicy(self.cfg,
                               np.random.default_rng(self.seed ^ 0xB0B))

    def _build_supervisor(self):
        if not self.method.supervisor:
            return None
        from .agent.supervisor import Supervisor
        return Supervisor(self.cfg, self.sens,
                          np.random.default_rng(self.seed ^ 0x50F1))

    # ------------------------------------------------------------------
    # one epoch
    # ------------------------------------------------------------------
    def _enter_evaluation(self) -> None:
        """Cross the burn-in boundary: common plant state, frozen proposer."""
        if self.reset_at_boundary:
            common = RealisticRAN(self.cfg)
            for e in range(self.burn_in):
                common.epoch = e
                common.step(self.pre_slots + self.post_slots, record=False)
            common.telemetry = self.telemetry
            self.ran = common
        # standard train/evaluate separation for the RL proposer; the online
        # sensitivity model, context head and probes keep adapting, because
        # adapting during evaluation IS the method under test
        self.eval_mode = True
        self._boundary_done = True
        self.log(f"    [{self.method.name}] epoch {self.epoch}: burn-in over; "
                 f"common plant state, proposer frozen")

    def step_epoch(self) -> EpochRecord:
        ep = self.epoch
        if not self._boundary_done and ep >= self.burn_in:
            self._enter_evaluation()
        t_wall = time.perf_counter()
        self.ran.epoch = ep

        # ---- mid-run arrivals -------------------------------------------
        added = self.ran.activate_pending(ep)
        if added:
            self._new_tenants.extend(added)
            self.log(f"    [epoch {ep}] tenant(s) arrived: "
                     f"{', '.join(added)}")
            if self.supervisor is not None:
                self.supervisor.register_new_entities(
                    ep, added, self.registry.active_intents(ep),
                    self.registry.active_claims(ep))

        tenants = self.registry.active_tenants(ep)
        intents = self.registry.active_intents(ep)
        claims = self.registry.active_claims(ep)
        # a claim whose parameter does not exist yet cannot be arbitrated
        # Decisions reason from the COMMANDED controls -- the policy that is
        # actually committed -- not from the actuators' lagged response.  A
        # claim's starting value, its dose, its trust region and its C2 room
        # are all statements about committed policy.  Reading the lagged
        # value made a knob that had just been lowered still count at its
        # old level, and measured every dose from where the actuator
        # happened to be rather than from what was last written.
        controls0 = self.ran.commanded_controls()
        claims = {j: c for j, c in claims.items() if c.param in controls0}

        # ---- PRE half ----------------------------------------------------
        kpm_pre = self.ran.step(self.pre_slots)
        g_before = {}
        for iid, it in intents.items():
            g = margin(it, kpm_pre)
            if g is not None:
                g_before[iid] = g
        self.tracker.update(intents, kpm_pre)

        # ---- latest telemetry for state-conditioned sensitivity priors ------
        if hasattr(self.sens, "set_context"):
            self.sens.set_context(self.ran, kpm_pre)

        # ---- regime ------------------------------------------------------
        regimes = self.regime_est.estimate(self.ran, kpm_pre,
                                           sorted(self.ran.slices), intents)


        # ---- proposals (identical stream for every method) ---------------
        controls = self.ran.commanded_controls()
        proposals = collect_proposals(self.xapps, claims, kpm_pre,
                                      controls, ep)

        # ---- oracle refresh (bound only) ---------------------------------
        # During evaluation the oracle can refresh EVERY epoch, restricted to
        # the knobs that have a proposal, so its slopes are never older than
        # the decision they inform.  That separates slope STALENESS (fixed
        # by refreshing) from MYOPIA (a one-step arbiter has no reason to
        # hold capacity for the next load phase, however current its slopes).
        if isinstance(self.sens, OracleSensitivity):
            every = int((self.cfg.get("oracle", {}) or {})
                        .get("refresh_every_eval", 0) or 0)
            if self.eval_mode and every > 0:
                if (ep - self.sens.last_refresh) >= every and proposals:
                    self.sens.refresh(self.ran, ep, force=True,
                                      params={claims[j].param
                                              for j in proposals})
            else:
                self.sens.refresh(self.ran, ep)

        # ---- supervisor probe, injected as a SYNTHETIC CLAIM -------------
        probe = None
        probe_jid = None
        if self.supervisor is not None:
            probe = self.supervisor.plan_probe(
                epoch=ep, claims=claims, intents=intents, margins=g_before,
                regimes=regimes, controls=controls)
        if probe is not None and self.probe_as_claim:
            src = next((c for c in claims.values() if c.param == probe.param),
                       None)
            if src is not None:
                probe_jid = "_probe"
                claims = dict(claims)
                # The probe is attributed to the tenant that OWNS the knob it
                # probes.  It was once labelled tenant="HOST": C2's envelope
                # check then filed the probe's dose under the host, which has
                # no PRB envelope, and admitted a probe plus that tenant's own
                # raise that together exceeded the tenant's envelope
                # (measured: agentic-noDRL, held-out seed 31003, epoch 199,
                # T2 at 46 + 2 + 6 = 54 > 52).  The write-time check, which
                # attributes by the knob written, caught it.
                claims[probe_jid] = Claim(
                    jid=probe_jid, xapp="_supervisor", tenant=src.tenant,
                    param=probe.param, scope=src.scope, kind=src.kind,
                    domain=src.domain, step=src.step,
                    resource=src.resource, d_bar=src.d_bar,
                    max_step_frac=src.max_step_frac, r_j=0.99)
                # the probe competes on the same terms: same trust region,
                # same C1/C2 tests, same robust safety floor.  A probe that
                # would break a promise is simply not selected.
                proposals = dict(proposals)
                proposals.pop(next((j for j, c in claims.items()
                                    if c.param == probe.param
                                    and j != probe_jid), ""), None)
                proposals[probe_jid] = float(
                    controls.get(probe.param, 0.0) + probe.delta)

        # ---- candidate set ----------------------------------------------
        candidates, pol_info = self._candidates(
            claims, proposals, intents, g_before, regimes, controls, kpm_pre,
            tenants)

        # ---- decide -------------------------------------------------------
        decisions, winner, info = self.arbiter.decide(
            epoch=ep, proposals=proposals, claims=claims, intents=intents,
            g_now=g_before, sens=self.sens, regimes=regimes,
            controls=controls, ran=self.ran, candidates=candidates,
            force=self.method.candidates in ("all", "none", "priority",
                                            "b3", "rollout"))

        # ---- B0 ALL-ADMIT, exactly as the original code defines it ---------
        # (intact/experiment.py, mode "none": "No mediation: every xApp
        # writes; exposes direct C1 conflicts.")  EVERY proposal is written,
        # in claim order, at its RAW requested value -- projected onto the
        # knob's physical domain and grid only, with no trust region and no
        # envelope clip -- and later writes see the state earlier ones
        # produced.  An earlier version kept one claim per parameter and
        # clipped every request to its envelope, so it could never violate
        # C1 or C2 and never paid for a conflict in the RAN.
        if self.method.candidates == "all":
            running = dict(controls)
            decisions = []
            for j in sorted(proposals):
                c = claims[j]
                nu = float(c.project(proposals[j]))
                decisions.append(Decision(
                    jid=j, param=c.param,
                    nu_old=float(running.get(c.param, nu)),
                    nu_req=float(proposals[j]), nu_star=nu,
                    outcome=Outcome.ADMIT, reason="B0 all-admit",
                    epoch=ep))
                running[c.param] = nu

        # ---- DECISION RELEVANCE (shadow decision, noise-free) -------------
        # Would a different sensitivity model have chosen a different
        # portfolio from these IDENTICAL proposals -- and if so, would that
        # choice have been better over the horizon in which a persistent
        # control's benefit actually accrues?  Both candidate portfolios are
        # applied to forks of the same live state with the same random
        # draws, so the difference is due to the decision alone.  This
        # never influences the run; it only measures it.
        if self.shadow_sens is not None:
            self._shadow_compare(ep, proposals, claims, intents, g_before,
                                 regimes, controls, candidates, winner)

        # ---- reference optimum, for search regret -------------------------
        best_u = float("nan")
        if self.method.candidates == "drl" and len(proposals) <= 12:
            ref = self.arbiter.enumerate_candidates(
                self.arbiter.normalise(proposals, claims, controls, self.ran),
                claims)
            reqs = self.arbiter.normalise(proposals, claims, controls,
                                          self.ran)
            scored = [self.arbiter.score(s, reqs, claims, intents, g_before,
                                         self.sens, regimes, controls,
                                         self.ran)
                      for s in ref]
            feas = [s for s in scored if s.admissible]
            if feas:
                best_u = max(s.utility for s in feas)
        elif info["candidate_source"] == "exhaustive":
            best_u = winner.utility

        # ---- counterfactual arm: the same plant, the same random draws, no
        # writes.  clone() carries the RNG, so traffic, fading, shadowing
        # and mobility are IDENTICAL in both arms and the only difference is
        # this epoch's decision.  g_after - g_cf is therefore the CAUSAL
        # effect of the decision, with the plant's own noise cancelled.
        cf_ran = self.ran.clone() if self.counterfactual else None

        # ---- execute -------------------------------------------------------
        doses: Dict[str, float] = {}
        write_rows: List[Dict] = []
        written: Dict[str, str] = {}
        n_c1 = n_c2 = 0
        # A MEDIATED portfolio is applied releases-first, raises-last, as a
        # controller pushing an RRM policy change does, so no intermediate
        # state commits more than the pool.  A tenant holding two knobs can
        # have one raised and one released in the same feasible portfolio;
        # executed in claim order, the raise could land first and the
        # per-write C2 check -- correctly -- flagged the transient overflow,
        # although the portfolio as a whole was feasible.  B0 keeps plain
        # claim order: under no mediation nobody sequences anything.
        if self.method.candidates != "all":
            decisions = sorted(decisions, key=lambda d: d.dnu if d.executed
                               else 0.0)
        for d in decisions:
            if not d.executed:
                continue
            c = claims[d.jid]
            # C1 and C2 are counted HERE, at the moment of writing, the same
            # way for every method -- as intact/experiment.py:543-553 does.
            # A violating write still lands: under no mediation the O-DU
            # executes what it is told.  Last writer wins on a C1 conflict,
            # and each write pays its own reconfiguration transient.
            if d.param in written:
                n_c1 += 1
            written[d.param] = d.jid
            if c.kind == Kind.ALLOCATIVE and c.resource:
                env, used = self.ran.headroom(c.tenant, c.resource)
                mine = float(self.ran.commanded_controls().get(d.param, 0.0))
                pool_over = False
                if self.arbiter.c2_cell_pool:
                    tot = sum(self.ran.headroom(t2, "prb")[1]
                              for t2 in self.ran.slices)
                    pool_over = tot - mine + d.nu_star > self.ran.n_prb + C2_TOL
                if used - mine + d.nu_star > env + C2_TOL or pool_over:
                    n_c2 += 1
            self.ran.apply(d.param, d.nu_star,
                           scope=c.scope.value, tenant=c.tenant)
            doses[d.param] = doses.get(d.param, 0.0) + d.dnu
            write_rows.append({"epoch": ep, "jid": d.jid, "param": d.param,
                               "nu_old": d.nu_old, "nu_star": d.nu_star,
                               "dose": d.dnu, "scope": c.scope.value,
                               "tenant": c.tenant,
                               "outcome": d.outcome.value,
                               "probe": int(d.jid == probe_jid)})

        # ---- POST half ------------------------------------------------------
        kpm_post = self.ran.step(self.post_slots)
        g_after = {}
        for iid, it in intents.items():
            g = margin(it, kpm_post)
            if g is not None:
                g_after[iid] = g

        g_cf = {}
        if cf_ran is not None:
            kpm_cf = cf_ran.step(self.post_slots, record=False)
            for iid, it in intents.items():
                g = margin(it, kpm_cf)
                if g is not None:
                    g_cf[iid] = g

        # ---- grade the model -------------------------------------------------
        predicted = {iid: 0.0 for iid in intents}
        for d in decisions:
            if not d.executed:
                continue
            for iid, v in d.predicted_dg.items():
                predicted[iid] = predicted.get(iid, 0.0) + v
        r_of_i = {iid: regimes.of_intent(iid) for iid in intents}
        if len(write_rows) > 0:
            self.arbiter.note_residuals(predicted, g_before, g_after)
        if hasattr(self.sens, "observe"):
            self.sens.observe(ep, r_of_i, doses, g_before, g_after, predicted,
                              probe=bool(probe_jid in
                                         {w["jid"] for w in write_rows}))

        # ---- learn ------------------------------------------------------------
        if self.head is not None:
            self.head.observe(self.ran, kpm_post, sorted(self.ran.slices))
        if self.policy is not None:
            realised = sum(g_after.get(i, 0.0) - g_before.get(i, 0.0)
                           for i in intents) / max(len(intents), 1)
            rew = self.policy.reward(
                chosen_utility=winner.utility,
                best_utility=best_u if np.isfinite(best_u) else winner.utility,
                realised_dg=realised, k=info["candidates"])
            self.policy.record(rew, ep, executed=[d.jid for d in decisions
                                                  if d.executed])
            if self.method.train_policy and not self.eval_mode:
                self.policy.update()

        # ---- digital-twin calibration (slow loop, one-epoch lag) ---------------
        # Measured on the plant state at the END of this epoch, folded in now,
        # so it informs decisions only from the next epoch.  Its compute is
        # accumulated separately from the near-RT decision latency.
        if self.twin is not None and proposals:
            every = self.twin_every_eval if self.eval_mode \
                else self.twin_every_burn
            if every > 0 and ep % every == 0:
                obs = self.twin.measure_slopes(
                    self.ran, {claims[j].param for j in proposals},
                    regimes, ep)
                self.sens.twin_observe(obs, inflate=self.twin_inflate)

        # ---- slow loop ---------------------------------------------------------
        if self.supervisor is not None:
            self.supervisor.note_decision(
                ep, winner, getattr(self.arbiter, 'last_reqs', {}), regimes)
            self.supervisor.tick(ep, list(set(r_of_i.values())), intents)

        # ---- record -------------------------------------------------------------
        rec = EpochRecord(
            epoch=ep, g_before=g_before, g_after=g_after, predicted=predicted,
            epsilon={i: it.epsilon for i, it in intents.items()},
            pi_class={i: it.pi_class for i, it in intents.items()},
            weight={i: it.weight for i, it in intents.items()},
            n_writes=len(write_rows), n_proposed=len(proposals),
            n_override=sum(1 for d in decisions
                           if d.outcome == Outcome.OVERRIDE),
            n_reject=sum(1 for d in decisions
                         if d.outcome == Outcome.REJECT),
            n_abstain=sum(1 for d in decisions
                          if d.outcome == Outcome.ABSTAIN),
            c1_violations=n_c1,
            c2_violations=n_c2,
            candidates=info["candidates"],
            candidate_source=info["candidate_source"],
            latency_ms=info["latency_ms"], utility=winner.utility,
            best_utility=best_u,
            probe=int(bool(probe_jid) and probe_jid in
                      {w["jid"] for w in write_rows}),
            reconfig_prb=float(kpm_post["_cell"].get("reconfig_prb", 0.0)),
            regimes=r_of_i,
            g_cf=g_cf,
            kpm_cell={k: v for k, v in kpm_post["_cell"].items()},
            unknown_entries=len(winner.unknown_intents),
        )
        self.metrics.add(rec)

        if self.telemetry.enabled:
            self.telemetry.record_epoch_kpm(ep, kpm_post)
            self.telemetry.record_decisions([d.as_row() for d in decisions])
            if write_rows:
                self.telemetry.record_writes(write_rows)
            self.telemetry.record_qos(self._qos_rows(ep, kpm_post))
            self.telemetry.record_agent(self._agent_row(ep, info, pol_info,
                                                        probe))
            new_rows = getattr(self.sens, "residuals", [])[len(self._sens_rows):]
            if new_rows:
                self.telemetry.record_sensitivity(new_rows)
                self._sens_rows.extend(new_rows)

        self.wall_s += time.perf_counter() - t_wall
        self.epoch += 1
        return rec

    # ------------------------------------------------------------------
    def _shadow_compare(self, ep, proposals, claims, intents, g_before,
                        regimes, controls, candidates, winner) -> None:
        if hasattr(self.shadow_sens, "refresh"):
            self.shadow_sens.refresh(self.ran, ep)
        saved = (getattr(self.arbiter, "last_reqs", None),
                 getattr(self.arbiter, "last_scores", None),
                 getattr(self.arbiter, "last_latency_ms", 0.0))
        s_dec, s_win, _ = self.arbiter.decide(
            epoch=ep, proposals=proposals, claims=claims, intents=intents,
            g_now=g_before, sens=self.shadow_sens, regimes=regimes,
            controls=controls, ran=self.ran, candidates=candidates)
        (self.arbiter.last_reqs, self.arbiter.last_scores,
         self.arbiter.last_latency_ms) = saved
        mine = tuple(sorted(winner.claims))
        theirs = tuple(sorted(s_win.claims))
        rec = {"epoch": ep, "agree": int(mine == theirs),
               "mine": "+".join(mine) or "-",
               "shadow": "+".join(theirs) or "-"}
        if mine != theirs:
            # which decision does this run actually execute?
            m_dec = self.arbiter._make_decisions(
                ep, winner, {**(self.arbiter.last_reqs or {}), **winner.reqs},
                claims, intents, g_before, self.sens, regimes)
            v = {}
            for tag, decs in (("mine", m_dec), ("shadow", s_dec)):
                fork = self.ran.clone()
                fork.rng = np.random.default_rng(
                    int(self.seed) * 7_919 + 104_729 * (ep + 1))
                for d in decs:
                    if d.executed:
                        c = claims[d.jid]
                        fork.apply(d.param, d.nu_star, scope=c.scope.value,
                                   tenant=c.tenant)
                fulf = []
                for _ in range(self.shadow_horizon):
                    k = fork.step(self.pre_slots + self.post_slots,
                                  record=False)
                    for iid, it in intents.items():
                        g = margin(it, k)
                        if g is not None:
                            fulf.append(1.0 if g >= 0 else 0.0)
                v[tag] = float(np.mean(fulf)) if fulf else float("nan")
            rec.update({"IF_mine": v["mine"], "IF_shadow": v["shadow"],
                        "shadow_minus_mine": v["shadow"] - v["mine"]})
        self.shadow_log.append(rec)

    def shadow_summary(self) -> Dict[str, float]:
        L = self.shadow_log
        if not L:
            return {}
        dis = [r for r in L if not r["agree"]]
        gains = [r["shadow_minus_mine"] for r in dis]
        n = len(L)
        return {
            "shadow_epochs": n,
            "disagreement_rate": len(dis) / n,
            # value of the shadow model's knowledge, per decision, averaged
            # over ALL epochs (zero wherever the two models agree)
            "value_per_epoch": (float(np.sum(gains)) / n) if n else 0.0,
            "value_when_disagree": (float(np.mean(gains)) if gains
                                    else float("nan")),
            "shadow_win_rate": (float(np.mean([g > 0 for g in gains]))
                                if gains else float("nan")),
            "shadow_loss_rate": (float(np.mean([g < 0 for g in gains]))
                                 if gains else float("nan")),
        }

    # ------------------------------------------------------------------
    def _candidates(self, claims, proposals, intents, g_before, regimes,
                    controls, kpm, tenants):
        """Produce the candidate portfolio list for this method."""
        mode = self.method.candidates
        reqs = self.arbiter.normalise(proposals, claims, controls, self.ran)

        if mode == "exhaustive":
            return None, {"policy": "exhaustive"}

        if mode == "none":                      # all-reject
            return [()], {"policy": "none"}

        if mode == "all":                       # all-accept
            keep, used = [], set()
            for j in sorted(reqs, key=lambda x: (claims[x].param, x)):
                p = claims[j].param
                if p in used:
                    continue        # a parameter can only take one value
                used.add(p)
                keep.append(j)
            return [portfolio_key(keep)], {"policy": "all"}

        if mode == "priority":                  # static priority ranking
            order = sorted(reqs, key=lambda j: (
                -(self.registry.tenants[claims[j].tenant].omega
                  * claims[j].r_j), j))
            keep, used = [], set()
            for j in order:
                if claims[j].param in used:
                    continue
                used.add(claims[j].param)
                keep.append(j)
            return [portfolio_key(keep)], {"policy": "priority"}

        if mode == "rollout":
            # ORACLE-R.  The same candidate set INTACT-RA would consider
            # (top-M by its own linear score, plus no-action), but chosen
            # by SIMULATING each one forward on forks of the live plant.
            # Futures are independent reseeded draws, NOT the live plant's
            # actual future: a fork that replayed the real future would be
            # clairvoyant about noise no causal controller can know, and
            # would overstate what is attainable.  C1/C2 are respected; the
            # choice is by realised fulfilment over the horizon.
            cands = self.arbiter.enumerate_candidates(reqs, claims)
            scored = [self.arbiter.score(s, reqs, claims, intents, g_before,
                                         self.prior, regimes, controls,
                                         self.ran) for s in cands]
            scored.sort(key=lambda s: -s.utility)
            pool = [s for s in scored[:self.rollout_top_m]]
            if not any(len(s.claims) == 0 for s in pool):
                pool.append(self.arbiter.score((), reqs, claims, intents,
                                               g_before, self.prior, regimes,
                                               controls, self.ran))
            best, best_v = (), -1.0
            for s in pool:
                decs = self.arbiter._make_decisions(
                    self.epoch, s, {**reqs, **s.reqs}, claims, intents,
                    g_before, self.prior, regimes)
                vals = []
                for r in range(self.rollout_reps):
                    fork = self.ran.clone()
                    fork.reseed(int(self.seed) * 104_729
                                + 7_919 * (self.epoch + 1) + 31 * r)
                    for d in decs:
                        if d.executed:
                            c = claims[d.jid]
                            fork.apply(d.param, d.nu_star,
                                       scope=c.scope.value, tenant=c.tenant)
                    for _ in range(self.rollout_horizon):
                        k = fork.step(self.pre_slots + self.post_slots,
                                      record=False)
                        for iid, it in intents.items():
                            g = margin(it, k)
                            if g is not None:
                                vals.append(1.0 if g >= 0 else 0.0)
                v = float(np.mean(vals)) if vals else -1.0
                if v > best_v + 1e-12:
                    best, best_v = tuple(s.claims), v
            return [portfolio_key(best)], {"policy": "rollout",
                                           "rollout_value": best_v,
                                           "rollout_pool": len(pool)}

        if mode == "b3":
            # Vq = sum_i s(regime_now, param_j, i) * dnu_j, UNWEIGHTED, at the
            # single cell-level regime.  Value is computed on the requests as
            # ADMITTED for that subset (raises resized by pool-conserving
            # transfers when enabled), and the best C1/C2-feasible subset is
            # chosen -- the legacy selector's rule, applied to the same
            # feasible set every mediated method sees.
            r_cell = regimes.cell
            import itertools as _it
            ids = sorted(reqs)
            best, best_v = (), 0.0
            if len(ids) <= 12:
                for n in range(1, len(ids) + 1):
                    for sub in _it.combinations(ids, n):
                        if not self.arbiter.conflict_free(list(sub), claims):
                            continue
                        ok, areq = self.arbiter.fit_requests(
                            list(sub), claims, reqs, controls, self.ran)
                        if not ok:
                            continue
                        v = sum(self.sens.get(r_cell, claims[k].param, i)
                                * areq[k].dose for k in sub for i in intents)
                        if v > best_v + 1e-12:
                            best, best_v = sub, v
            return [portfolio_key(best)], {"policy": "b3", "b3_value": best_v}

        if mode == "greedy":
            return [greedy_portfolio(reqs, claims, intents, g_before,
                                     self.sens, regimes)], {"policy": "greedy"}

        if mode == "drl":
            cands, info = self.policy.propose(
                ran=self.ran, kpm=kpm, tenants=sorted(self.ran.slices),
                claims=claims, reqs=reqs, intents=intents, margins=g_before,
                sens=self.sens, regimes=regimes, epoch=self.epoch,
                explore=self.method.train_policy and not self.eval_mode)
            return cands, info

        raise ValueError(f"unknown candidate mode {mode!r}")

    # ------------------------------------------------------------------
    def _count_c1(self, write_rows) -> int:
        """Two executed writes to the same parameter in one epoch."""
        seen, bad = set(), 0
        for w in write_rows:
            if w["param"] in seen:
                bad += 1
            seen.add(w["param"])
        return bad

    def _count_c2(self, write_rows) -> int:
        """A tenant's co-authorised allocative writes exceeding its envelope."""
        bad = 0
        by_tenant: Dict[str, float] = {}
        for w in write_rows:
            c = self.registry.claims.get(w["jid"])
            if c is None or c.kind != Kind.ALLOCATIVE or not c.resource:
                continue
            by_tenant[c.tenant] = by_tenant.get(c.tenant, 0.0) + w["nu_star"]
        for tid, tot in by_tenant.items():
            env = self.registry.tenants[tid].env("prb")
            if np.isfinite(env) and tot > env + 1e-6:
                bad += 1
        return bad

    def _qos_rows(self, ep, kpm) -> List[Dict]:
        rows = []
        for tid in self.ran.slices:
            r = kpm[tid]
            rows.append({"epoch": ep, "tenant": tid,
                         "throughput_mbps": r["throughput_mbps"],
                         "delay_ms": r["delay_ms"], "bler": r["bler"],
                         "delivery_pct": r["delivery_pct"],
                         "regret_throughput": r["regret_throughput"],
                         "regret_delay": r["regret_delay"],
                         "regret_bler": r["regret_bler"]})
        return rows

    def _agent_row(self, ep, info, pol_info, probe) -> Dict:
        row = {"epoch": ep, "candidates": info["candidates"],
               "candidate_source": info["candidate_source"],
               "latency_ms": info["latency_ms"],
               "winner_utility": info["winner_utility"],
               "winner_writes": info["winner_writes"],
               "policy": str(pol_info.get("policy", "")),
               "probe": int(probe is not None),
               "probe_param": probe.param if probe else "",
               "probe_voi": float(probe.voi) if probe else 0.0}
        for k in ("mean_p", "entropy", "k"):
            if k in pol_info:
                row[k] = float(pol_info[k])
        if hasattr(self.sens, "diagnostics"):
            for k, v in self.sens.diagnostics().items():
                if isinstance(v, (int, float)):
                    row[f"sens_{k}"] = float(v)
        if self.head is not None:
            for k, v in self.head.diagnostics().items():
                if isinstance(v, (int, float)):
                    row[f"head_{k}"] = float(v)
        if self.policy is not None:
            for k, v in self.policy.diagnostics().items():
                if isinstance(v, (int, float)):
                    row[f"pol_{k}"] = float(v)
        if self.supervisor is not None:
            for k, v in self.supervisor.diagnostics().items():
                if isinstance(v, (int, float)):
                    row[f"sup_{k}"] = float(v)
        return row

    # ------------------------------------------------------------------
    # run / resume
    # ------------------------------------------------------------------
    def run(self, until: Optional[int] = None) -> Dict:
        target = int(until if until is not None else self.n_epochs)
        t0 = time.perf_counter()
        while self.epoch < target:
            self.step_epoch()
            if self.log_every and self.epoch % self.log_every == 0:
                s = self.metrics.summary()
                self.log(f"    [{self.method.name}] epoch {self.epoch}/"
                         f"{target}  wIF={s.get('wIF', float('nan')):.3f}  "
                         f"cross={s.get('safety_crossings', 0)}  "
                         f"w/ep={s.get('writes_per_epoch', 0):.2f}  "
                         f"lat={s.get('latency_ms_mean', 0):.2f}ms")
            if self.checkpoint_every and \
                    self.epoch % self.checkpoint_every == 0:
                self.save_checkpoint()
        summary = self.finish()
        summary["wall_s"] = time.perf_counter() - t0
        return summary

    def finish(self) -> Dict:
        summary = self.metrics.summary()
        summary.update({
            "method": self.method.name, "label": self.method.label,
            "family": self.method.family,
            "scenario": scenario_name(self.cfg), "seed": self.seed,
            "epochs_run": self.epoch,
        })
        if hasattr(self.sens, "diagnostics"):
            summary.update({f"sens_{k}": v
                            for k, v in self.sens.diagnostics().items()
                            if isinstance(v, (int, float))})
        if self.supervisor is not None:
            summary.update({f"sup_{k}": v
                            for k, v in self.supervisor.diagnostics().items()
                            if isinstance(v, (int, float))})
            _write_json(self.rundir / "supervisor_events.json",
                        self.supervisor.event_rows())
        if self.policy is not None:
            summary.update({f"pol_{k}": v
                            for k, v in self.policy.diagnostics().items()
                            if isinstance(v, (int, float))})
        if self.twin is not None:
            summary.update({"twin_seconds_total": self.twin.seconds,
                            "twin_calls": self.twin.calls,
                            "twin_seconds_per_call": self.twin.seconds
                            / max(self.twin.calls, 1),
                            "twin_rollouts": self.twin.rollouts})
            summary.update({f"twin_{k}": v
                            for k, v in self.twin.describe().items()})
        if self.head is not None:
            summary.update({f"head_{k}": v
                            for k, v in self.head.diagnostics().items()
                            if isinstance(v, (int, float))})
        _write_json(self.rundir / "summary.json", summary)
        _write_rows_csv(self.rundir / "epochs.csv", self.metrics.rows())
        self.telemetry.close()
        return summary

    # ------------------------------------------------------------------
    # Everything an Experiment holds is checkpointed EXCEPT these: the
    # logger (not picklable), telemetry and its file handles, and objects
    # rebuilt identically from the configuration on load.
    _CKPT_DENY = frozenset({"log", "telemetry", "rundir", "cfg", "registry",
                            "method"})
    _CKPT_FORMAT = 2

    def save_checkpoint(self) -> Path:
        """Everything needed to resume this run bit-for-bit.

        The WHOLE experiment state is pickled in one piece -- the plant with
        its random streams, the metrics, and every learned or stateful
        component (sensitivity model, context head, proposer, supervisor,
        digital twin, arbiter, xApps, regime estimator) -- so shared
        references between components survive.  An earlier version saved
        components piecemeal and silently missed the Kalman posteriors (no
        loader existed, so they were skipped), the twin's random stream and
        the supervisor: a resumed INTACT-RA-Agentic run continued with an
        agent that had lost its memory.  Piecemeal saving is how such gaps
        open, one forgotten component at a time.
        """
        path = self.rundir / "checkpoint.pkl"
        ran_tel, self.ran.telemetry = self.ran.telemetry, None
        try:
            state = {k: v for k, v in self.__dict__.items()
                     if k not in self._CKPT_DENY}
            blob = {"format": self._CKPT_FORMAT, "method": self.method.name,
                    "seed": self.seed, "epoch": self.epoch, "state": state}
            tmp = path.with_suffix(".tmp")
            with open(tmp, "wb") as fh:
                pickle.dump(blob, fh, protocol=4)
            tmp.replace(path)                # atomic: never a torn checkpoint
        finally:
            self.ran.telemetry = ran_tel
        return path

    def load_checkpoint(self, path: Optional[os.PathLike] = None) -> bool:
        path = Path(path or (self.rundir / "checkpoint.pkl"))
        if not path.exists():
            return False
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
        if blob.get("method") != self.method.name:
            raise ValueError(
                f"checkpoint at {path} is for method {blob.get('method')!r}, "
                f"not {self.method.name!r}")
        if blob.get("format") != self._CKPT_FORMAT:
            raise ValueError(
                f"checkpoint at {path} uses an old format that did not save "
                f"every learned component, so resuming from it would continue "
                f"a DIFFERENT agent. Delete it and re-run from the start.")
        if int(blob.get("seed", self.seed)) != int(self.seed):
            raise ValueError(f"checkpoint at {path} is for seed "
                             f"{blob.get('seed')}, not {self.seed}")
        for k, v in blob["state"].items():
            setattr(self, k, v)
        self.ran.telemetry = self.telemetry
        self.log(f"    resumed {self.method.name} from epoch {self.epoch}")
        return True

# ---------------------------------------------------------------------------
def _ran_state(ran) -> Dict:
    import copy
    tel, ran.telemetry = ran.telemetry, None
    st = copy.deepcopy(ran.__dict__)
    ran.telemetry = tel
    st.pop("telemetry", None)
    st.pop("cfg", None)
    return st


def _restore_ran(ran, st: Dict) -> None:
    tel = ran.telemetry
    for k, v in st.items():
        setattr(ran, k, v)
    ran.telemetry = tel


def _try_state(obj):
    for name in ("state_dict", "to_json"):
        if hasattr(obj, name):
            try:
                return getattr(obj, name)()
            except Exception:
                return None
    return None


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=str)


def _write_rows_csv(path: Path, rows: Sequence[Dict]) -> None:
    import csv
    if not rows:
        return
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ---------------------------------------------------------------------------
def build_prior(cfg: Dict, registry: Registry, log=print,
                cache: Optional[os.PathLike] = None,
                force: bool = False,
                per_tenant: bool = False) -> StaticSensitivity:
    """Phase 1: the offline sweep that produces INTACT-RA's frozen table.

    Cached, because it is expensive and because EVERY method that uses a
    prior must use the SAME one -- re-sweeping per method would make the
    comparison depend on sweep noise.
    """
    from .arbiter.regime import all_regimes
    cache = Path(cache or (cfg.get("calibration", {}) or {})
                 .get("cache", "artifacts/sensitivity_prior.json"))
    if cache.exists() and not force:
        log(f"  loading cached offline table: {cache}")
        return StaticSensitivity.load(cfg, cache)
    log("  running the offline sensitivity sweep "
        "(mobility disabled, knobs frozen, common random numbers)...")
    ran = RealisticRAN(cfg)
    regimes = all_regimes(channel_aware=False)   # the prior is load-indexed
    tab = offline_sweep(ran, cfg, registry.intents, registry.claims,
                        regimes, log=lambda m: log("    " + m),
                        per_tenant=per_tenant)
    cache.parent.mkdir(parents=True, exist_ok=True)
    tab.save(cache)
    log(f"  offline table cached at {cache}")
    return tab
