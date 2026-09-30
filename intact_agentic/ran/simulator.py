"""
intact_agentic/ran/simulator.py
===============================
The RAN under control.  Puts mobility, traffic, the channel and the MAC
scheduler together and exposes exactly four things to the mediation plane:

    current_controls()   what is applied right now      (nu_old)
    apply(param, value)  commit one control value       (nu_star)
    step(n_slots)        advance and return KPMs
    headroom(t, r)       envelope accounting for the feasibility test

Everything above the four-method interface -- the arbiter, the DRL agent,
the reporting -- is backend agnostic, so swapping this file for an E2
adapter against srsRAN or OAI changes no mediation code.

WHAT THIS SIMULATOR MODELS THAT THE PREDECESSOR DID NOT
-------------------------------------------------------
1.  **Per-UE geometry and mobility.**  UEs have (x, y), velocity and a
    trajectory.  Distance is a state variable, not a fixed draw.
2.  **A real link budget.**  3GPP UMa path loss, LOS/NLOS, correlated
    shadowing, speed-dependent fading, a 3D antenna pattern with tilt, and
    explicit inter-cell interference (see ``channel.py``).
3.  **Link adaptation.**  CQI -> MCS -> spectral efficiency -> BLER ->
    HARQ, so "theoretical capacity", "achievable PHY rate" and "delivered
    application throughput" are three different, separately reported
    numbers.
4.  **Queues with finite buffers.**  Delay comes from Little's law on the
    actual backlog; jitter from the per-slot delay variance; drops from
    buffer overflow.
5.  **Actuation is not free.**  Two mechanisms:
      * FIRST-ORDER LAG.  A commanded value is approached exponentially
        with time constant ``actuation_time_constant_epochs``.  Power
        amplifiers and BWP reconfiguration are not instantaneous.
      * RECONFIGURATION TRANSIENT.  Every executed write costs the
        affected slices ``reconfig_loss_frac`` of their capacity for
        ``reconfig_slots`` slots, because the O-DU must reprogram the BWP
        / scheduler and the affected UEs briefly stop being served.  A
        CELL-scoped write (power, tilt) charges every slice; a slice-
        scoped write charges only its own slice; and TWO writes to the
        same parameter in one epoch charge the transient TWICE.

    Item 5 is what makes the benchmark non-degenerate.  Without a cost of
    acting, "write everything, every epoch" is weakly dominant and no
    arbiter can beat an all-accept baseline; with it, the question
    "which small set of writes is worth its transient?" is exactly the
    question INTACT-RA answers.  The cost is a declared, measured,
    physically-motivated quantity, reported in the run summary as
    ``reconfig_prb_cost`` so a reviewer can see how big it is.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .channel import MCS_MAX_INDEX, ChannelModel
from .mobility import MobilityModel
from .traffic import TrafficModel


# ---------------------------------------------------------------------------
@dataclass
class SliceState:
    tid: str
    n_ue: int
    queue_bits: np.ndarray
    dropped_bits: np.ndarray
    served_bits_total: np.ndarray
    offered_bits_total: np.ndarray
    delay_hist: deque = field(default_factory=lambda: deque(maxlen=64))
    active_from_epoch: int = 0


def _waterfill(weights: Dict[str, float], demands: Dict[str, float],
               pool: float) -> Dict[str, float]:
    """Weighted max-min fair division of ``pool`` capped at each demand."""
    alloc = {k: 0.0 for k in weights}
    active = {k for k in weights if demands.get(k, 0.0) > 1e-9
              and weights[k] > 0.0}
    rem = float(pool)
    for _ in range(len(weights) + 1):
        if not active or rem <= 1e-9:
            break
        wsum = sum(weights[k] for k in active)
        used, done = 0.0, set()
        for k in active:
            give = rem * weights[k] / wsum
            need = demands[k] - alloc[k]
            g = min(give, need)
            alloc[k] += g
            used += g
            if alloc[k] >= demands[k] - 1e-9:
                done.add(k)
        rem -= used
        if not done:
            break
        active -= done
    return alloc


def _waterfill_vec(w: np.ndarray, need: np.ndarray, pool: float) -> np.ndarray:
    """The same, over the UEs of one slice (proportional-fair weights)."""
    out = np.zeros_like(need, dtype=float)
    act = (need > 1e-9) & (w > 0)
    rem = float(pool)
    for _ in range(need.size + 1):
        if not act.any() or rem <= 1e-9:
            break
        share = np.where(act, w, 0.0)
        share = share / max(share.sum(), 1e-12) * rem
        give = np.minimum(share, need - out)
        out += give
        rem -= float(give.sum())
        full = out >= need - 1e-9
        if not (act & full).any():
            break
        act &= ~full
    return out


class RealisticRAN:
    """Multi-slice analytical-but-physical RAN with full telemetry."""

    # ------------------------------------------------------------------
    def __init__(self, cfg: Dict, telemetry=None):
        self.cfg = cfg
        self.ran = cfg["ran"]
        self.telemetry = telemetry
        self.n_prb = int(self.ran["n_prb"])
        self.prb_hz = float(self.ran["prb_bandwidth_hz"])
        self.slot_s = float(self.ran["slot_ms"]) / 1000.0
        self.base_delay_ms = float(self.ran.get("base_delay_ms", 2.0))
        self.max_queue_bits = float(self.ran.get("max_queue_kb", 250.0)) * 8e3
        self.max_delay_ms = float(self.ran.get("max_delay_ms", 1500.0))
        # 'legacy' = hard per-slice cap, scale-down only when oversubscribed
        # 'shares' = work-conserving weighted-fair water-filling
        self.allocation = str(self.ran.get("allocation", "legacy"))
        self.min_grant_prb = float(self.ran.get("min_grant_prb", 0.5))
        self.k_subband = float(self.ran.get("coupling_subband", 1.0))
        self.k_txpower = float(self.ran.get("coupling_txpower", 1.0))
        self.k_retx = float(self.ran.get("coupling_retx", 1.0))
        self.cio_load_gain = float(self.ran.get("cio_load_gain", 0.05))

        act = self.ran.get("actuation", {}) or {}
        self.tau_epochs = float(act.get("time_constant_epochs", 0.0))
        self.reconfig_slots = int(act.get("reconfig_slots", 0))
        self.reconfig_loss = float(act.get("reconfig_loss_frac", 0.0))
        self.reconfig_cell_multiplier = float(
            act.get("reconfig_cell_multiplier", 1.0))

        self.seed = int(self.ran.get("seed", cfg.get("seed", 0)))
        self.t = 0
        self.epoch = 0
        self._reconfig_cost_prb = 0.0
        self._write_events: List[Dict] = []
        self.reset(self.seed)

    # ------------------------------------------------------------------
    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.mobility = MobilityModel(self.cfg, self.rng)
        self.traffic = TrafficModel(self.cfg, self.rng)
        self.channel = ChannelModel(self.cfg, self.rng)
        self.t = 0
        self.epoch = 0
        self._reconfig_cost_prb = 0.0
        self._write_events = []
        self.slices: Dict[str, SliceState] = {}
        self._reconfig_left: Dict[str, int] = {}
        self._cell_reconfig_left = 0
        for tid, sl in (self.ran.get("slices") or {}).items():
            if int(sl.get("arrives_at_epoch", 0)) > 0:
                continue
            self._add_slice(tid, sl)
        self._controls = dict(self.ran["initial_controls"])
        self._command = dict(self._controls)
        self.channel.set_serving_power(self._controls.get("txpower", 46.0),
                                       self._controls.get("tilt", 6.0))

    def _add_slice(self, tid: str, sl: Dict) -> None:
        n = int(sl["n_ue"])
        self.mobility.spawn(tid, n, sl.get("placement"))
        self.traffic.spawn(tid, n, float(sl["load_mbps_per_ue"]),
                           spread=float(sl.get("load_spread", 0.25)))
        self.channel.register(tid, self.mobility.groups[tid].xy)
        self.slices[tid] = SliceState(
            tid=tid, n_ue=n,
            queue_bits=np.zeros(n), dropped_bits=np.zeros(n),
            served_bits_total=np.zeros(n), offered_bits_total=np.zeros(n),
            active_from_epoch=int(sl.get("arrives_at_epoch", 0)))
        self._reconfig_left[tid] = 0

    # ------------------------------------------------------------------
    def activate_pending(self, epoch: int) -> List[str]:
        """Bring mid-run tenants online.  Returns the tenants added."""
        added = []
        for tid, sl in (self.ran.get("slices") or {}).items():
            at = int(sl.get("arrives_at_epoch", 0))
            if at > 0 and at <= epoch and tid not in self.slices:
                self._add_slice(tid, sl)
                # a new slice needs its knobs to exist
                for k, v in (sl.get("initial_controls") or {}).items():
                    self._controls.setdefault(k, float(v))
                    self._command.setdefault(k, float(v))
                added.append(tid)
        return added

    # ------------------------------------------------------------------
    # control interface
    # ------------------------------------------------------------------
    def current_controls(self) -> Dict[str, float]:
        return dict(self._controls)

    def commanded_controls(self) -> Dict[str, float]:
        return dict(self._command)

    def apply(self, param: str, value: float, scope: str = "slice",
              tenant: Optional[str] = None) -> None:
        """Commit one control value and charge its reconfiguration cost."""
        value = float(value)
        changed = abs(value - float(self._command.get(param, value))) > 1e-12
        self._command[param] = value
        if self.tau_epochs <= 0.0:
            self._controls[param] = value
        if self.reconfig_slots > 0 and changed:
            if str(scope).lower() == "cell":
                self._cell_reconfig_left = self.reconfig_slots
                for tid in self.slices:
                    self._reconfig_left[tid] = max(
                        self._reconfig_left.get(tid, 0), self.reconfig_slots)
            else:
                tgt = tenant or self._param_tenant(param)
                if tgt in self._reconfig_left:
                    # Two writes to the same slice in one epoch charge the
                    # transient twice: the O-DU reprograms twice.
                    self._reconfig_left[tgt] = (self._reconfig_left[tgt]
                                                + self.reconfig_slots)
        self._write_events.append({"epoch": self.epoch, "slot": self.t,
                                   "param": param, "value": value,
                                   "scope": scope})
        if param in ("txpower", "tilt"):
            self.channel.set_serving_power(
                self._controls.get("txpower", 46.0),
                self._controls.get("tilt", 6.0))

    def _param_tenant(self, param: str) -> Optional[str]:
        for tid in self.slices:
            if param.endswith("_" + tid) or param.endswith(tid):
                return tid
        return None

    def _advance_actuators(self, n_slots: int) -> None:
        if self.tau_epochs <= 0.0:
            return
        pre = int(self.ran.get("pre_slots", self.ran["slots_per_epoch"]))
        post = int(self.ran.get("post_slots", self.ran["slots_per_epoch"]))
        epoch_slots = max(pre + post, 1)
        frac = 1.0 - math.exp(-max(int(n_slots), 0)
                              / max(self.tau_epochs * epoch_slots, 1e-9))
        for p, target in self._command.items():
            old = float(self._controls.get(p, target))
            self._controls[p] = old + frac * (float(target) - old)
        self.channel.set_serving_power(self._controls.get("txpower", 46.0),
                                       self._controls.get("tilt", 6.0))

    # ------------------------------------------------------------------
    def headroom(self, tenant: str, resource: str) -> Tuple[float, float]:
        """(envelope, currently committed) for the C2 feasibility test."""
        env = float((self.ran.get("envelopes", {}) or {})
                    .get(tenant, {}).get(resource, float("inf")))
        prefix = f"quota_{tenant}"
        used = 0.0
        # committed policy, not the lagged actuator state
        for k, v in self._command.items():
            if k == prefix or (k.startswith(prefix)
                               and not k[len(prefix):][:1].isdigit()):
                used += float(v)
        return env, used

    def quota_of(self, tid: str, controls: Optional[Dict] = None) -> float:
        c = controls if controls is not None else self._controls
        prefix = f"quota_{tid}"
        vals = [v for k, v in c.items()
                if k == prefix or (k.startswith(prefix)
                                   and not k[len(prefix):][:1].isdigit())]
        if vals:
            return float(sum(vals))
        return self.n_prb / max(len(self.slices), 1)

    # ------------------------------------------------------------------
    def reseed(self, seed: int) -> None:
        """Give this plant -- and EVERY stochastic sub-model -- a fresh stream.

        The mobility, traffic and channel models are constructed holding a
        reference to the plant's generator.  clone() deep-copies that shared
        reference, so on a clone they all keep drawing from the clone's copy
        of the ORIGINAL stream.  Rebinding ``self.rng`` alone therefore
        changes nothing any sub-model draws from, and every "independent
        replicate" silently replays the identical future.

        That is exactly right for a paired counterfactual (the two arms
        SHOULD share every draw) and exactly wrong for replication (the
        replicates must NOT).  Use clone() for the first and clone() then
        reseed() for the second.
        """
        g = np.random.default_rng(int(seed))
        self.rng = g
        for sub in ("mobility", "traffic", "channel"):
            obj = getattr(self, sub, None)
            if obj is not None and hasattr(obj, "rng"):
                obj.rng = g

    def clone(self):
        """Deep copy INCLUDING RNG state, for paired counterfactuals."""
        import copy as _c
        tel, self.telemetry = self.telemetry, None
        out = _c.deepcopy(self)
        self.telemetry = tel
        out.telemetry = None
        return out

    # ------------------------------------------------------------------
    def _subband_quality(self, controls: Dict[str, float]) -> Dict[str, float]:
        """Zero-sum subband quality from the scheduler-weight knobs.

        A slice whose weight rises steers its UEs onto better subbands,
        which necessarily leaves worse ones for everyone else.  Modelled
        as an explicitly conserved quantity so it can never be a free
        lunch -- this is cross-tenant coupling mechanism (b).
        """
        w = {t: max(float(controls.get(f"schedw_{t}", 1.0)), 1e-6)
             for t in self.slices}
        n_ue = {t: self.slices[t].n_ue for t in self.slices}
        raw = {t: w[t] ** (0.5 * self.k_subband) for t in self.slices}
        tot = max(sum(n_ue.values()), 1)
        norm = sum(raw[t] * n_ue[t] for t in self.slices) / tot
        return {t: raw[t] / max(norm, 1e-9) for t in self.slices}

    # ------------------------------------------------------------------
    def step(self, n_slots: int, record: bool = True
             ) -> Dict[str, Dict[str, float]]:
        """Advance the RAN and return averaged KPMs per tenant plus ``_cell``."""
        self._advance_actuators(n_slots)
        c = self._controls
        q = self._subband_quality(c)

        acc = {t: {k: [] for k in
                   ("tput", "phy_rate", "cap", "delay", "jitter", "buf",
                    "deliv", "bler", "sinr", "cqi", "mcs", "prb", "se",
                    "drop", "offered", "offered_slice", "retx_prb",
                    "util_prb", "delay_p50", "delay_p90")}
               for t in self.slices}
        cell_hist = {k: [] for k in
                     ("prb_used", "prb_retx", "prb_demand", "scale",
                      "offered_mbps", "delivered_mbps", "reconfig_prb")}

        for _ in range(int(n_slots)):
            self.t += 1
            self.mobility.step(1)
            self.traffic.advance(1)

            tx = float(c.get("txpower", 46.0))
            tilt = float(c.get("tilt", 6.0))
            plans: Dict[str, Dict] = {}
            total_req = 0.0
            total_retx = 0.0

            for tid, st in self.slices.items():
                grp = self.mobility.groups[tid]
                mcs_cap = int(round(float(c.get(f"mcs_{tid}",
                                                MCS_MAX_INDEX))))
                cio = float(c.get(f"cio_{tid}", 0.0))
                phy = self.channel.evaluate(
                    tid, grp.xy, grp.speed_mps(),
                    tx_dbm=self.channel.serving.tx_dbm
                    if self.k_txpower >= 1.0 else tx,
                    tilt_deg=tilt, cio_db=cio,
                    subband_quality=q[tid], mcs_cap=mcs_cap)

                arrivals = self.traffic.arrivals_bits(tid)
                # CIO also steers traffic: a positive offset attracts UEs
                # from neighbouring cells, which raises this slice's load.
                arrivals = arrivals * (1.0 + self.cio_load_gain * cio)
                backlog = st.queue_bits + arrivals

                # per-PRB delivered bits after HARQ overhead
                phy_bits_prb = self.prb_hz * phy["se_mcs"] * self.slot_s
                eff_bits_prb = np.maximum(
                    phy_bits_prb * (1.0 - phy["resid_bler"])
                    / np.maximum(phy["harq_tx"] ** self.k_retx, 1.0), 1.0)

                quota = self.quota_of(tid, c)
                cap = float(c.get(f"prbcap_{tid}", self.n_prb))
                eff_quota = float(min(quota, cap))
                pol = float(c.get(f"schedpol_{tid}", 0.5))
                pf_w = np.maximum(phy["se_mcs"], 1e-3) ** pol
                share = pf_w / max(pf_w.sum(), 1e-9)
                prb_avail = share * eff_quota
                prb_needed = backlog / eff_bits_prb
                prb_req = np.minimum(prb_avail, prb_needed)
                retx_prb = float((prb_req * (phy["harq_tx"] - 1.0)).sum())

                total_req += float(prb_req.sum())
                total_retx += retx_prb
                plans[tid] = {"phy": phy, "arrivals": arrivals,
                              "backlog": backlog, "prb_req": prb_req,
                              "eff_bits_prb": eff_bits_prb,
                              "phy_bits_prb": phy_bits_prb,
                              "retx_prb": retx_prb, "eff_quota": eff_quota,
                              "prb_needed": np.minimum(prb_needed, cap),
                              "pf_w": pf_w, "cap": cap}

            demand = total_req + total_retx
            if self.allocation == "shares":
                # WORK-CONSERVING WEIGHTED-FAIR SHARING (GPS-style).
                #
                # Each slice's quota is a WEIGHT, not a hard cap.  The whole
                # PRB pool is water-filled among backlogged slices in
                # proportion to weight; a slice that needs less than its
                # share keeps only what it needs and the remainder flows to
                # whoever is still backlogged.  This is how weighted slice
                # schedulers and RRM policy ratios behave in practice, and
                # it has two properties the legacy cap model lacks:
                #
                #   * no idle PRB while any slice has queued data, and
                #   * ZERO-SUM whenever two or more slices are backlogged:
                #     raising one slice's weight takes PRBs from the others.
                #
                # The legacy model capped each slice at share*quota and only
                # scaled down when total demand exceeded the pool.  With
                # quotas summing below the pool, raising a quota then cost
                # nobody anything -- measured, +8 PRB to T1 raised the NET
                # throughput of all three tenants by +1.17 Mb/s/UE -- so the
                # dominant policy was "raise everything" and the value of
                # knowing ANY slope was close to nil.
                wts = {t2: max(plans[t2]["eff_quota"], 0.0) for t2 in plans}
                dem = {t2: float(plans[t2]["prb_needed"].sum())
                       for t2 in plans}
                alloc = _waterfill(wts, dem, float(self.n_prb))
                for t2, pl2 in plans.items():
                    pl2["prb_final"] = _waterfill_vec(
                        pl2["pf_w"], pl2["prb_needed"], alloc[t2])
                scale = 1.0
            elif self.allocation in ("dedicated", "dedicated_shared"):
                # NORMALISED DEDICATED RESERVATIONS (3GPP rRMPolicy
                # dedicated-ratio semantics).
                #
                # Each slice's quota is a DEDICATED reservation: PRBs that
                # slice may use and nobody else may, whether or not it uses
                # them.  Reservations are normalised so they never exceed
                # the pool, exactly as the dedicated ratios of all slices
                # must sum to at most 100%:
                #
                #     res_t = quota_t * min(1, N / sum_k quota_k)
                #
                # Two properties follow, and this model is the only one of
                # the three here that has BOTH:
                #
                #   * OWN-STATE DEPENDENCE.  A slice gains from a larger
                #     reservation only while its own demand exceeds its
                #     current one; a slice below its reservation gains
                #     nothing.  So a tenant's quota slope is a function of
                #     THAT tenant's load -- which is exactly what a
                #     per-tenant regime label measures, and the mechanism
                #     by which INTACT-RA beat B3 in the original study.
                #   * ZERO-SUM.  With quotas summing above the pool,
                #     raising one reservation shrinks every other one.
                #     Giving PRBs to a tenant that cannot use them WASTES
                #     them, so misallocation is costly and the choice of
                #     write, not the number of writes, is the decision.
                #
                # 'legacy' has the first property but not the second (with
                # slack in the pool, raising a cap is free); 'shares' has
                # the second but not the first (unused capacity flows to
                # others, so slopes depend on everyone's joint backlog).
                qsum = sum(max(plans[t2]["eff_quota"], 0.0) for t2 in plans)
                norm = min(1.0, float(self.n_prb) / max(qsum, 1e-9))
                for t2, pl2 in plans.items():
                    res = max(pl2["eff_quota"], 0.0) * norm
                    pl2["prb_final"] = _waterfill_vec(
                        pl2["pf_w"], pl2["prb_needed"], res)
                    pl2["reservation"] = res
                if self.allocation == "dedicated_shared":
                    # Capacity NOT dedicated to any slice is SHARED, as in
                    # 3GPP RRM policy, rather than left idle.  Only the
                    # unreserved remainder is shared -- a slice's own unused
                    # dedicated PRBs still belong to it -- so the isolation
                    # that makes a tenant's quota slope depend on its own
                    # load is preserved.  What changes is that releasing
                    # reservations below the pool no longer strands PRBs:
                    # under plain 'dedicated', every controller walked the
                    # total reservation down to ~68 of 106 PRBs and starved
                    # the whole cell beside idle capacity.
                    remainder = max(float(self.n_prb) - sum(
                        pl2["reservation"] for pl2 in plans.values()), 0.0)
                    if remainder > 1e-9:
                        resid = {t2: float(np.maximum(
                            pl2["prb_needed"] - pl2["prb_final"], 0.0).sum())
                            for t2, pl2 in plans.items()}
                        extra = _waterfill({t2: 1.0 for t2 in plans}, resid,
                                           remainder)
                        for t2, pl2 in plans.items():
                            if extra[t2] > 1e-9:
                                pl2["prb_final"] = pl2["prb_final"] + \
                                    _waterfill_vec(
                                        pl2["pf_w"],
                                        np.maximum(pl2["prb_needed"]
                                                   - pl2["prb_final"], 0.0),
                                        extra[t2])
                scale = 1.0
            else:
                scale = 1.0 if demand <= self.n_prb else \
                    self.n_prb / max(demand, 1e-9)
            reconfig_prb = 0.0

            for tid, st in self.slices.items():
                pl = plans[tid]
                phy = pl["phy"]
                prb = (pl["prb_final"]
                       if self.allocation in ("shares", "dedicated",
                                              "dedicated_shared")
                       else pl["prb_req"] * scale)
                # reconfiguration transient
                loss = 0.0
                if self._reconfig_left.get(tid, 0) > 0:
                    loss = self.reconfig_loss
                    if self._cell_reconfig_left > 0:
                        loss = min(0.95, loss * self.reconfig_cell_multiplier)
                    self._reconfig_left[tid] -= 1
                if loss > 0:
                    reconfig_prb += float(prb.sum()) * loss
                    prb = prb * (1.0 - loss)

                delivered = np.minimum(prb * pl["eff_bits_prb"], pl["backlog"])
                phy_rate = prb * pl["phy_bits_prb"] / self.slot_s / 1e6
                capacity = (prb * self.prb_hz * phy["se_shannon"]
                            * self.slot_s) / self.slot_s / 1e6
                rem = np.maximum(pl["backlog"] - delivered, 0.0)
                dropped = np.maximum(rem - self.max_queue_bits, 0.0)
                st.queue_bits = np.minimum(rem, self.max_queue_bits)
                st.dropped_bits += dropped
                st.served_bits_total += delivered
                st.offered_bits_total += pl["arrivals"]

                # Queueing delay needs a defensible SERVICE RATE floor.  A
                # UE that got nothing this slot is not served at 0 bit/s
                # forever: it is served the next time the scheduler reaches
                # it.  The floor is therefore one PRB's worth of its own
                # current modulation, which is the smallest grant the MAC
                # can actually issue, never an arbitrary constant.
                floor_rate = np.maximum(
                    pl["eff_bits_prb"] * self.min_grant_prb / self.slot_s,
                    1e4)
                serve_rate = np.maximum(delivered / self.slot_s, floor_rate)
                delay_ms = np.minimum(
                    self.base_delay_ms + 1000.0 * st.queue_bits / serve_rate,
                    self.max_delay_ms)
                st.delay_hist.append(float(np.mean(delay_ms)))
                jitter = float(np.std(list(st.delay_hist))) \
                    if len(st.delay_hist) > 1 else 0.0

                off = float(pl["backlog"].sum())
                acc[tid]["tput"].append(float(np.mean(delivered / self.slot_s
                                                      / 1e6)))
                acc[tid]["phy_rate"].append(float(np.mean(phy_rate)))
                acc[tid]["cap"].append(float(np.mean(capacity)))
                acc[tid]["delay"].append(float(np.mean(delay_ms)))
                acc[tid]["delay_p50"].append(float(np.median(delay_ms)))
                acc[tid]["delay_p90"].append(float(np.percentile(delay_ms,
                                                                 90)))
                acc[tid]["jitter"].append(jitter)
                acc[tid]["buf"].append(float(np.mean(st.queue_bits) / 8e3))
                acc[tid]["deliv"].append(
                    100.0 * (1.0 - float(dropped.sum()) / max(off, 1e-9)))
                acc[tid]["bler"].append(float(np.mean(phy["bler"])))
                acc[tid]["sinr"].append(float(np.mean(phy["sinr_db"])))
                acc[tid]["cqi"].append(float(np.mean(phy["cqi"])))
                acc[tid]["mcs"].append(float(np.mean(phy["mcs"])))
                acc[tid]["se"].append(float(np.mean(phy["se_mcs"])))
                acc[tid]["prb"].append(float(prb.sum()))
                acc[tid]["retx_prb"].append(pl["retx_prb"] * scale)
                acc[tid]["drop"].append(float(dropped.sum()))
                # per-UE mean, so that offered and delivered are in the
                # SAME unit as throughput_mbps and their ratio is meaningful
                acc[tid]["offered"].append(
                    float(np.mean(pl["arrivals"])) / self.slot_s / 1e6)
                acc[tid]["offered_slice"].append(
                    float(pl["arrivals"].sum()) / self.slot_s / 1e6)
                acc[tid]["util_prb"].append(float(prb.sum()))

                if record and self.telemetry is not None:
                    self.telemetry.record_slot(
                        t=self.t, epoch=self.epoch, tid=tid,
                        group=self.mobility.groups[tid], phy=phy,
                        prb=prb, delivered=delivered, queue=st.queue_bits,
                        dropped=dropped, arrivals=pl["arrivals"],
                        delay_ms=delay_ms, slot_s=self.slot_s)

            if self._cell_reconfig_left > 0:
                self._cell_reconfig_left -= 1
            self._reconfig_cost_prb += reconfig_prb
            cell_hist["prb_used"].append(total_req * scale)
            cell_hist["prb_retx"].append(total_retx * scale)
            cell_hist["prb_demand"].append(demand)
            cell_hist["scale"].append(scale)
            cell_hist["reconfig_prb"].append(reconfig_prb)
            cell_hist["offered_mbps"].append(
                sum(acc[t]["offered_slice"][-1] for t in self.slices))
            cell_hist["delivered_mbps"].append(
                sum(acc[t]["tput"][-1] * self.slices[t].n_ue
                    for t in self.slices))

        return self._assemble_kpm(acc, cell_hist, n_slots)

    # ------------------------------------------------------------------
    def _assemble_kpm(self, acc, cell_hist, n_slots) -> Dict:
        kpm: Dict[str, Dict[str, float]] = {}
        for tid, st in self.slices.items():
            a = acc[tid]
            qos = self.traffic.qos.get(tid, {})
            tput = float(np.mean(a["tput"]))
            delay = float(np.mean(a["delay"]))
            bler = float(np.mean(a["bler"]))
            kpm[tid] = {
                "throughput_mbps": tput,
                "phy_rate_mbps": float(np.mean(a["phy_rate"])),
                "shannon_capacity_mbps": float(np.mean(a["cap"])),
                "delay_ms": delay,
                # A mean over a queueing window is dominated by a handful of
                # congested slots, which makes any derivative measured
                # against it mostly noise.  The median is both the more
                # standard latency SLA statistic and the one a local slope
                # can actually be estimated from; p90 is kept so the tail is
                # still reported rather than hidden.
                "delay_p50_ms": float(np.median(a["delay_p50"])),
                "delay_p90_ms": float(np.mean(a["delay_p90"])),
                "jitter_ms": float(np.mean(a["jitter"])),
                "buffer_kb": float(np.mean(a["buf"])),
                "delivery_pct": float(np.mean(a["deliv"])),
                "delivery_ratio": float(np.mean(a["deliv"])),
                "bler": bler,
                "sinr_db": float(np.mean(a["sinr"])),
                "cqi": float(np.mean(a["cqi"])),
                "mcs": float(np.mean(a["mcs"])),
                "spectral_efficiency": float(np.mean(a["se"])),
                "prb_alloc": float(np.mean(a["prb"])),
                "prb_retx": float(np.mean(a["retx_prb"])),
                "offered_mbps": float(np.mean(a["offered"])),
                "offered_slice_mbps": float(np.mean(a["offered_slice"])),
                # fraction of the tenant's OFFERED traffic actually delivered:
                # the natural service-level intent ("serve my traffic"),
                # independent of how loaded the tenant happens to be
                "served_ratio": float(min(1.5, np.mean(a["tput"])
                                          / max(np.mean(a["offered"]), 1e-6))),
                "offered_input_ratio": float(self.traffic.multiplier(tid)),
                "n_ue": float(st.n_ue),
                "active_sessions": float(self.traffic.active_sessions(tid)),
                "mean_dist_m": float(self.mobility.groups[tid].distance()
                                     .mean()),
                "edge_fraction": float(np.mean(
                    self.mobility.groups[tid].distance()
                    > float(self.ran.get("edge_radius_m", 250.0)))),
            }
            # xSlice-style QoS regret components
            p_dem = float(qos.get("throughput_mbps", 0.0))
            t_dem = float(qos.get("delay_ms", 0.0))
            z_dem = float(qos.get("bler", 0.0))
            kpm[tid]["regret_throughput"] = (
                max((p_dem - tput) / p_dem, 0.0) if p_dem > 0 else 0.0)
            kpm[tid]["regret_delay"] = (
                max((delay - t_dem) / t_dem, 0.0) if t_dem > 0 else 0.0)
            kpm[tid]["regret_bler"] = (
                max((bler - z_dem) / z_dem, 0.0) if z_dem > 0 else 0.0)

        used = float(np.mean(cell_hist["prb_used"])) \
            + float(np.mean(cell_hist["prb_retx"]))
        util = 100.0 * min(used / self.n_prb, 1.0)
        offered_ratios = [self.traffic.multiplier(t) for t in self.slices]
        prb_by_tenant = {t: kpm[t]["prb_alloc"] for t in self.slices}
        tput_by_ue = []
        for t in self.slices:
            tput_by_ue.extend([kpm[t]["throughput_mbps"]] *
                              self.slices[t].n_ue)

        kpm["_cell"] = {
            "txpower_dbm": float(self._controls.get("txpower", 46.0)),
            "tilt_deg": float(self._controls.get("tilt", 6.0)),
            "prb_util_pct": util,
            "offered_load_pct": 100.0 * float(np.mean(
                cell_hist["prb_demand"])) / self.n_prb,
            "offered_input_ratio": float(np.mean(offered_ratios))
            if offered_ratios else 1.0,
            "prb_used": used,
            "prb_retx": float(np.mean(cell_hist["prb_retx"])),
            "reconfig_prb": float(np.sum(cell_hist["reconfig_prb"])),
            "cell_budget_scale": float(np.mean(cell_hist["scale"])),
            "offered_mbps": float(np.mean(cell_hist["offered_mbps"])),
            "delivered_mbps": float(np.mean(cell_hist["delivered_mbps"])),
            "jain_prb": _jain(list(prb_by_tenant.values())),
            "jain_throughput": _jain(tput_by_ue),
            "mean_sinr_db": float(np.mean([kpm[t]["sinr_db"]
                                           for t in self.slices])),
            "mean_dist_m": float(np.mean([kpm[t]["mean_dist_m"]
                                          for t in self.slices])),
            "edge_fraction": float(np.mean([kpm[t]["edge_fraction"]
                                            for t in self.slices])),
            "n_ue_total": float(sum(s.n_ue for s in self.slices.values())),
            "spectral_efficiency": float(np.mean(
                [kpm[t]["spectral_efficiency"] for t in self.slices])),
            "slot": float(self.t),
        }
        kpm["_cell"]["total_regret"] = float(sum(
            kpm[t]["regret_throughput"] + kpm[t]["regret_delay"]
            + kpm[t]["regret_bler"] for t in self.slices))
        return kpm

    # ------------------------------------------------------------------
    def resource_pressure(self, tid: str,
                          kpm: Optional[Dict] = None) -> float:
        """rho_k = offered demand / effective deliverable capacity.

        The context variable the agentic regime head is trained to predict.
        Unlike raw offered load it already accounts for the CQI/SINR mix,
        which is precisely the information a load-only regime label throws
        away.  Computed here from simulator state so it is a LABEL, not a
        guess: the agent has to learn it from observable telemetry.
        """
        st = self.slices.get(tid)
        if st is None:
            return 1.0
        grp = self.mobility.groups[tid]
        phy = self.channel.evaluate(
            tid, grp.xy, grp.speed_mps(),
            tx_dbm=self.channel.serving.tx_dbm,
            tilt_deg=float(self._controls.get("tilt", 6.0)),
            cio_db=float(self._controls.get(f"cio_{tid}", 0.0)),
            subband_quality=1.0,
            mcs_cap=int(round(float(self._controls.get(
                f"mcs_{tid}", MCS_MAX_INDEX)))))
        quota = self.quota_of(tid)
        cap_mbps = float((quota / max(st.n_ue, 1)) * self.prb_hz
                         * np.mean(phy["se_mcs"]) * st.n_ue / 1e6)
        demand_mbps = float(self.traffic.offered_mbps(tid).sum())
        return float(demand_mbps / max(cap_mbps, 1e-6))

    def coverage_index(self, tid: str) -> float:
        """Fraction of a tenant's UEs whose bottleneck is SIGNAL, not PRBs.

        A UE is coverage limited when its spectral efficiency is low
        enough that one more PRB buys little and one more dB buys a lot.
        This is the hidden state variable the load label cannot see.
        """
        grp = self.mobility.groups.get(tid)
        if grp is None:
            return 0.0
        phy = self.channel.evaluate(
            tid, grp.xy, grp.speed_mps(),
            tx_dbm=self.channel.serving.tx_dbm,
            tilt_deg=float(self._controls.get("tilt", 6.0)))
        thresh = float(self.ran.get("coverage_sinr_db", 5.0))
        return float(np.mean(phy["sinr_db"] < thresh))

    # ------------------------------------------------------------------
    def snapshot_ue_features(self, tid: str) -> np.ndarray:
        """Per-UE feature matrix for the GCN encoder (xSlice-style).

        Columns (all normalised to roughly [0, 1] so a small network can
        learn from them):
            0 SINR/40 dB       1 CQI/15          2 MCS/28
            3 BLER/0.5         4 queue/maxbuf    5 offered/peak
            6 distance/rmax    7 speed/10 mps    8 LOS flag
            9 spectral efficiency / 6
        """
        grp = self.mobility.groups.get(tid)
        st = self.slices.get(tid)
        if grp is None or st is None:
            return np.zeros((0, 10))
        phy = self.channel.evaluate(
            tid, grp.xy, grp.speed_mps(),
            tx_dbm=self.channel.serving.tx_dbm,
            tilt_deg=float(self._controls.get("tilt", 6.0)),
            cio_db=float(self._controls.get(f"cio_{tid}", 0.0)),
            mcs_cap=int(round(float(self._controls.get(
                f"mcs_{tid}", MCS_MAX_INDEX)))))
        peak = max(float(np.max(self.traffic.base_mbps.get(
            tid, np.array([1.0])))) * 2.0, 1e-6)
        cols = [
            (phy["sinr_db"] + 10.0) / 40.0,
            phy["cqi"] / 15.0,
            phy["mcs"] / 28.0,
            phy["bler"] / 0.5,
            st.queue_bits / max(self.max_queue_bits, 1.0),
            self.traffic.offered_mbps(tid) / peak,
            grp.distance() / max(self.mobility.r_max, 1.0),
            grp.speed_mps() / 10.0,
            phy["los"],
            phy["se_mcs"] / 6.0,
        ]
        return np.clip(np.column_stack(cols), -2.0, 3.0)

    # ------------------------------------------------------------------
    def describe(self) -> Dict:
        return {
            "n_prb": self.n_prb,
            "prb_bandwidth_hz": self.prb_hz,
            "bandwidth_mhz": self.n_prb * self.prb_hz / 1e6,
            "slot_ms": self.slot_s * 1000.0,
            "subcarrier_spacing_khz": self.prb_hz / 12.0 / 1e3,
            "max_queue_kb": self.max_queue_bits / 8e3,
            "base_delay_ms": self.base_delay_ms,
            "actuation": {"time_constant_epochs": self.tau_epochs,
                          "reconfig_slots": self.reconfig_slots,
                          "reconfig_loss_frac": self.reconfig_loss},
            "coupling": {"txpower": self.k_txpower,
                         "subband": self.k_subband, "retx": self.k_retx},
            "channel": self.channel.describe(),
            "mobility": self.mobility.describe(),
            "traffic": self.traffic.describe(),
            "slices": {t: {"n_ue": s.n_ue} for t, s in self.slices.items()},
            "envelopes": self.ran.get("envelopes", {}),
            "initial_controls": self.ran.get("initial_controls", {}),
            "scheduler": "inter-slice quota + PF(alpha=schedpol) intra-slice",
        }

    def reconfig_cost_prb(self) -> float:
        return float(self._reconfig_cost_prb)


def _jain(xs: Sequence[float]) -> float:
    a = np.asarray([x for x in xs if np.isfinite(x)], dtype=float)
    if a.size == 0 or a.sum() == 0:
        return 1.0
    return float(a.sum() ** 2 / (a.size * (a ** 2).sum()))
