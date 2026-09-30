"""
intact_agentic/ran/traffic.py
=============================
Offered traffic.  Strictly EXOGENOUS: nothing in this file reads a control
value, a scheduler decision or a KPI.  That is not a stylistic choice --
if offered load depended on the controller, every method would face a
different workload and no comparison between methods would be valid.  The
predecessor codebase lost several weeks to exactly that bug.

MODELS
------
``cbr``     constant bit rate.  Deterministic, useful for debugging and
            for the calibration sweep.
``poisson`` packets arrive as a Poisson process at the configured mean
            rate; byte volume per slot is therefore over-dispersed
            relative to CBR, which is what makes queues form.
``onoff``   exponential on/off session model (the usual web/video
            abstraction).  Gives burstiness at the session timescale and
            genuine idle periods, so "offered load" and "active sessions"
            are different quantities, as they are in a real cell.

PER-TENANT LOAD PROFILE
-----------------------
On top of the per-UE model, each tenant carries a multiplicative profile
``levels`` held for ``hold_slots`` with an optional linear ``ramp_slots``
transition and an optional per-tenant phase offset.  Phase offsets are
what make the cell average USELESS as a regime indicator: T1 can be at
1.35 while T2 is at 0.70 and the cell average reads 1.02.  Reproducing
that disagreement is a precondition for the paper's per-tenant-regime
claim, so :func:`regime_disagreement_rate` measures it directly and the
scenario gate refuses a scenario in which it is too small.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class TrafficProfile:
    levels: Sequence[float] = (1.0,)
    hold_slots: int = 800
    ramp_slots: int = 0
    phase: int = 0

    def multiplier(self, t: int) -> float:
        lv = list(self.levels) or [1.0]
        hold = max(int(self.hold_slots), 1)
        absolute = max(int(t), 0)
        idx = ((absolute // hold) + int(self.phase)) % len(lv)
        value = float(lv[idx])
        ramp = min(max(int(self.ramp_slots), 0), hold)
        pos = absolute % hold
        if ramp > 0 and pos >= hold - ramp:
            nxt = float(lv[(idx + 1) % len(lv)])
            frac = (pos - (hold - ramp) + 1) / float(ramp)
            value = (1.0 - frac) * value + frac * nxt
        return max(value, 0.0)


class TrafficModel:
    """Per-UE arrival generator plus the per-tenant exogenous profile."""

    def __init__(self, cfg: Dict, rng: np.random.Generator):
        self.rng = rng
        ran = cfg["ran"]
        self.slot_s = float(ran["slot_ms"]) / 1000.0
        tr = ran.get("traffic", {}) or {}
        self.kind = str(tr.get("model", "onoff")).lower()
        self.packet_bits = float(tr.get("packet_bytes", 1200)) * 8.0
        self.on_mean_slots = float(tr.get("on_mean_slots", 24.0))
        self.off_mean_slots = float(tr.get("off_mean_slots", 12.0))
        self.burstiness = float(tr.get("burstiness", 1.0))

        self.base_mbps: Dict[str, np.ndarray] = {}
        self.state_on: Dict[str, np.ndarray] = {}
        self.timer: Dict[str, np.ndarray] = {}
        self.profiles: Dict[str, TrafficProfile] = {}
        self.qos: Dict[str, Dict[str, float]] = {}

        default = ran.get("load_profile", {}) or {}
        for tid, sl in (ran.get("slices") or {}).items():
            p = (sl.get("load_profile") or default)
            self.profiles[tid] = TrafficProfile(
                levels=tuple(p.get("levels", (1.0,))),
                hold_slots=int(p.get("hold_slots", 800)),
                ramp_slots=int(p.get("ramp_slots", 0)),
                phase=int(sl.get("load_phase", p.get("phase", 0))))
            self.qos[tid] = {
                "throughput_mbps": float(sl.get("qos_throughput_mbps", 0.0)),
                "delay_ms": float(sl.get("qos_delay_ms", 0.0)),
                "bler": float(sl.get("qos_bler", 0.1)),
            }
        self.t = 0

    # ------------------------------------------------------------------
    def spawn(self, tid: str, n_ue: int, load_mbps_per_ue: float,
              spread: float = 0.25) -> None:
        """Give each UE its own baseline rate, log-normally spread."""
        mult = np.exp(self.rng.normal(0.0, spread, n_ue)
                      - 0.5 * spread ** 2)
        self.base_mbps[tid] = float(load_mbps_per_ue) * mult
        on0 = self.on_mean_slots / max(self.on_mean_slots
                                       + self.off_mean_slots, 1e-9)
        self.state_on[tid] = self.rng.random(n_ue) < on0
        self.timer[tid] = self.rng.exponential(
            np.where(self.state_on[tid], self.on_mean_slots,
                     self.off_mean_slots))
        if tid not in self.profiles:
            self.profiles[tid] = TrafficProfile()
        if tid not in self.qos:
            self.qos[tid] = {"throughput_mbps": 0.0, "delay_ms": 0.0,
                             "bler": 0.1}

    def remove(self, tid: str) -> None:
        for d in (self.base_mbps, self.state_on, self.timer):
            d.pop(tid, None)

    # ------------------------------------------------------------------
    def multiplier(self, tid: str, t: Optional[int] = None) -> float:
        prof = self.profiles.get(tid)
        if prof is None:
            return 1.0
        return prof.multiplier(self.t if t is None else t)

    def forecast_multiplier(self, tid: str, horizon_slots: int) -> float:
        return self.multiplier(tid, self.t + int(horizon_slots))

    # ------------------------------------------------------------------
    def arrivals_bits(self, tid: str) -> np.ndarray:
        """Bits arriving for each UE of one tenant during the current slot."""
        base = self.base_mbps.get(tid)
        if base is None:
            return np.zeros(0)
        mult = self.multiplier(tid)
        mean_bits = base * mult * 1e6 * self.slot_s
        if self.kind == "cbr":
            return mean_bits
        if self.kind == "poisson":
            lam = np.maximum(mean_bits / max(self.packet_bits, 1.0), 0.0)
            return self.rng.poisson(lam) * self.packet_bits
        # on/off: renew the session state, then emit at the peak rate
        self.timer[tid] = self.timer[tid] - 1.0
        flip = self.timer[tid] <= 0
        if flip.any():
            self.state_on[tid][flip] = ~self.state_on[tid][flip]
            means = np.where(self.state_on[tid][flip], self.on_mean_slots,
                             self.off_mean_slots)
            self.timer[tid][flip] = self.rng.exponential(means)
        duty = self.on_mean_slots / max(self.on_mean_slots
                                        + self.off_mean_slots, 1e-9)
        peak_bits = mean_bits / max(duty, 1e-6)
        lam = np.where(self.state_on[tid],
                       peak_bits / max(self.packet_bits, 1.0), 0.0)
        lam = np.maximum(lam * self.burstiness, 0.0)
        return self.rng.poisson(lam) * self.packet_bits / max(
            self.burstiness, 1e-9)

    def offered_mbps(self, tid: str) -> np.ndarray:
        """Mean (not realised) offered rate per UE, for reporting."""
        base = self.base_mbps.get(tid)
        if base is None:
            return np.zeros(0)
        return base * self.multiplier(tid)

    def active_sessions(self, tid: str) -> int:
        st = self.state_on.get(tid)
        return int(st.sum()) if st is not None else 0

    def advance(self, n_slots: int = 1) -> None:
        self.t += int(n_slots)

    # ------------------------------------------------------------------
    def describe(self) -> Dict:
        return {
            "model": self.kind,
            "packet_bytes": self.packet_bits / 8.0,
            "on_mean_slots": self.on_mean_slots,
            "off_mean_slots": self.off_mean_slots,
            "profiles": {k: {"levels": list(v.levels),
                             "hold_slots": v.hold_slots,
                             "ramp_slots": v.ramp_slots, "phase": v.phase}
                         for k, v in self.profiles.items()},
            "qos": self.qos,
        }


# ---------------------------------------------------------------------------
def regime_disagreement_rate(profiles: Dict[str, TrafficProfile],
                             n_slots: int, edges=(0.90, 1.25)) -> float:
    """Fraction of (slot, tenant) pairs where the tenant's load band differs
    from the cell-average band.

    This is the quantity INTACT-RA's per-tenant regime resolution exists to
    exploit.  If it is near zero, a cell-average arbiter is just as good and
    the scenario cannot discriminate the two -- so the scenario gate
    (``gates.py``, gate G4) refuses it.
    """
    if not profiles:
        return 0.0
    def band(x):
        return 0 if x < edges[0] else (1 if x < edges[1] else 2)
    dis = 0
    tot = 0
    step = max(int(n_slots) // 400, 1)
    for t in range(0, int(n_slots), step):
        vals = {k: v.multiplier(t) for k, v in profiles.items()}
        cell = float(np.mean(list(vals.values())))
        cb = band(cell)
        for v in vals.values():
            tot += 1
            if band(v) != cb:
                dis += 1
    return dis / max(tot, 1)
