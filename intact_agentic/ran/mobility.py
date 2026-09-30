"""
intact_agentic/ran/mobility.py
==============================
UE mobility.  Positions in metres, origin at the serving gNB.

FOUR MODELS, EACH WITH A DIFFERENT JOB
--------------------------------------
``static``        UEs never move.  Used for the calibration sweep so the
                  offline sensitivity table is measured in a frozen world,
                  exactly as the INTACT-RA design assumes.

``rwp``           Random waypoint inside an annulus.  Ordinary churn.
                  Individual UEs move, but the *distribution* of distances
                  is stationary -- so the sensitivity table stays valid.
                  This is the control condition.

``group_drift``   A named subset of tenants has its centroid pulled
                  radially outward (or inward) at a constant rate over a
                  declared window, with per-UE random-walk noise on top.
                  Offered load is untouched.  THIS IS THE SCENARIO THE
                  WHOLE PAPER IS ABOUT: the load-regime label never
                  changes, the distance distribution changes a lot, and
                  therefore the true value of transmit power rises while
                  the true value of a PRB falls.

``diurnal``       Sinusoidal excursion of the centroid with a configurable
                  period.  A recurring, forecastable version of the same
                  thing; useful for asking whether an online learner
                  merely tracks or actually anticipates.

All models write into the SAME per-UE ``(x, y, vx, vy)`` arrays, so the
telemetry schema does not depend on which one is active.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class UEGroup:
    """The mobility state of one tenant's UE population."""
    tid: str
    xy: np.ndarray                     # (n, 2) metres
    v: np.ndarray                      # (n, 2) m/s
    target: np.ndarray                 # (n, 2) waypoint for rwp
    speed: np.ndarray                  # (n,) m/s scalar speed
    xy0: np.ndarray = field(default=None)   # positions at t = 0
    ref: np.ndarray = field(default=None)   # drifting reference centre

    def __post_init__(self):
        if self.xy0 is None:
            self.xy0 = self.xy.copy()
        if self.ref is None:
            self.ref = self.xy.copy()

    @property
    def n(self) -> int:
        return self.xy.shape[0]

    def distance(self) -> np.ndarray:
        return np.hypot(self.xy[:, 0], self.xy[:, 1])

    def speed_mps(self) -> np.ndarray:
        return np.hypot(self.v[:, 0], self.v[:, 1])


class MobilityModel:
    """Advances every UE by one slot and reports where they are."""

    def __init__(self, cfg: Dict, rng: np.random.Generator):
        m = (cfg["ran"].get("mobility", {}) or {})
        self.cfg = m
        self.rng = rng
        self.slot_s = float(cfg["ran"]["slot_ms"]) / 1000.0
        self.mode = str(m.get("mode", "rwp")).lower()
        self.r_min = float(m.get("min_radius_m", 25.0))
        self.r_max = float(m.get("max_radius_m", 420.0))
        self.speed_lo = float(m.get("speed_mps_min", 0.5))
        self.speed_hi = float(m.get("speed_mps_max", 3.0))
        self.pause_prob = float(m.get("pause_prob", 0.02))
        self.wander_radius = float(m.get("wander_radius_m", 35.0))
        # group drift
        self.drift = m.get("drift", {}) or {}
        self.drift_tenants: List[str] = list(self.drift.get("tenants", []))
        self.drift_rate = float(self.drift.get("rate_m_per_slot", 0.0))
        self.drift_start = int(self.drift.get("start_slot", 0))
        self.drift_end = self.drift.get("end_slot")
        self.drift_jitter = float(self.drift.get("jitter_m_per_slot", 0.0))
        # diurnal
        self.diurnal = m.get("diurnal", {}) or {}
        self.groups: Dict[str, UEGroup] = {}
        # tid -> [r_lo, r_hi] of the annulus this tenant's waypoints live in
        self.annulus: Dict[str, List[float]] = {}
        self.t = 0

    # ------------------------------------------------------------------
    def spawn(self, tid: str, n_ue: int,
              placement: Optional[Dict] = None) -> UEGroup:
        """Create one tenant's UE population.

        ``placement`` lets a scenario start a tenant close in or far out,
        which is how a "capacity-limited tenant" and a "coverage-limited
        tenant" are declared without touching the physics.
        """
        p = placement or {}
        r_lo = float(p.get("radius_min_m", self.r_min))
        r_hi = float(p.get("radius_max_m", self.r_max))
        ang_lo = math.radians(float(p.get("angle_min_deg", 0.0)))
        ang_hi = math.radians(float(p.get("angle_max_deg", 360.0)))
        # uniform in AREA, not in radius, so the population is not
        # artificially concentrated near the gNB
        u = self.rng.random(n_ue)
        r = np.sqrt(u * (r_hi ** 2 - r_lo ** 2) + r_lo ** 2)
        a = self.rng.uniform(ang_lo, ang_hi, n_ue)
        xy = np.column_stack([r * np.cos(a), r * np.sin(a)])
        speed = self.rng.uniform(float(p.get("speed_mps_min", self.speed_lo)),
                                 float(p.get("speed_mps_max", self.speed_hi)),
                                 n_ue)
        tgt = self._draw_waypoints(n_ue, r_lo, r_hi)
        v = np.zeros((n_ue, 2))
        g = UEGroup(tid=tid, xy=xy, v=v, target=tgt, speed=speed)
        self.groups[tid] = g
        # A tenant's placement is a PERSISTENT property of that tenant's
        # population, not merely an initial condition.  Without this the
        # waypoint process diffuses every tenant toward the same cell-wide
        # uniform distribution within a few thousand slots, which silently
        # destroys the declared coverage-limited / capacity-limited split
        # and fights any group drift applied on top of it.
        self.annulus[tid] = [r_lo, r_hi]
        return g

    def _draw_waypoints(self, n: int, r_lo: float, r_hi: float) -> np.ndarray:
        u = self.rng.random(n)
        r = np.sqrt(u * (r_hi ** 2 - r_lo ** 2) + r_lo ** 2)
        a = self.rng.uniform(0, 2 * np.pi, n)
        return np.column_stack([r * np.cos(a), r * np.sin(a)])

    # ------------------------------------------------------------------
    def step(self, n_slots: int = 1) -> None:
        for _ in range(int(n_slots)):
            self.t += 1
            for tid, g in self.groups.items():
                if self.mode == "static":
                    g.v[:] = 0.0
                    continue
                if self.mode == "group_drift" and tid in self.drift_tenants:
                    # Coherent group migration: the tenant's reference
                    # position moves radially outward and each UE wanders
                    # locally around it.  Modelling this as "a small radial
                    # nudge added to a cell-wide waypoint process" does not
                    # work: the waypoint process is a random walk of the
                    # same per-slot magnitude, so it swamps the drift and
                    # the group never actually migrates.
                    self._drift_step(tid, g)
                    self._wander_step(g)
                else:
                    self._rwp_step(g, tid)
                    self._diurnal_step(tid, g)
                self._clamp(g)

    def _rwp_step(self, g: UEGroup, tid: Optional[str] = None) -> None:
        lo, hi = self.annulus.get(tid or g.tid, [self.r_min, self.r_max])
        d = g.target - g.xy
        dist = np.hypot(d[:, 0], d[:, 1])
        reached = dist < 5.0
        if reached.any():
            newt = self._draw_waypoints(int(reached.sum()), lo, hi)
            g.target[reached] = newt
            d = g.target - g.xy
            dist = np.hypot(d[:, 0], d[:, 1])
        paused = self.rng.random(g.n) < self.pause_prob
        unit = d / np.maximum(dist, 1e-9)[:, None]
        step = (g.speed * self.slot_s)[:, None] * unit
        step[paused] = 0.0
        g.xy += step
        g.v = step / max(self.slot_s, 1e-9)

    def _drift_step(self, tid: str, g: UEGroup) -> None:
        if not self.drift_tenants or tid not in self.drift_tenants:
            return
        if self.t < self.drift_start:
            return
        if self.drift_end is not None and self.t > int(self.drift_end):
            return
        r = np.hypot(g.xy[:, 0], g.xy[:, 1])
        unit = g.xy / np.maximum(r, 1e-9)[:, None]
        push = self.drift_rate
        if self.drift_jitter > 0:
            push = push + self.rng.normal(0.0, self.drift_jitter, g.n)
            push = push[:, None]
        g.xy += unit * push
        # the REFERENCE point moves with the group, so local wander is
        # around the migrating centre rather than the spawn location
        rr = np.hypot(g.ref[:, 0], g.ref[:, 1])
        g.ref += (g.ref / np.maximum(rr, 1e-9)[:, None]) * push
        ann = self.annulus.get(tid)
        if ann is not None:
            ann[0] = min(ann[0] + self.drift_rate, self.r_max - 10.0)
            ann[1] = min(ann[1] + self.drift_rate, self.r_max)

    def _wander_step(self, g: UEGroup) -> None:
        """Local mobility around each UE's own drifting reference point."""
        d = g.ref - g.xy
        dist = np.hypot(d[:, 0], d[:, 1])
        far = dist > self.wander_radius
        unit = d / np.maximum(dist, 1e-9)[:, None]
        step_len = (g.speed * self.slot_s)[:, None]
        # head back when too far from the reference, otherwise wander
        ang = self.rng.uniform(0, 2 * np.pi, g.n)
        rnd = np.column_stack([np.cos(ang), np.sin(ang)])
        move = np.where(far[:, None], unit, rnd) * step_len
        paused = self.rng.random(g.n) < self.pause_prob
        move[paused] = 0.0
        g.xy += move
        g.v = move / max(self.slot_s, 1e-9)

    def _diurnal_step(self, tid: str, g: UEGroup) -> None:
        d = self.diurnal
        if not d or tid not in (d.get("tenants") or []):
            return
        amp = float(d.get("amplitude_m", 0.0))
        period = max(float(d.get("period_slots", 20000.0)), 1.0)
        if amp <= 0:
            return
        phase = 2 * np.pi * self.t / period
        prev_phase = 2 * np.pi * (self.t - 1) / period
        delta = amp * (math.sin(phase) - math.sin(prev_phase))
        r = np.hypot(g.xy0[:, 0], g.xy0[:, 1])
        unit = g.xy0 / np.maximum(r, 1e-9)[:, None]
        g.xy += unit * delta

    def _clamp(self, g: UEGroup) -> None:
        r = np.hypot(g.xy[:, 0], g.xy[:, 1])
        too_far = r > self.r_max
        too_near = r < self.r_min
        if too_far.any():
            g.xy[too_far] *= (self.r_max / r[too_far])[:, None]
        if too_near.any():
            scale = np.where(r[too_near] > 1e-6,
                             self.r_min / np.maximum(r[too_near], 1e-6), 1.0)
            g.xy[too_near] *= scale[:, None]

    # ------------------------------------------------------------------
    def stats(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        shifts = []
        for tid, g in self.groups.items():
            d = g.distance()
            out[f"mean_dist_m_{tid}"] = float(d.mean())
            out[f"p90_dist_m_{tid}"] = float(np.percentile(d, 90))
            shift = float(d.mean() - np.hypot(g.xy0[:, 0], g.xy0[:, 1]).mean())
            out[f"dist_shift_m_{tid}"] = shift
            out[f"mean_speed_mps_{tid}"] = float(g.speed_mps().mean())
            shifts.append(shift)
        out["mean_dist_shift_m"] = float(np.mean(shifts)) if shifts else 0.0
        return out

    def describe(self) -> Dict:
        return {"mode": self.mode, "min_radius_m": self.r_min,
                "max_radius_m": self.r_max,
                "speed_mps_range": [self.speed_lo, self.speed_hi],
                "pause_prob": self.pause_prob,
                "drift": self.drift, "diurnal": self.diurnal}
