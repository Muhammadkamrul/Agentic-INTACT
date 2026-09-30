"""
intact_agentic/xapps.py
=======================
The xApps.  BLACK BOXES from the arbiter's point of view: we see what they
ask for, never how they decided.  They do not coordinate, do not know
about each other, are never retrained, and are never told that an
arbiter exists.  That is the premise of the whole problem -- if we could
change the xApps there would be nothing to mediate.

Each xApp is a proportional controller toward a private objective with a
private trigger.  The trigger matters: an xApp that only writes when its
buffer crosses a threshold cannot conflict with anyone while the buffer
is low, so the conflict rate is itself a state-dependent quantity rather
than a constant.

THE SIX KINDS
-------------
``throughput``   wants more resource for its own tenant when its
                 throughput is below target.
``latency``      raises its slice's scheduler weight when delay is high.
``robustness``   lowers the MCS ceiling when the buffer grows, because
                 retransmissions are eating the slice.
``energy``       HOST-owned, CELL-SCOPED.  Pushes transmit power or tilt
                 toward an energy setpoint that ALTERNATES on a duty
                 cycle, so it never converges and never goes quiet.  This
                 is the xApp that damages every tenant at once.
``coverage``     HOST-owned, CELL-SCOPED.  Pushes transmit power UP when
                 the cell-edge population is suffering.  It writes the
                 SAME knob as the energy xApp and wants the opposite
                 thing: this is the direct C1 conflict, and it is also
                 the pair whose correct resolution depends on a
                 sensitivity slope that drifts.
``steering``     adjusts a cell individual offset to move load between
                 slices.

DUTY CYCLE AND CLOCK DISCIPLINE
-------------------------------
An xApp's internal clock is set from the EPOCH NUMBER by the experiment,
never incremented inside ``propose``.  If it advanced on each poll, a
method that polls every claim would run one tick ahead of a method that
polls only the selected ones, and the two would face different workloads.
That bug cost the predecessor project several invalid comparisons, so the
clock is set externally and ``scripts/selftest.py`` asserts that every
method sees an identical proposal stream when the controls are identical.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


class XApp:
    kind = "base"

    def __init__(self, name: str, tenant: str, param: str,
                 domain, step: float, gain: float, cfg: Dict, **kw):
        self.name = name
        self.tenant = tenant
        self.param = param
        self.lo, self.hi = float(domain[0]), float(domain[1])
        self.step = float(step)
        self.gain = float(gain)
        self.cfg = cfg
        self.t = 0
        self.extra = kw

    def set_clock(self, epoch: int) -> None:
        self.t = int(epoch)

    def quantise(self, v: float) -> float:
        v = min(max(float(v), self.lo), self.hi)
        return round(round((v - self.lo) / self.step) * self.step + self.lo, 9)

    def propose(self, kpm: Dict, controls: Dict[str, float]
                ) -> Optional[float]:
        raise NotImplementedError


class ThroughputXApp(XApp):
    kind = "throughput"

    def propose(self, kpm, controls):
        row = kpm.get(self.tenant)
        if not row:
            return None
        err = float(self.extra["target"]) - float(
            row[self.extra.get("kpi", "throughput_mbps")])
        if abs(err) < float(self.extra.get("deadband", 0.12)):
            return None
        cur = float(controls.get(self.param, 0.5 * (self.lo + self.hi)))
        return self.quantise(cur + self.gain * err)


class LatencyXApp(XApp):
    kind = "latency"

    def propose(self, kpm, controls):
        row = kpm.get(self.tenant)
        if not row:
            return None
        err = float(row[self.extra.get("kpi", "delay_ms")]) - float(
            self.extra["target"])
        if abs(err) < float(self.extra.get("deadband", 1.5)):
            return None
        cur = float(controls.get(self.param, 1.0))
        return self.quantise(cur + self.gain * err)


class RobustnessXApp(XApp):
    kind = "robustness"

    def propose(self, kpm, controls):
        row = kpm.get(self.tenant)
        if not row:
            return None
        buf = float(row["buffer_kb"])
        thr = float(self.extra["target"])
        cur = float(controls.get(self.param, 20.0))
        if buf >= thr:
            return self.quantise(cur - self.gain * (buf - thr))
        # release the ceiling again once the buffer has drained, otherwise
        # the knob ratchets down and stops being an interesting claim
        if buf < 0.5 * thr and cur < self.hi:
            return self.quantise(cur + self.gain * 0.5 * (thr - buf))
        return None


class EnergyXApp(XApp):
    """Host energy saver.  Duty-cycled setpoint, cell scope."""
    kind = "energy"

    def propose(self, kpm, controls):
        cur = float(controls.get(self.param, 0.5 * (self.lo + self.hi)))
        tgt = float(self.extra["target"])
        alt = self.extra.get("target_alt")
        period = int(self.extra.get("period", 0))
        if period > 0 and alt is not None and (self.t // period) % 2 == 1:
            tgt = float(alt)
        err = cur - tgt
        if abs(err) < max(self.step, 1e-9):
            return None
        frac = min(max(self.gain, 0.05), 1.0)
        return self.quantise(cur - frac * err)


class CoverageXApp(XApp):
    """Host coverage keeper.  Writes the same cell knob as the energy xApp
    and wants the opposite thing when the edge population suffers."""
    kind = "coverage"

    def propose(self, kpm, controls):
        cell = kpm.get("_cell", {})
        edge = float(cell.get("edge_fraction", 0.0))
        sinr = float(cell.get("mean_sinr_db", 10.0))
        trigger = float(self.extra.get("edge_trigger", 0.25))
        sinr_target = float(self.extra.get("sinr_target", 8.0))
        cur = float(controls.get(self.param, 0.5 * (self.lo + self.hi)))
        if edge < trigger and sinr > sinr_target:
            # comfortable: hand power back, slowly
            if cur > self.lo + self.step:
                return self.quantise(cur - self.step)
            return None
        deficit = max(sinr_target - sinr, 0.0) + 4.0 * max(edge - trigger,
                                                           0.0)
        if deficit < 0.15:
            return None
        return self.quantise(cur + self.gain * deficit)


class SteeringXApp(XApp):
    """Adjusts a cell individual offset to move load between slices."""
    kind = "steering"

    def propose(self, kpm, controls):
        row = kpm.get(self.tenant)
        if not row:
            return None
        util = float(kpm.get("_cell", {}).get("prb_util_pct", 50.0))
        target = float(self.extra.get("target", 75.0))
        err = util - target
        if abs(err) < float(self.extra.get("deadband", 6.0)):
            return None
        cur = float(controls.get(self.param, 0.0))
        return self.quantise(cur - self.gain * err * 0.01)


_KINDS = {"throughput": ThroughputXApp, "latency": LatencyXApp,
          "robustness": RobustnessXApp, "energy": EnergyXApp,
          "coverage": CoverageXApp, "steering": SteeringXApp}


def build_xapps(cfg: Dict) -> Dict[str, XApp]:
    out: Dict[str, XApp] = {}
    for x in cfg["xapps"]:
        kind = str(x["kind"])
        cls = _KINDS.get(kind)
        if cls is None:
            raise ValueError(f"unknown xApp kind {kind!r}")
        extra = {k: v for k, v in x.items()
                 if k not in ("name", "tenant", "param", "domain", "step",
                              "gain", "kind")}
        out[x["name"]] = cls(name=x["name"], tenant=x["tenant"],
                             param=x["param"], domain=x["domain"],
                             step=float(x["step"]), gain=float(x["gain"]),
                             cfg=cfg, **extra)
    return out


def collect_proposals(xapps: Dict[str, XApp], claims: Dict, kpm: Dict,
                      controls: Dict[str, float], epoch: int
                      ) -> Dict[str, float]:
    """One proposal per active claim, cached so every method sees the same
    stream.

    A stateful controller polled twice in one epoch can return two
    different values.  The arbiter must score the value it will execute,
    so the proposal is taken ONCE and reused.
    """
    for x in xapps.values():
        x.set_clock(epoch)
    cache: Dict[str, Optional[float]] = {}
    out: Dict[str, float] = {}
    for jid in sorted(claims):
        c = claims[jid]
        key = (c.xapp, c.param)
        if key not in cache:
            x = xapps.get(c.xapp)
            cache[key] = None if x is None else x.propose(kpm, controls)
        v = cache[key]
        if v is None:
            continue
        cur = float(controls.get(c.param, v))
        if abs(v - cur) < 1e-12:
            continue                      # nothing to ask for
        out[jid] = float(v)
    return out
