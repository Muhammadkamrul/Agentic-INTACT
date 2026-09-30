"""
intact_agentic/arbiter/regime.py
================================
Which page of the sensitivity table do we read?

A slope is only meaningful inside an operating regime; the same knob and
the same intent can have slopes an order of magnitude apart between a
lightly loaded cell and a congested one, or between a cell whose UEs are
near the gNB and one whose UEs are at the edge.  The regime label is the
INDEX into the table, so choosing it badly is equivalent to reading the
wrong slope.

THREE ESTIMATORS, WHICH ARE THREE POINTS ON THE PAPER'S ARGUMENT
-----------------------------------------------------------------
``CellRegime``      one label for the whole cell, from cell-average
                    offered load.  What every prior arbiter does.  Fails
                    whenever tenants are busy at different times: T1 at
                    1.35 and T2 at 0.70 average to 1.02 and the arbiter
                    reads the "nominal" page for BOTH.

``TenantLoadRegime`` one label per tenant, from that tenant's own measured
                    offered load.  THIS IS FROZEN INTACT-RA.  It fixes the
                    averaging problem, and it is what the existing results
                    are built on.  Its remaining blind spot is the whole
                    point of this paper: load is not the only thing that
                    decides a slope.  If UEs drift to the cell edge at
                    constant load, this label does not move at all.

``LearnedRegime``   one label per tenant from a LEARNED context vector
                    that includes resource pressure (demand over
                    deliverable capacity, so CQI enters), the CQI/SINR
                    distribution, the edge-user fraction, mobility, the
                    traffic mix and interference.  Implemented by the GCN
                    context head in ``agent/context_head.py``; this file
                    only holds the discretisation and the fallback.

REGIME SPACE
------------
Two axes, because the two things that matter are independent:

    load axis      light | nominal | heavy      from resource pressure
    channel axis   good  | poor                 from the coverage index

giving six labels ``L0C0 .. L2C1``.  ``CellRegime`` and
``TenantLoadRegime`` only ever emit the ``C0`` column, which is exactly
their limitation made explicit: they cannot express "heavy AND coverage
limited", so a table indexed by them has nowhere to put that state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

LOAD_NAMES = ("L0", "L1", "L2")
CHAN_NAMES = ("C0", "C1")


def all_regimes(channel_aware: bool = True) -> List[str]:
    if not channel_aware:
        return [f"{l}C0" for l in LOAD_NAMES]
    return [f"{l}{c}" for l in LOAD_NAMES for c in CHAN_NAMES]


def band(value: float, edges: Sequence[float]) -> int:
    for k, e in enumerate(edges):
        if value < float(e):
            return k
    return len(edges)


def make_label(load_idx: int, chan_idx: int = 0) -> str:
    return f"{LOAD_NAMES[int(np.clip(load_idx, 0, 2))]}" \
           f"{CHAN_NAMES[int(np.clip(chan_idx, 0, 1))]}"


# ---------------------------------------------------------------------------
@dataclass
class RegimeReport:
    """Everything the arbiter and the logs need about the current context."""
    per_tenant: Dict[str, str]
    per_intent: Dict[str, str]
    cell: str
    features: Dict[str, Dict[str, float]]
    confidence: Dict[str, float]
    source: str

    def of_intent(self, iid: str, default: str = "L1C0") -> str:
        return self.per_intent.get(iid, default)


class RegimeEstimator:
    """Base class: build a RegimeReport from a KPM report and the RAN."""

    source = "base"

    def __init__(self, cfg: Dict):
        a = cfg.get("arbiter", {}) or {}
        self.load_edges = tuple(a.get("regime_load_edges", (0.90, 1.25)))
        # resource pressure rho = demand / effective capacity lives on a
        # different scale from the offered-load ratio, so it gets its own
        # band edges, chosen by scripts/calibrate.py to line up with the
        # load scales the offline sweep used for its three table columns.
        self.pressure_edges = tuple(a.get("regime_pressure_edges",
                                          (0.45, 0.62)))
        self.cov_edge = float(a.get("regime_coverage_edge", 0.35))
        self.channel_aware = bool(a.get("channel_aware_regime", False))

    def _features(self, ran, kpm, tenants) -> Dict[str, Dict[str, float]]:
        out = {}
        for t in tenants:
            row = kpm.get(t, {})
            out[t] = {
                "offered_input_ratio": float(row.get("offered_input_ratio",
                                                     1.0)),
                "resource_pressure": float(ran.resource_pressure(t, kpm)),
                "coverage_index": float(ran.coverage_index(t)),
                "sinr_db": float(row.get("sinr_db", 0.0)),
                "cqi": float(row.get("cqi", 7.0)),
                "edge_fraction": float(row.get("edge_fraction", 0.0)),
                "mean_dist_m": float(row.get("mean_dist_m", 0.0)),
                "buffer_kb": float(row.get("buffer_kb", 0.0)),
                "prb_alloc": float(row.get("prb_alloc", 0.0)),
            }
        return out

    def estimate(self, ran, kpm, tenants, intents) -> RegimeReport:
        raise NotImplementedError


# ---------------------------------------------------------------------------
class CellRegime(RegimeEstimator):
    """One cell-wide label from the cell-average offered load."""
    source = "cell"

    def estimate(self, ran, kpm, tenants, intents) -> RegimeReport:
        feats = self._features(ran, kpm, tenants)
        cell_load = float(kpm.get("_cell", {}).get("offered_input_ratio", 1.0))
        lab = make_label(band(cell_load, self.load_edges), 0)
        per_t = {t: lab for t in tenants}
        per_i = {iid: lab for iid in intents}
        return RegimeReport(per_t, per_i, lab, feats,
                            {t: 1.0 for t in tenants}, self.source)


class TenantLoadRegime(RegimeEstimator):
    """Per-tenant label from that tenant's own measured offered load.

    This is FROZEN INTACT-RA.  It is a genuine improvement over the cell
    average -- and it is deliberately blind to everything except load, so
    the paper can measure exactly what that blindness costs.
    """
    source = "tenant_load"

    def estimate(self, ran, kpm, tenants, intents) -> RegimeReport:
        feats = self._features(ran, kpm, tenants)
        per_t = {}
        for t in tenants:
            per_t[t] = make_label(
                band(feats[t]["offered_input_ratio"], self.load_edges), 0)
        cell_load = float(kpm.get("_cell", {}).get("offered_input_ratio", 1.0))
        cell = make_label(band(cell_load, self.load_edges), 0)
        per_i = {}
        for iid, it in intents.items():
            per_i[iid] = per_t.get(it.tenant, cell)
        return RegimeReport(per_t, per_i, cell, feats,
                            {t: 1.0 for t in tenants}, self.source)


class OracleRegime(RegimeEstimator):
    """Ground-truth two-axis label straight from simulator state.

    Never deployable -- it reads quantities (true resource pressure, true
    coverage index) that a real RIC would have to estimate.  It exists as
    an UPPER BOUND so the learned head can be scored against the best any
    context estimator could do, which is the only honest way to say how
    much of the remaining gap is estimation error.
    """
    source = "oracle"

    def estimate(self, ran, kpm, tenants, intents) -> RegimeReport:
        feats = self._features(ran, kpm, tenants)
        per_t = {}
        for t in tenants:
            li = band(feats[t]["resource_pressure"], self.pressure_edges)
            ci = 1 if feats[t]["coverage_index"] > self.cov_edge else 0
            per_t[t] = make_label(li, ci if self.channel_aware else 0)
        cell_rho = float(np.mean([feats[t]["resource_pressure"]
                                  for t in tenants])) if tenants else 1.0
        cell = make_label(band(cell_rho, self.pressure_edges), 0)
        per_i = {iid: per_t.get(it.tenant, cell)
                 for iid, it in intents.items()}
        return RegimeReport(per_t, per_i, cell, feats,
                            {t: 1.0 for t in tenants}, self.source)


class LearnedRegime(RegimeEstimator):
    """Per-tenant label from the learned GCN context head.

    The head predicts two scalars per tenant -- resource pressure rho and
    coverage index kappa -- from OBSERVABLE telemetry only (per-UE SINR,
    CQI, MCS, BLER, queue, offered rate, distance, speed, LOS).  It is
    trained online against the simulator's own rho and kappa with a
    one-epoch lag, which is a supervised signal a real deployment also
    has: both labels are computable after the fact from KPM and MAC data.

    ``confidence`` is 1 - normalised prediction error over a rolling
    window.  A low-confidence tenant is reported as such, and the
    arbiter's robust bound widens accordingly rather than the head being
    allowed to invent a regime it cannot see.
    """
    source = "learned"

    def __init__(self, cfg: Dict, head):
        super().__init__(cfg)
        self.head = head
        self.channel_aware = True

    def estimate(self, ran, kpm, tenants, intents) -> RegimeReport:
        feats = self._features(ran, kpm, tenants)
        pred = self.head.predict(ran, kpm, tenants)
        per_t, conf = {}, {}
        for t in tenants:
            rho, kappa, c = pred.get(t, (feats[t]["offered_input_ratio"],
                                         feats[t]["coverage_index"], 0.0))
            feats[t]["rho_hat"] = float(rho)
            feats[t]["kappa_hat"] = float(kappa)
            li = band(rho, self.pressure_edges)
            ci = 1 if kappa > self.cov_edge else 0
            per_t[t] = make_label(li, ci)
            conf[t] = float(c)
        cell_rho = float(np.mean([feats[t].get("rho_hat", 1.0)
                                  for t in tenants])) if tenants else 1.0
        cell = make_label(band(cell_rho, self.pressure_edges), 0)
        per_i = {iid: per_t.get(it.tenant, cell)
                 for iid, it in intents.items()}
        return RegimeReport(per_t, per_i, cell, feats, conf, self.source)


# ---------------------------------------------------------------------------
class FixedRegime(RegimeEstimator):
    """One constant label.  Used by the degenerate baselines, which consult
    no sensitivity model, so the label they are given cannot matter."""

    def estimate(self, ran, kpm, tenants, intents) -> RegimeReport:
        lab = make_label(1, 0)
        return RegimeReport(per_tenant={t: lab for t in tenants},
                            per_intent={i: lab for i in intents},
                            cell=lab, confidence={i: 1.0 for i in intents},
                            features={}, source="fixed")


def build_regime_estimator(cfg: Dict, kind: str, head=None
                           ) -> RegimeEstimator:
    kind = str(kind).lower()
    if kind in ("cell", "cell_avg"):
        return CellRegime(cfg)
    if kind in ("tenant", "tenant_load", "tenant_now"):
        return TenantLoadRegime(cfg)
    if kind == "oracle":
        est = OracleRegime(cfg)
        est.channel_aware = True
        return est
    if kind == "learned":
        if head is None:
            raise ValueError("learned regime estimator needs a context head")
        return LearnedRegime(cfg, head)
    if kind == "fixed":
        return FixedRegime(cfg)
    raise ValueError(f"unknown regime estimator {kind!r}")
