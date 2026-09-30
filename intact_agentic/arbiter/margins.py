"""
intact_agentic/arbiter/margins.py
=================================
Turn incomparable KPIs into a dimensionless MARGIN.

    g_i = d_i * (measured - target) / |target|,   clipped to +/- clip

    d_i = +1 if higher is better, -1 if lower is better

    g = 0      exactly on target
    g = +0.15  fifteen percent better than promised
    g = -0.05  five percent short

This single step is what makes "which intent is in more trouble" a
well-posed question across a throughput intent in Mb/s, a latency intent
in ms, a buffer intent in kB and a host power intent in dBm.  Every
weight, every safety floor, every sensitivity slope and every utility in
the rest of the package is arithmetic on these comparable numbers.

FULFILMENT
----------
``rho_i`` is the fraction of the rolling window in which ``g_i >= 0``.
The paper's headline metric, weighted intent fulfilment (wIF), is the
pi_class-weighted mean of the per-intent fulfilment indicator, so it is
computed from exactly the same quantity the arbiter optimises -- no
metric/objective mismatch.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Optional, Sequence

import numpy as np

from ..types import Direction, Intent


def kpi_value(intent: Intent, kpm: Dict[str, Dict[str, float]]
              ) -> Optional[float]:
    """Read the intent's KPI out of a KPM report, or None if absent.

    Host intents are written against cell-wide measurements, tenant
    intents against their own slice.  A tenant that has not arrived yet
    has no row, and the caller must treat that as "no observation", never
    as zero.
    """
    cell = kpm.get("_cell", {})
    if intent.tenant == "_cell" or intent.kpi in cell:
        if intent.kpi in cell:
            return float(cell[intent.kpi])
        return None
    row = kpm.get(intent.tenant)
    if not row or intent.kpi not in row:
        return None
    return float(row[intent.kpi])


def margin(intent: Intent, kpm: Dict[str, Dict[str, float]]
           ) -> Optional[float]:
    v = kpi_value(intent, kpm)
    if v is None:
        return None
    g = intent.sign * (v - intent.target) / max(abs(intent.target), 1e-12)
    return float(np.clip(g, -intent.clip, intent.clip))


def margin_to_kpi(intent: Intent, g: float) -> float:
    """Inverse of :func:`margin`, used to report a predicted KPI."""
    return float(intent.target + intent.sign * g * abs(intent.target))


class MarginTracker:
    """Rolling per-intent margin history, fulfilment and trend."""

    def __init__(self, window: int = 40):
        self.window = int(window)
        self.hist: Dict[str, Deque[float]] = {}

    def ensure(self, iid: str) -> None:
        if iid not in self.hist:
            self.hist[iid] = deque(maxlen=self.window)

    def update(self, intents: Dict[str, Intent],
               kpm: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for iid, it in intents.items():
            g = margin(it, kpm)
            if g is None:
                continue
            self.ensure(iid)
            self.hist[iid].append(g)
            out[iid] = g
        return out

    def fulfilment(self, iid: str) -> float:
        h = self.hist.get(iid)
        if not h:
            return 1.0
        return sum(1 for x in h if x >= 0.0) / len(h)

    def trend(self, iid: str, k: int = 8) -> float:
        h = list(self.hist.get(iid, ()))
        if len(h) < 3:
            return 0.0
        k = min(k, len(h) - 1)
        return (h[-1] - h[-1 - k]) / k

    def volatility(self, iid: str) -> float:
        h = list(self.hist.get(iid, ()))
        return float(np.std(h)) if len(h) > 2 else 0.0


def weighted_intent_fulfilment(intents: Dict[str, Intent],
                               fulfilled: Dict[str, Sequence[int]]) -> float:
    """wIF = sum_i pi_i * mean(1[g_i >= 0]) / sum_i pi_i."""
    num = 0.0
    den = 0.0
    for iid, it in intents.items():
        seq = fulfilled.get(iid)
        if not seq:
            continue
        num += it.pi_class * float(np.mean(seq))
        den += it.pi_class
    return float(num / den) if den > 0 else 0.0
