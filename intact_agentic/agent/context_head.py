"""
intact_agentic/agent/context_head.py
====================================
The LOAD / REGIME estimator.  Requirement: "estimate the load more
accurately considering channel quality, capacity, noise, interference,
mobility, traffic."

WHAT IT PREDICTS, AND WHY THOSE TWO NUMBERS
--------------------------------------------
Per tenant, two scalars:

  rho    RESOURCE PRESSURE = offered demand / deliverable capacity.
         Not offered load.  Deliverable capacity already contains the
         CQI/SINR distribution, the MCS ceiling and the PRB quota, so a
         tenant whose users moved to the cell edge has HIGHER pressure at
         UNCHANGED traffic.  This is the quantity a load meter cannot
         see, and it is the first axis of the regime label.

  kappa  COVERAGE INDEX = fraction of the tenant's UEs whose bottleneck is
         signal rather than blocks (SINR below the configured knee).  It
         is the second axis, and it is what tells the table whether a dB
         of power or a PRB is the thing worth spending.

Both are computable from the simulator, so they act as LABELS.  That is
not cheating: in a real deployment both are computable after the fact from
the same KPM/MAC report the xApp already receives (per-UE throughput,
scheduled PRBs, CQI, MCS, BLER).  The head learns to produce them from the
CURRENT observation, one epoch before the label exists, which is the part
that has to be learned.

ARCHITECTURE
------------
GCN encoder (``gcn.py``) over the bipartite session/slice graph, then a
per-slice head:  [slice embedding ; pooled cell embedding] -> 2 outputs.
Sharing the encoder with the PPO policy is deliberate -- it is the same
representation-learning problem, and it halves the online cost.

TRAINING
--------
Online, supervised, with a replay buffer and a small learning rate.
Huber loss, because a single congestion spike should not dominate.  The
head is evaluated continuously: ``confidence`` is 1 minus the normalised
rolling absolute error, and a low-confidence tenant is reported as such so
the arbiter's robust bound widens instead of the head being trusted
blindly.

COLD START
----------
Before ``warmup_epochs`` the head returns the ANALYTICAL fallback
(measured offered-load ratio and the geometric coverage index) with
confidence 0.  A learned component that is not yet learned must not be
allowed to make the system worse than the baseline it replaces; the
fallback guarantees INTACT-RA-Agentic is never worse than INTACT-RA at
epoch 1.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .gcn import BipartiteGraph, GCNEncoder, build_graph
from .nn import Adam, Linear, Module, ReLU, Sequential, mlp


class ContextHead(Module):
    """Online GCN regressor for (resource pressure, coverage index)."""

    def __init__(self, cfg: Dict, rng: np.random.Generator,
                 encoder: Optional[GCNEncoder] = None):
        a = (cfg.get("agent", {}) or {}).get("context", {}) or {}
        self.hidden = int(a.get("hidden", 24))
        self.embed = int(a.get("embed", 12))
        self.lr = float(a.get("lr", 3e-3))
        self.warmup = int(a.get("warmup_epochs", 20))
        self.buffer_size = int(a.get("buffer", 240))
        self.batch = int(a.get("batch", 16))
        self.updates_per_epoch = int(a.get("updates_per_epoch", 2))
        self.huber_delta = float(a.get("huber_delta", 0.5))
        self.conf_scale = float(a.get("confidence_scale", 0.25))
        self.max_sessions = int(a.get("max_sessions_per_slice", 24))
        self.rho_scale = float(a.get("rho_proxy_scale", 1.0))
        self.fb_err: Dict = {}
        self.source: Dict = {}
        self.rho_offset = float(a.get("rho_proxy_offset", 0.0))

        self.rng = rng
        self.encoder = encoder or GCNEncoder(10, 10, self.hidden, self.embed,
                                             rng, layers=3, name="ctx_gcn")
        self.head = mlp([2 * self.embed, self.hidden, 2], rng, "ctx_head")
        self.opt = Adam(self.parameters(), lr=self.lr)
        self.buffer: Deque[Tuple[BipartiteGraph, np.ndarray]] = deque(
            maxlen=self.buffer_size)
        self.err_hist: Dict[str, Deque[float]] = {}
        self.epochs_seen = 0
        self.train_loss: List[float] = []

    # ------------------------------------------------------------------
    def _forward_graph(self, graph: BipartiteGraph
                       ) -> Tuple[np.ndarray, np.ndarray]:
        pooled, per_slice = self.encoder.forward(graph)
        if per_slice.shape[0] == 0:
            return np.zeros((0, 2)), pooled
        x = np.concatenate(
            [per_slice, np.tile(pooled, (per_slice.shape[0], 1))], axis=1)
        y = self.head.forward(x)
        # rho is positive and typically in [0, 3]; kappa is a fraction
        out = np.column_stack([
            np.maximum(y[:, 0], 0.0) if y.shape[1] > 0 else y[:, 0],
            np.clip(y[:, 1], 0.0, 1.0)])
        self._cache = (graph, x, y)
        return out, pooled

    # ------------------------------------------------------------------
    def _analytic(self, ran, kpm: Dict, tenants: Sequence[str]) -> Dict:
        """Analytic (rho, kappa) from observable telemetry; see predict()."""
        fallback = {}
        for t in tenants:
            row = kpm.get(t, {})
            offered = float(row.get("offered_slice_mbps",
                                    row.get("offered_mbps", 0.0)))
            # capacity of the tenant's RESERVATION -- which the agent itself
            # configured, so it is observable -- rather than of the PRBs it
            # happened to use: a tenant below its reservation uses exactly
            # what it needs, which pinned the old proxy near 1 and hid
            # whether the reservation was binding
            try:
                prb = max(float(ran.quota_of(t, ran.commanded_controls())),
                          1e-6)
            except Exception:
                prb = max(float(row.get("prb_alloc", 0.0)), 1e-6)
            se = max(float(row.get("spectral_efficiency", 0.1)), 1e-3)
            cap_mbps = prb * float(getattr(ran, "prb_hz", 360000.0)) \
                * se / 1e6
            raw = offered / max(cap_mbps, 1e-6)
            # affine correction fitted once by scripts/calibrate.py: the
            # raw ratio is monotone in true pressure but biased high,
            # because allocated PRBs already exclude the demand the
            # scheduler could not serve
            rho = float(np.clip(self.rho_scale * raw + self.rho_offset,
                                0.0, 3.0))
            fallback[t] = (rho, float(row.get("edge_fraction", 0.0)), 0.0)
        return fallback

    def predict(self, ran, kpm: Dict, tenants: Sequence[str]
                ) -> Dict[str, Tuple[float, float, float]]:
        """{tenant: (rho_hat, kappa_hat, confidence)}."""
        # Cold-start fallback: an ANALYTIC estimate of the same two
        # quantities the head is trained to predict, computed from KPM
        # fields a real RIC already receives.  It must be on the same
        # scale as the label -- an earlier version fell back to the
        # offered-load RATIO (order 1.0) while the consumer banded it as
        # resource PRESSURE (order 0.5), which silently pushed every
        # cold-start label into the wrong table column.
        fallback = self._analytic(ran, kpm, tenants)
        if self.epochs_seen < self.warmup:
            return fallback
        graph = build_graph(ran, kpm, tenants,
                            max_sessions_per_slice=self.max_sessions)
        pred, _ = self._forward_graph(graph)
        out = {}
        self.source = {}
        for k, tid in enumerate(graph.slice_ids):
            if k >= pred.shape[0]:
                continue
            e = self.err_hist.get(tid)
            fe = self.fb_err.get(tid)
            conf = 1.0 if not e else float(np.clip(
                1.0 - np.mean(e) / max(self.conf_scale, 1e-6), 0.0, 1.0))
            # ONLINE MODEL SELECTION.  Both estimators are scored against the
            # realised pressure every epoch (it is computable from telemetry
            # after the fact).  The network is used for a tenant only while
            # its recent error is lower than the analytic estimate's.
            # Trusting the network unconditionally after warm-up let a head
            # that output exactly 0.0 for a tenant at true pressure 3+ file
            # it as non-binding, and the agent then refused to restore that
            # tenant's reservation.
            if e and fe and np.mean(e) < np.mean(fe):
                out[tid] = (float(pred[k, 0]), float(pred[k, 1]), conf)
                self.source[tid] = "gcn"
            else:
                out[tid] = fallback[tid]
                self.source[tid] = "analytic"
        for t in tenants:
            out.setdefault(t, fallback[t])
        return out

    # ------------------------------------------------------------------
    def observe(self, ran, kpm: Dict, tenants: Sequence[str]) -> Dict:
        """Record one labelled example and take a few gradient steps."""
        self.epochs_seen += 1
        graph = build_graph(ran, kpm, tenants,
                            max_sessions_per_slice=self.max_sessions)
        labels = []
        for tid in graph.slice_ids:
            labels.append([float(ran.resource_pressure(tid, kpm)),
                           float(ran.coverage_index(tid))])
        if not labels:
            return {}
        Y = np.asarray(labels, dtype=float)
        self.buffer.append((graph, Y))

        # rolling error of the ANALYTIC estimate, for model selection
        fb = self._analytic(ran, kpm, tenants)
        for k, tid in enumerate(graph.slice_ids):
            if tid in fb:
                ferr = float(abs(fb[tid][0] - Y[k, 0])
                             + abs(fb[tid][1] - Y[k, 1])) / 2.0
                self.fb_err.setdefault(tid, deque(maxlen=20)).append(ferr)
        # rolling per-tenant error for the confidence report
        if self.epochs_seen > self.warmup:
            pred, _ = self._forward_graph(graph)
            for k, tid in enumerate(graph.slice_ids):
                if k >= pred.shape[0]:
                    continue
                err = float(abs(pred[k, 0] - Y[k, 0])
                            + abs(pred[k, 1] - Y[k, 1])) / 2.0
                self.err_hist.setdefault(tid, deque(maxlen=20)).append(err)

        loss = 0.0
        for _ in range(self.updates_per_epoch):
            loss = self._train_batch()
        self.train_loss.append(loss)
        return {"context_loss": loss, "context_buffer": len(self.buffer)}

    def _train_batch(self) -> float:
        if len(self.buffer) < 4:
            return 0.0
        idx = self.rng.integers(0, len(self.buffer),
                                size=min(self.batch, len(self.buffer)))
        total = 0.0
        self.opt.zero_grad()
        for i in idx:
            graph, Y = self.buffer[int(i)]
            pred, _ = self._forward_graph(graph)
            n = min(pred.shape[0], Y.shape[0])
            if n == 0:
                continue
            diff = pred[:n] - Y[:n]
            # Huber
            d = self.huber_delta
            absd = np.abs(diff)
            loss = np.where(absd <= d, 0.5 * diff ** 2,
                            d * (absd - 0.5 * d))
            total += float(loss.mean())
            grad = np.where(absd <= d, diff, d * np.sign(diff)) / max(
                n * 2, 1)
            # backprop through the output activations
            _, x, y = self._cache
            g = np.zeros_like(y)
            g[:n, 0] = grad[:, 0] * (y[:n, 0] > 0.0)
            g[:n, 1] = grad[:, 1] * ((y[:n, 1] > 0.0) & (y[:n, 1] < 1.0))
            gx = self.head.backward(g)
            g_slice = gx[:, :self.embed]
            g_pool = gx[:, self.embed:].sum(axis=0)
            self.encoder.backward(g_pool, g_slice)
        self.opt.step()
        return total / max(len(idx), 1)

    # ------------------------------------------------------------------
    def diagnostics(self) -> Dict:
        errs = [np.mean(v) for v in self.err_hist.values() if v]
        return {"context_epochs": self.epochs_seen,
                "context_mae": float(np.mean(errs)) if errs else float("nan"),
                "context_loss": float(self.train_loss[-1])
                if self.train_loss else float("nan")}

    def state_dict(self) -> Dict:
        return {"encoder": self.encoder.state_dict(),
                "head": self.head.state_dict(),
                "epochs_seen": self.epochs_seen}

    def load_state_dict(self, blob: Dict) -> None:
        if not blob:
            return
        self.encoder.load_state_dict(blob.get("encoder", {}))
        self.head.load_state_dict(blob.get("head", {}))
        self.epochs_seen = int(blob.get("epochs_seen", 0))
