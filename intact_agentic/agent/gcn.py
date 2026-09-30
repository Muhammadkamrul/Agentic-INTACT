"""
intact_agentic/agent/gcn.py
===========================
A bipartite graph convolutional encoder, ported from the xSlice design to
pure NumPy with hand-written gradients.

WHY A GRAPH ENCODER AT ALL
--------------------------
This is the one architectural idea taken directly from xSlice, and it is
taken because it solves a problem this project genuinely has.  The number
of UEs, sessions, tenants, intents and claims CHANGES DURING A RUN: UEs
move in and out of coverage, sessions start and stop, and a tenant can
arrive mid-run.  A fixed-width MLP cannot take that as input without
zero-padding to a maximum that is either wasteful or wrong.  A GCN over a
bipartite (session, slice) graph produces a FIXED-SIZE embedding from a
VARIABLE-SIZE graph, which is exactly the property required to support
mid-run tenant arrival without retraining.

THE GRAPH
---------
Nodes are of two kinds:

    session nodes   one per UE (per traffic session in the xSlice
                    formulation).  Features: SINR, CQI, MCS, BLER, queue
                    occupancy, offered rate, distance, speed, LOS flag,
                    spectral efficiency -- exactly the KPM/MAC quantities
                    an E2 KPM service model reports.
    slice nodes     one per tenant.  Features: aggregate demand, PRB
                    quota, allocated PRBs, margin of the tenant's worst
                    intent, offered-load ratio, coverage index.

Edges connect a session to the slice it belongs to.  The adjacency is
therefore the slice membership matrix, which is what makes the
convolution "aggregate my slice's sessions" and "broadcast my slice's
state back to its sessions".

PROPAGATION
-----------
Symmetrically normalised, as in Kipf and Welling and as in xSlice:

    H' = sigma( D^-1/2 (A + I) D^-1/2 H W )

with a separate weight matrix per node type, then mean pooling over the
slice nodes to a fixed ``embed_dim`` vector.  Three layers: the paper's
own hyperparameter study found three to be best, and the same finding
applies here (deeper is slower to converge and buys nothing on a graph
with this little structure).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .nn import Linear, Module, Param, ReLU


# ---------------------------------------------------------------------------
class BipartiteGraph:
    """Session and slice node features plus the membership adjacency."""

    def __init__(self, session_x: np.ndarray, slice_x: np.ndarray,
                 membership: np.ndarray, slice_ids: Sequence[str]):
        self.session_x = np.asarray(session_x, dtype=float)
        self.slice_x = np.asarray(slice_x, dtype=float)
        # membership[u, k] = 1 if session u belongs to slice k
        self.membership = np.asarray(membership, dtype=float)
        self.slice_ids = list(slice_ids)

    @property
    def n_sessions(self) -> int:
        return self.session_x.shape[0]

    @property
    def n_slices(self) -> int:
        return self.slice_x.shape[0]

    def normalised(self) -> Tuple[np.ndarray, np.ndarray]:
        """Row-normalised aggregation operators in both directions."""
        m = self.membership
        deg_slice = np.maximum(m.sum(axis=0), 1.0)      # sessions per slice
        deg_sess = np.maximum(m.sum(axis=1), 1.0)       # slices per session
        up = m / deg_slice[None, :]          # session -> slice  (mean)
        down = m / deg_sess[:, None]         # slice  -> session
        return up, down


# ---------------------------------------------------------------------------
class GCNLayer(Module):
    """One bipartite propagation step with per-type weights."""

    def __init__(self, d_sess_in: int, d_slice_in: int, d_out: int,
                 rng: np.random.Generator, name: str):
        self.w_ss = Linear(d_sess_in, d_out, rng, f"{name}.ss", bias=False)
        self.w_su = Linear(d_sess_in, d_out, rng, f"{name}.su", bias=False)
        self.w_kk = Linear(d_slice_in, d_out, rng, f"{name}.kk", bias=False)
        self.w_kd = Linear(d_slice_in, d_out, rng, f"{name}.kd", bias=False)
        self.b_s = Param(f"{name}.bs", np.zeros(d_out))
        self.b_k = Param(f"{name}.bk", np.zeros(d_out))
        self.act_s = ReLU()
        self.act_k = ReLU()
        self._cache = None

    def forward(self, hs: np.ndarray, hk: np.ndarray, up: np.ndarray,
                down: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # slice update: own state + mean of its sessions
        zk = self.w_kk.forward(hk) + up.T @ self.w_ss.forward(hs) + \
            self.b_k.value
        # session update: own state + its slice's state
        zs = self.w_su.forward(hs) + down @ self.w_kd.forward(hk) + \
            self.b_s.value
        self._cache = (up, down)
        return self.act_s.forward(zs), self.act_k.forward(zk)

    def backward(self, gs: np.ndarray, gk: np.ndarray
                 ) -> Tuple[np.ndarray, np.ndarray]:
        up, down = self._cache
        gzs = self.act_s.backward(gs)
        gzk = self.act_k.backward(gk)
        self.b_s.grad += gzs.sum(axis=0)
        self.b_k.grad += gzk.sum(axis=0)
        # slice branch
        g_hk_1 = self.w_kk.backward(gzk)
        g_ss = self.w_ss.backward(up @ gzk)          # via up.T @ (.)
        # session branch
        g_hs_1 = self.w_su.backward(gzs)
        g_kd = self.w_kd.backward(down.T @ gzs)
        return g_hs_1 + g_ss, g_hk_1 + g_kd


class GCNEncoder(Module):
    """Three-layer bipartite GCN with mean pooling to a fixed embedding."""

    def __init__(self, d_session: int, d_slice: int, hidden: int,
                 embed: int, rng: np.random.Generator, layers: int = 3,
                 name: str = "gcn"):
        self.layers: List[GCNLayer] = []
        din_s, din_k = d_session, d_slice
        for k in range(layers - 1):
            self.layers.append(GCNLayer(din_s, din_k, hidden, rng,
                                        f"{name}.l{k}"))
            din_s = din_k = hidden
        self.out = GCNLayer(din_s, din_k, embed, rng, f"{name}.out")
        self.embed = embed
        self._last: Optional[Tuple] = None

    def forward(self, graph: BipartiteGraph) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (pooled embedding (embed,), per-slice embedding (K,embed))."""
        up, down = graph.normalised()
        hs, hk = graph.session_x, graph.slice_x
        for l in self.layers:
            hs, hk = l.forward(hs, hk, up, down)
        hs, hk = self.out.forward(hs, hk, up, down)
        pooled = hk.mean(axis=0) if hk.shape[0] else np.zeros(self.embed)
        self._last = (graph, hk.shape[0])
        return pooled, hk

    def backward(self, g_pooled: np.ndarray,
                 g_slice: Optional[np.ndarray] = None) -> None:
        graph, nk = self._last
        gk = np.zeros((nk, self.embed))
        if nk:
            gk += g_pooled[None, :] / nk
        if g_slice is not None:
            gk += g_slice
        gs = np.zeros((graph.n_sessions, self.embed))
        gs, gk = self.out.backward(gs, gk)
        for l in reversed(self.layers):
            gs, gk = l.backward(gs, gk)


# ---------------------------------------------------------------------------
def build_graph(ran, kpm: Dict, tenants: Sequence[str],
                margins: Optional[Dict[str, float]] = None,
                intents: Optional[Dict] = None,
                max_sessions_per_slice: int = 24) -> BipartiteGraph:
    """Assemble the bipartite graph from live RAN telemetry.

    ``max_sessions_per_slice`` bounds the graph so the encoder cost stays
    predictable inside the near-RT budget; when a slice has more UEs than
    that, a deterministic stratified subsample by SINR is taken (worst,
    median and best deciles are always represented) rather than a random
    one, so the embedding does not jitter between epochs for no reason.
    """
    sess_rows: List[np.ndarray] = []
    member: List[int] = []
    slice_rows: List[np.ndarray] = []
    slice_ids: List[str] = []

    for k, tid in enumerate(tenants):
        feats = ran.snapshot_ue_features(tid)
        if feats.shape[0] == 0:
            continue
        if feats.shape[0] > max_sessions_per_slice:
            order = np.argsort(feats[:, 0])          # by normalised SINR
            idx = np.linspace(0, len(order) - 1,
                              max_sessions_per_slice).astype(int)
            feats = feats[order[idx]]
        for row in feats:
            sess_rows.append(row)
            member.append(len(slice_ids))
        row = kpm.get(tid, {})
        worst_g = 0.0
        if margins and intents:
            gs = [margins[i] for i, it in intents.items()
                  if it.tenant == tid and i in margins]
            worst_g = min(gs) if gs else 0.0
        slice_rows.append(np.array([
            float(row.get("offered_input_ratio", 1.0)),
            float(row.get("prb_alloc", 0.0)) / max(ran.n_prb, 1),
            ran.quota_of(tid) / max(ran.n_prb, 1),
            float(row.get("buffer_kb", 0.0)) / 100.0,
            float(np.clip(worst_g, -1.5, 1.5)),
            float(row.get("edge_fraction", 0.0)),
            float(row.get("cqi", 7.0)) / 15.0,
            float(row.get("bler", 0.0)) / 0.5,
            float(row.get("delay_ms", 0.0)) / 200.0,
            float(row.get("n_ue", 1.0)) / 20.0,
        ], dtype=float))
        slice_ids.append(tid)

    n_s = len(sess_rows)
    n_k = max(len(slice_ids), 1)
    if n_s == 0:
        return BipartiteGraph(np.zeros((1, 10)), np.zeros((n_k, 10)),
                              np.zeros((1, n_k)), slice_ids or ["_"])
    m = np.zeros((n_s, n_k))
    for u, k in enumerate(member):
        m[u, k] = 1.0
    return BipartiteGraph(np.vstack(sess_rows), np.vstack(slice_rows), m,
                          slice_ids)


SESSION_FEATURE_NAMES = ("sinr", "cqi", "mcs", "bler", "queue", "offered",
                         "distance", "speed", "los", "spectral_efficiency")
SLICE_FEATURE_NAMES = ("offered_ratio", "prb_share", "quota_share",
                       "buffer", "worst_margin", "edge_fraction", "cqi",
                       "bler", "delay", "n_ue")
