"""
intact_agentic/agent/candidate_policy.py
========================================
The claim-aware constrained DRL policy.  Its ONLY job is SEARCH
REDUCTION: read the active claims and return a short list of candidate
portfolios for the deterministic arbiter to evaluate properly.

WHY THIS IS THE RIGHT JOB FOR A LEARNED COMPONENT
-------------------------------------------------
The arbiter must evaluate conflict-free subsets of the active claims.
With J claims that is up to 2^J portfolios, and each one costs
O(|S| x |I|) slope lookups.  At J = 10 exhaustive enumeration is about a
thousand portfolios and fits comfortably inside a near-RT budget; at
J = 18 it is a quarter of a million and does not.  A real multi-vendor
deployment is at the second number, not the first.

So there are exactly two options: give up optimality with a hand-written
greedy heuristic, or learn which few subsets are worth scoring.  The
second is strictly better if the learner is good, and CANNOT BE WORSE if
the deterministic greedy chain and the no-action portfolio are always
appended to the learner's list -- which they are, unconditionally, in
``Arbiter.decide``.  That is the whole safety argument for putting a
neural network in this loop: it can only ADD candidates, never remove the
fallbacks and never overrule the safety filter.

STATE
-----
The xSlice-style pooled GCN embedding of the (session, slice) graph,
concatenated with per-claim features:

    normalised dose, |dose|, predicted weighted margin gain, predicted
    uncertainty, whether the slope is known at all, the claim's scope and
    kind, its tenant's estimated resource pressure and coverage index,
    the worst margin among its tenant's intents, and how many epochs it
    is since this claim last wrote.

ACTION
------
An independent inclusion logit per claim.  The policy is a product of
Bernoullis, not a softmax over portfolios: the number of portfolios is
combinatorial and changes every epoch, whereas the number of claims is
small and interpretable.  The top-K list is then

    S_1   the deterministic argmax (every claim with p > 0.5, conflicts
          resolved in favour of the higher probability)
    S_2.. K-1 samples from the same Bernoulli product, de-duplicated

CONSTRAINTS
-----------
Claims whose parameter has no usable slope for a protected intent, or
that are already infeasible, are MASKED OUT before sampling.  Masking is
cheap and it keeps the policy from wasting its K slots on portfolios the
arbiter will reject anyway.

REWARD
------
    r = -(U* - U_topK) / scale        search regret: how much utility the
                                      reduced search left on the table
        + eta * realised weighted margin change next epoch
        - c_lat * (K / K_max)         the cost of asking for a big list

``U*`` is the exhaustive optimum, computed only while J is small enough
to afford it (training and evaluation both log it, but the DECISION never
uses it).  When J is too large, the greedy chain optimum stands in and
the substitution is recorded in ``agent.csv`` so nobody mistakes one for
the other.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..types import Claim, Intent, Kind, Portfolio, Scope, portfolio_key
from .gcn import GCNEncoder, build_graph
from .nn import Adam, Module, Sequential, mlp, softmax

N_CLAIM_FEATURES = 14


# ---------------------------------------------------------------------------
@dataclass
class Transition:
    claim_feats: np.ndarray
    pooled: np.ndarray
    mask: np.ndarray
    action: np.ndarray
    logp: float
    value: float
    reward: float = 0.0
    advantage: float = 0.0
    ret: float = 0.0


# ---------------------------------------------------------------------------
class CandidatePolicy(Module):
    """PPO actor-critic over claims, emitting Top-K candidate portfolios."""

    def __init__(self, cfg: Dict, rng: np.random.Generator,
                 encoder: Optional[GCNEncoder] = None):
        p = (cfg.get("agent", {}) or {}).get("policy", {}) or {}
        self.rng = rng
        self.hidden = int(p.get("hidden", 32))
        self.embed = int(p.get("embed", 12))
        self.top_k = int(p.get("top_k", 6))
        self.lr = float(p.get("lr", 1e-3))
        self.clip_eps = float(p.get("clip_eps", 0.2))
        self.gamma = float(p.get("gamma", 0.9))
        self.lam = float(p.get("gae_lambda", 0.95))
        self.epochs_per_update = int(p.get("epochs_per_update", 4))
        self.batch = int(p.get("batch", 32))
        self.buffer_size = int(p.get("buffer", 256))
        self.entropy_coef = float(p.get("entropy_coef", 0.01))
        self.value_coef = float(p.get("value_coef", 0.5))
        self.warmup = int(p.get("warmup_epochs", 25))
        _cap = p.get("cold_start_max_candidates", None)
        self.cold_start_cap = None if _cap in (None, "", 0) else int(_cap)
        self.latency_cost = float(p.get("latency_cost", 0.02))
        self.regret_scale = float(p.get("regret_scale", 0.02))
        self.outcome_gain = float(p.get("outcome_gain", 1.0))
        self.explore_sigma = float(p.get("explore_logit_noise", 0.0))
        self.max_sessions = int(p.get("max_sessions_per_slice", 24))

        self.encoder = encoder or GCNEncoder(10, 10, self.hidden, self.embed,
                                             rng, layers=3, name="pol_gcn")
        self.actor = mlp([self.embed + N_CLAIM_FEATURES, self.hidden,
                          self.hidden, 1], rng, "actor")
        self.critic = mlp([self.embed + N_CLAIM_FEATURES, self.hidden, 1],
                          rng, "critic")
        self.opt = Adam(self.parameters(), lr=self.lr)
        self.buffer: Deque[Transition] = deque(maxlen=self.buffer_size)
        self.epochs_seen = 0
        self.updates = 0
        self.last_loss = float("nan")
        self.last_entropy = float("nan")
        self.stats: List[Dict] = []
        self._pending: Optional[Transition] = None
        self._last_write_epoch: Dict[str, int] = {}

    # ------------------------------------------------------------------
    def claim_features(self, *, claims: Dict[str, Claim], reqs, intents,
                       margins: Dict[str, float], sens, regimes, epoch: int,
                       jids: Sequence[str]) -> np.ndarray:
        rows = []
        for j in jids:
            c = claims[j]
            rq = reqs[j]
            dose_n = rq.dose / max(c.width, 1e-9)
            gain, unc, known = 0.0, 0.0, 1.0
            for iid, it in intents.items():
                r = regimes.of_intent(iid)
                s = sens.get(r, c.param, iid)
                gain += it.pi_class * it.weight * s * rq.dose
                unc += (sens.sigma(r, c.param, iid) * rq.dose) ** 2
                if not sens.known(r, c.param, iid):
                    known = 0.0
            own = [margins.get(i, 0.0) for i, it in intents.items()
                   if it.tenant == c.tenant]
            worst_own = min(own) if own else 0.0
            others = [margins.get(i, 0.0) for i, it in intents.items()
                      if it.tenant != c.tenant]
            worst_other = min(others) if others else 0.0
            feats = regimes.features.get(c.tenant, {})
            rows.append([
                float(np.clip(dose_n, -1, 1)),
                float(min(abs(dose_n), 1.0)),
                float(np.clip(gain * 10.0, -5, 5)),
                float(min(np.sqrt(unc) * 10.0, 5.0)),
                known,
                1.0 if c.scope == Scope.CELL else 0.0,
                1.0 if c.kind == Kind.ALLOCATIVE else 0.0,
                float(np.clip(worst_own, -1.5, 1.5)),
                float(np.clip(worst_other, -1.5, 1.5)),
                float(np.clip(feats.get("rho_hat",
                                        feats.get("resource_pressure", 1.0)),
                              0, 3)) / 3.0,
                float(np.clip(feats.get("kappa_hat",
                                        feats.get("coverage_index", 0.0)),
                              0, 1)),
                float(np.clip(feats.get("offered_input_ratio", 1.0), 0, 3))
                / 3.0,
                float(min((epoch - self._last_write_epoch.get(j, 0)) / 20.0,
                          2.0)),
                float(c.r_j),
            ])
        return np.asarray(rows, dtype=float) if rows \
            else np.zeros((0, N_CLAIM_FEATURES))

    # ------------------------------------------------------------------
    def _logits(self, pooled: np.ndarray, cf: np.ndarray) -> np.ndarray:
        if cf.shape[0] == 0:
            return np.zeros(0)
        x = np.concatenate([np.tile(pooled, (cf.shape[0], 1)), cf], axis=1)
        self._x = x
        return self.actor.forward(x)[:, 0]

    def _values(self, pooled: np.ndarray, cf: np.ndarray) -> float:
        if cf.shape[0] == 0:
            return 0.0
        x = np.concatenate([np.tile(pooled, (cf.shape[0], 1)), cf], axis=1)
        self._xv = x
        return float(self.critic.forward(x).mean())

    # ------------------------------------------------------------------
    def propose(self, *, ran, kpm, tenants, claims, reqs, intents, margins,
                sens, regimes, epoch: int, mask: Optional[Dict[str, bool]] =
                None, explore: bool = False
                ) -> Tuple[List[Portfolio], Dict]:
        """Return the top-K candidate portfolios plus diagnostics."""
        jids = sorted(reqs)
        if not jids:
            return [()], {"k": 1, "policy": "empty"}
        cf = self.claim_features(claims=claims, reqs=reqs, intents=intents,
                                 margins=margins, sens=sens, regimes=regimes,
                                 epoch=epoch, jids=jids)
        graph = build_graph(ran, kpm, tenants, margins, intents,
                            max_sessions_per_slice=self.max_sessions)
        pooled, _ = self.encoder.forward(graph)

        if self.epochs_seen < self.warmup:
            # cold start: return the full enumeration when affordable, so
            # the agentic method is never worse than the static one while
            # the policy is still untrained
            self._pending = None
            return self._cold_start(jids, claims, reqs), \
                {"k": 0, "policy": "warmup", "pooled_norm":
                 float(np.linalg.norm(pooled))}

        logits = self._logits(pooled, cf)
        m = np.ones(len(jids), dtype=bool)
        if mask:
            m = np.array([bool(mask.get(j, True)) for j in jids])
        if explore and self.explore_sigma > 0:
            logits = logits + self.rng.normal(0.0, self.explore_sigma,
                                              logits.shape)
        logits = np.where(m, logits, -30.0)
        p = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))

        cands: List[Portfolio] = []
        # deterministic argmax candidate
        det = self._resolve(jids, p > 0.5, p, claims)
        cands.append(det)
        # sampled candidates
        for _ in range(max(self.top_k - 1, 0) * 3):
            if len(cands) >= self.top_k:
                break
            draw = self.rng.random(len(jids)) < p
            s = self._resolve(jids, draw, p, claims)
            if s not in cands:
                cands.append(s)
        action = np.isin(np.array(jids), np.array(det, dtype=object)) \
            if det else np.zeros(len(jids), dtype=bool)
        logp = float(np.sum(np.where(action, np.log(np.maximum(p, 1e-9)),
                                     np.log(np.maximum(1 - p, 1e-9)))))
        value = self._values(pooled, cf)
        self._pending = Transition(claim_feats=cf, pooled=pooled, mask=m,
                                   action=action.astype(float), logp=logp,
                                   value=value)
        ent = float(np.mean(-(p * np.log(np.maximum(p, 1e-9))
                              + (1 - p) * np.log(np.maximum(1 - p, 1e-9)))))
        self.last_entropy = ent
        return cands, {"k": len(cands), "policy": "ppo",
                       "mean_p": float(p.mean()), "entropy": ent,
                       "masked": int((~m).sum())}

    def _cold_start(self, jids, claims, reqs) -> List[Portfolio]:
        """Before the policy is trained, fall back to the exhaustive list.

        With ``cold_start_max_candidates`` set, the fallback is BOUNDED so a
        warm-up decision still meets the near-RT budget: no-action, every
        single claim, the greedy chain by |dose|, then C1-free pairs in
        order of combined |dose|, until the budget.  It is deterministic and
        consumes no random numbers.  Measured on S1 (nine claims): the
        unbounded fallback scored up to 391 candidates and took up to 71 ms,
        against a 2.3 ms median once the policy was trained.

        The default (None) is the unbounded behaviour INTACT-RA-Agentic had
        when it was frozen for the held-out evaluation, so those results
        reproduce exactly; on S16 its warm-up fell inside the unscored
        burn-in.
        """
        import itertools
        cap = self.cold_start_cap
        if cap is not None:
            by = sorted(jids, key=lambda j: (-abs(reqs[j].dose), j))
            out, seen = [()], {()}

            def add(sub):
                key = tuple(sorted(sub))
                params = [claims[j].param for j in key]
                if key not in seen and len(set(params)) == len(params) \
                        and len(out) < cap:
                    seen.add(key)
                    out.append(key)
            for j in by:
                add((j,))
            cur, used = [], set()
            for j in by:
                if claims[j].param not in used:
                    cur.append(j)
                    used.add(claims[j].param)
                    add(tuple(cur))
            pairs = sorted(itertools.combinations(by, 2),
                           key=lambda s: (-(abs(reqs[s[0]].dose)
                                            + abs(reqs[s[1]].dose)), s))
            for s in pairs:
                if len(out) >= cap:
                    break
                add(s)
            return out
        if len(jids) > 12:
            order = sorted(jids, key=lambda j: (-abs(reqs[j].dose), j))
            chain, cur, used = [()], [], set()
            for j in order:
                if claims[j].param in used:
                    continue
                cur.append(j)
                used.add(claims[j].param)
                chain.append(tuple(cur))
            return chain
        out = []
        for r in range(len(jids) + 1):
            for sub in itertools.combinations(jids, r):
                params = [claims[j].param for j in sub]
                if len(set(params)) == len(params):
                    out.append(sub)
        return out

    @staticmethod
    def _resolve(jids: Sequence[str], sel: np.ndarray, p: np.ndarray,
                 claims: Dict[str, Claim]) -> Portfolio:
        """Turn a raw Bernoulli draw into a conflict-free portfolio."""
        chosen: Dict[str, Tuple[str, float]] = {}
        for k, j in enumerate(jids):
            if not sel[k]:
                continue
            prm = claims[j].param
            if prm not in chosen or p[k] > chosen[prm][1]:
                chosen[prm] = (j, float(p[k]))
        return portfolio_key([v[0] for v in chosen.values()])

    # ------------------------------------------------------------------
    def reward(self, *, chosen_utility: float, best_utility: float,
               realised_dg: float, k: int) -> float:
        regret = max(best_utility - chosen_utility, 0.0)
        r = -regret / max(self.regret_scale, 1e-9)
        r += self.outcome_gain * realised_dg / max(self.regret_scale, 1e-9)
        r -= self.latency_cost * (k / max(self.top_k, 1))
        return float(np.clip(r, -10.0, 10.0))

    def record(self, reward: float, epoch: int,
               executed: Sequence[str] = ()) -> None:
        for j in executed:
            self._last_write_epoch[j] = epoch
        self.epochs_seen += 1
        if self._pending is None:
            return
        self._pending.reward = float(reward)
        self.buffer.append(self._pending)
        self._pending = None

    # ------------------------------------------------------------------
    def update(self) -> Dict:
        """One PPO update over the replay buffer."""
        if len(self.buffer) < max(self.batch, 8):
            return {}
        data = list(self.buffer)
        # GAE over the stored sequence (a single continuing trajectory)
        adv, gae = np.zeros(len(data)), 0.0
        for t in reversed(range(len(data))):
            nxt = data[t + 1].value if t + 1 < len(data) else 0.0
            delta = data[t].reward + self.gamma * nxt - data[t].value
            gae = delta + self.gamma * self.lam * gae
            adv[t] = gae
        ret = adv + np.array([d.value for d in data])
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        losses = []
        for _ in range(self.epochs_per_update):
            idx = self.rng.permutation(len(data))[:self.batch]
            self.opt.zero_grad()
            tot = 0.0
            for i in idx:
                d = data[int(i)]
                if d.claim_feats.shape[0] == 0:
                    continue
                logits = self._logits(d.pooled, d.claim_feats)
                logits = np.where(d.mask, logits, -30.0)
                p = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
                logp = np.sum(np.where(d.action > 0.5,
                                       np.log(np.maximum(p, 1e-9)),
                                       np.log(np.maximum(1 - p, 1e-9))))
                ratio = float(np.exp(np.clip(logp - d.logp, -10, 10)))
                a = float(adv[int(i)])
                unclipped = ratio * a
                clipped = float(np.clip(ratio, 1 - self.clip_eps,
                                        1 + self.clip_eps)) * a
                pg = -min(unclipped, clipped)
                # entropy bonus
                ent = float(np.mean(-(p * np.log(np.maximum(p, 1e-9))
                                      + (1 - p)
                                      * np.log(np.maximum(1 - p, 1e-9)))))
                # ---- gradients -----------------------------------
                use_unclipped = unclipped <= clipped
                dpg = np.zeros_like(p)
                if use_unclipped and abs(ratio) < 50:
                    dlogp = (d.action - p)
                    dpg = -a * ratio * dlogp
                dent = -(np.log(np.maximum(p, 1e-9))
                         - np.log(np.maximum(1 - p, 1e-9))) * p * (1 - p) \
                    / max(len(p), 1)
                glog = (dpg - self.entropy_coef * dent) * np.where(
                    d.mask, 1.0, 0.0)
                self.actor.backward(glog[:, None])
                # critic
                v = self.critic.forward(self._xv if hasattr(self, "_xv")
                                        else np.concatenate(
                    [np.tile(d.pooled, (d.claim_feats.shape[0], 1)),
                     d.claim_feats], axis=1))
                vm = float(v.mean())
                dv = 2.0 * (vm - float(ret[int(i)])) / max(len(v), 1)
                self.critic.backward(np.full_like(v, dv) * self.value_coef)
                tot += pg + self.value_coef * (vm - float(ret[int(i)])) ** 2
            self.opt.step()
            losses.append(tot / max(len(idx), 1))
        self.updates += 1
        self.last_loss = float(np.mean(losses)) if losses else float("nan")
        return {"policy_loss": self.last_loss, "policy_updates": self.updates,
                "policy_entropy": self.last_entropy,
                "buffer": len(self.buffer)}

    # ------------------------------------------------------------------
    def diagnostics(self) -> Dict:
        return {"policy_epochs": self.epochs_seen,
                "policy_updates": self.updates,
                "policy_loss": self.last_loss,
                "policy_entropy": self.last_entropy}

    def state_dict(self) -> Dict:
        return {"encoder": self.encoder.state_dict(),
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "epochs_seen": self.epochs_seen}

    def load_state_dict(self, blob: Dict) -> None:
        if not blob:
            return
        self.encoder.load_state_dict(blob.get("encoder", {}))
        self.actor.load_state_dict(blob.get("actor", {}))
        self.critic.load_state_dict(blob.get("critic", {}))
        self.epochs_seen = int(blob.get("epochs_seen", 0))
