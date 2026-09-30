"""Outcome metrics.

The headline metric is weighted intent fulfilment (wIF): the pi_class
weighted fraction of epochs in which an intent's margin is non-negative.
It is the same definition used throughout INTACT-RA, kept unchanged so
the two papers' numbers are directly comparable.

The safety metrics are deliberately stricter than "did the KPI dip".  A
SAFETY CROSSING is an epoch where an intent was at or above its floor
before the writes and below it after -- that is, where the controller had
the information to know better and acted anyway.  A drift-induced dip
that no write caused is NOT a safety crossing, and counting it as one
would let a do-nothing controller look reckless.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
@dataclass
class EpochRecord:
    """Everything one epoch contributes to the outcome metrics."""
    epoch: int
    g_before: Dict[str, float] = field(default_factory=dict)
    g_after: Dict[str, float] = field(default_factory=dict)
    predicted: Dict[str, float] = field(default_factory=dict)
    epsilon: Dict[str, float] = field(default_factory=dict)
    pi_class: Dict[str, float] = field(default_factory=dict)
    weight: Dict[str, float] = field(default_factory=dict)
    n_writes: int = 0
    n_proposed: int = 0
    n_override: int = 0
    n_reject: int = 0
    n_abstain: int = 0
    c1_violations: int = 0
    c2_violations: int = 0
    candidates: int = 0
    candidate_source: str = ""
    latency_ms: float = 0.0
    utility: float = 0.0
    best_utility: float = float("nan")   # exhaustive optimum, when computed
    probe: int = 0
    reconfig_prb: float = 0.0
    regimes: Dict[str, str] = field(default_factory=dict)
    kpm_cell: Dict[str, float] = field(default_factory=dict)
    unknown_entries: int = 0
    g_cf: Dict[str, float] = field(default_factory=dict)   # no-write arm

    def as_row(self) -> Dict:
        row = {
            "epoch": self.epoch, "n_writes": self.n_writes,
            "n_proposed": self.n_proposed, "n_override": self.n_override,
            "n_reject": self.n_reject, "n_abstain": self.n_abstain,
            "c1_violations": self.c1_violations,
            "c2_violations": self.c2_violations,
            "candidates": self.candidates,
            "candidate_source": self.candidate_source,
            "latency_ms": self.latency_ms, "utility": self.utility,
            "best_utility": self.best_utility, "probe": self.probe,
            "reconfig_prb": self.reconfig_prb,
            "unknown_entries": self.unknown_entries,
        }
        for iid, g in self.g_after.items():
            row[f"g_{iid}"] = g
        for iid, g in self.g_before.items():
            row[f"gpre_{iid}"] = g
        for iid, r in self.regimes.items():
            row[f"regime_{iid}"] = r
        for k, v in self.kpm_cell.items():
            row[f"cell_{k}"] = v
        return row


# ---------------------------------------------------------------------------
class MetricAccumulator:
    """Streams epoch records and produces the summary at the end."""

    def __init__(self, warmup_epochs: int = 0):
        self.warmup = int(warmup_epochs)
        self.records: List[EpochRecord] = []

    def add(self, rec: EpochRecord) -> None:
        self.records.append(rec)

    # ------------------------------------------------------------------
    @property
    def scored(self) -> List[EpochRecord]:
        return [r for r in self.records if r.epoch >= self.warmup]

    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, float]:
        rs = self.scored
        if not rs:
            return {}
        iids = sorted({i for r in rs for i in r.g_after})

        # ---- fulfilment -------------------------------------------------
        per_intent: Dict[str, float] = {}
        for iid in iids:
            vals = [1.0 if r.g_after[iid] >= 0.0 else 0.0
                    for r in rs if iid in r.g_after]
            per_intent[iid] = float(np.mean(vals)) if vals else float("nan")

        pi = {}
        for r in rs:
            pi.update(r.pi_class)
        wsum = sum(pi.get(i, 1.0) for i in iids) or 1.0
        wif = sum(pi.get(i, 1.0) * per_intent[i]
                  for i in iids if np.isfinite(per_intent[i])) / wsum

        worst = min((per_intent[i] for i in iids
                     if np.isfinite(per_intent[i])), default=float("nan"))

        # ---- PRIMARY: unweighted intent fulfilment ----------------------
        # The fraction of (intent, epoch) pairs whose margin is
        # non-negative.  Unweighted, because no controller in this study
        # optimises contract weights: B3 and INTACT-RA (frozen as
        # "INTACT-RA-lean, no contract weights") and INTACT-RA-Agentic all
        # treat every intent equally.  Scoring them against pi_class
        # weights they never saw would measure alignment with a preference
        # the controllers were not asked to satisfy, and could favour one
        # of them purely according to which intents it happened to serve.
        IF = float(np.mean([v for v in per_intent.values()
                            if np.isfinite(v)])) if per_intent else float("nan")

        # ---- SECONDARY: shortfall -- HOW BADLY promises are broken -------
        # Fulfilment is a threshold, so it cannot tell a margin of -0.01
        # from one of -1.00.  An SLA penalty is usually proportional to the
        # shortfall, and a continuous measure also has more statistical
        # power: it responds to a real improvement that does not happen to
        # push any intent across zero.  It responds equally to a real
        # regression, which is why it is fixed in advance rather than
        # chosen after the fact.
        short = [max(0.0, -g) for r in rs for g in r.g_after.values()]
        mean_shortfall = float(np.mean(short)) if short else float("nan")
        viol = [s for s in short if s > 0]
        shortfall_when_violated = float(np.mean(viol)) if viol else 0.0

        # mean margin, and mean margin of the WORST intent each epoch --
        # a controller that keeps the average up by sacrificing one tenant
        # should be visible.
        mean_margin = float(np.mean([np.mean(list(r.g_after.values()))
                                     for r in rs if r.g_after]))
        worst_margin = float(np.mean([min(r.g_after.values())
                                      for r in rs if r.g_after]))

        # ---- safety -----------------------------------------------------
        crossings = 0
        zero_crossings = 0
        exposure = 0
        for r in rs:
            for iid, gb in r.g_before.items():
                ga = r.g_after.get(iid)
                if ga is None:
                    continue
                eps = r.epsilon.get(iid, 0.0)
                if gb >= eps:
                    exposure += 1
                    if ga < eps and r.n_writes > 0:
                        crossings += 1
                if gb >= 0.0 > ga and r.n_writes > 0:
                    zero_crossings += 1
        crossing_rate = crossings / max(exposure, 1)

        # ---- CAUSAL safety and prediction (paired no-write counterfactual)
        # The crossing count above attributes a threshold crossing to the
        # decision whenever the epoch contained a write, whether or not the
        # write caused it.  With several intents near their floors, plant
        # noise alone flips some of them in any short window, so that count
        # scales with how OFTEN a controller writes rather than with how
        # much harm it does.  The causal version requires the no-write arm
        # -- same state, same random draws -- to have stayed at or above
        # the floor: the write, and only the write, pushed it under.
        c_cross, c_rescue, c_err, c_gain = 0, 0, [], []
        have_cf = any(r.g_cf for r in rs)
        for r in rs:
            if not r.g_cf:
                continue
            for iid, ga in r.g_after.items():
                gc = r.g_cf.get(iid)
                gb = r.g_before.get(iid)
                if gc is None or gb is None:
                    continue
                eps = r.epsilon.get(iid, 0.0)
                if r.n_writes > 0:
                    if gb >= eps and gc >= eps and ga < eps:
                        c_cross += 1
                    if gc < 0.0 <= ga:
                        c_rescue += 1
                    c_err.append(abs((ga - gc) - r.predicted.get(iid, 0.0)))
                # causal fulfilment effect of THIS epoch's decision
                c_gain.append((1.0 if ga >= 0 else 0.0)
                              - (1.0 if gc >= 0 else 0.0))

        # ---- prediction quality -----------------------------------------
        errs, rel = [], []
        for r in rs:
            if r.n_writes == 0:
                continue
            for iid, p in r.predicted.items():
                if iid in r.g_after and iid in r.g_before:
                    obs = r.g_after[iid] - r.g_before[iid]
                    errs.append(abs(obs - p))
                    denom = max(abs(obs), 1e-3)
                    rel.append(min(abs(obs - p) / denom, 10.0))
        mae = float(np.mean(errs)) if errs else float("nan")

        # ---- search quality (what the DRL top-K gave up) -----------------
        gaps = [max(r.best_utility - r.utility, 0.0) for r in rs
                if np.isfinite(r.best_utility)]
        search_regret = float(np.mean(gaps)) if gaps else float("nan")
        optimal_frac = (float(np.mean([g <= 1e-9 for g in gaps]))
                        if gaps else float("nan"))

        n = len(rs)
        prop = sum(r.n_proposed for r in rs)
        out = {
            "epochs": n,
            "IF": IF,                                 # PRIMARY
            "mean_shortfall": mean_shortfall,         # secondary, lower=better
            "shortfall_when_violated": shortfall_when_violated,
            "wIF": float(wif),                        # legacy, pi-weighted
            "worst_intent_fulfilment": float(worst),
            "mean_margin": mean_margin,
            "mean_worst_margin": worst_margin,
            "safety_crossings": int(crossings),
            "causal_crossings": int(c_cross) if have_cf else float("nan"),
            "causal_rescues": int(c_rescue) if have_cf else float("nan"),
            "causal_prediction_mae": (float(np.mean(c_err)) if c_err
                                      else float("nan")),
            "causal_IF_gain": (float(np.mean(c_gain)) if c_gain
                               else float("nan")),
            "safety_crossing_rate": float(crossing_rate),
            "zero_crossings": int(zero_crossings),
            "c1_violations": int(sum(r.c1_violations for r in rs)),
            "c2_violations": int(sum(r.c2_violations for r in rs)),
            "writes_total": int(sum(r.n_writes for r in rs)),
            "writes_per_epoch": float(np.mean([r.n_writes for r in rs])),
            "proposals_total": int(prop),
            "admit_frac": float(sum(r.n_writes for r in rs) / max(prop, 1)),
            "override_frac": float(sum(r.n_override for r in rs)
                                   / max(prop, 1)),
            "reject_frac": float(sum(r.n_reject for r in rs) / max(prop, 1)),
            "abstain_frac": float(sum(r.n_abstain for r in rs) / max(prop, 1)),
            "prediction_mae": mae,
            "prediction_rel_err": float(np.mean(rel)) if rel else float("nan"),
            "search_regret": search_regret,
            "optimal_portfolio_frac": optimal_frac,
            "candidates_mean": float(np.mean([r.candidates for r in rs])),
            "latency_ms_mean": float(np.mean([r.latency_ms for r in rs])),
            "latency_ms_p95": float(np.percentile([r.latency_ms for r in rs],
                                                  95)),
            "latency_ms_max": float(np.max([r.latency_ms for r in rs])),
            "probes": int(sum(r.probe for r in rs)),
            "reconfig_prb_total": float(sum(r.reconfig_prb for r in rs)),
            "unknown_entries_mean": float(np.mean(
                [r.unknown_entries for r in rs])),
        }
        for iid in iids:
            out[f"fulfilment_{iid}"] = per_intent[iid]
        # cell-level averages, so the RAN story travels with the metrics
        cell_keys = sorted({k for r in rs for k in r.kpm_cell})
        for k in cell_keys:
            vals = [r.kpm_cell[k] for r in rs if k in r.kpm_cell]
            if vals:
                out[f"cell_{k}"] = float(np.mean(vals))
        return out

    # ------------------------------------------------------------------
    def adaptation_delay(self, change_epoch: int,
                         tolerance: float = 0.05,
                         window: int = 10) -> float:
        """Epochs from a known plant change until prediction error recovers.

        Defined against the method's OWN pre-change error, so a method
        that was always inaccurate is not rewarded for staying that way:
        recovery means returning to within ``tolerance`` of the error it
        had before the change, and a method whose error never recovers
        gets ``inf``.
        """
        rs = self.records
        pre = [r for r in rs if r.epoch < change_epoch and r.n_writes > 0]
        if not pre:
            return float("nan")

        def err(recs):
            e = []
            for r in recs:
                for iid, p in r.predicted.items():
                    if iid in r.g_after and iid in r.g_before:
                        e.append(abs((r.g_after[iid] - r.g_before[iid]) - p))
            return float(np.mean(e)) if e else float("nan")

        base = err(pre[-40:])
        if not np.isfinite(base):
            return float("nan")
        post = [r for r in rs if r.epoch >= change_epoch]
        for k in range(len(post) - window):
            e = err(post[k:k + window])
            if np.isfinite(e) and e <= base * (1.0 + tolerance):
                return float(post[k].epoch - change_epoch)
        return float("inf")

    # ------------------------------------------------------------------
    def rows(self) -> List[Dict]:
        return [r.as_row() for r in self.records]

    def timeseries(self, key: str) -> np.ndarray:
        return np.array([getattr(r, key) for r in self.records], dtype=float)


# ---------------------------------------------------------------------------
def compare(summaries: Dict[str, Dict[str, float]],
            reference: str = "intact-ra") -> Dict[str, Dict[str, float]]:
    """Deltas of every method against a reference method."""
    ref = summaries.get(reference)
    if not ref:
        return {}
    out = {}
    for name, s in summaries.items():
        if name == reference:
            continue
        out[name] = {
            "d_wIF": s.get("wIF", float("nan")) - ref.get("wIF", float("nan")),
            "d_safety_crossings": (s.get("safety_crossings", 0)
                                   - ref.get("safety_crossings", 0)),
            "d_worst_intent": (s.get("worst_intent_fulfilment", float("nan"))
                               - ref.get("worst_intent_fulfilment",
                                         float("nan"))),
            "d_writes_per_epoch": (s.get("writes_per_epoch", float("nan"))
                                   - ref.get("writes_per_epoch",
                                             float("nan"))),
            "d_latency_ms": (s.get("latency_ms_mean", float("nan"))
                             - ref.get("latency_ms_mean", float("nan"))),
        }
    return out


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000,
                 alpha: float = 0.05,
                 seed: int = 0) -> Dict[str, float]:
    """Percentile bootstrap CI over per-seed values."""
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n": 0}
    if v.size == 1:
        return {"mean": float(v[0]), "lo": float(v[0]), "hi": float(v[0]),
                "n": 1}
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(v, v.size, replace=True).mean()
                      for _ in range(int(n_boot))])
    return {"mean": float(v.mean()),
            "lo": float(np.percentile(boots, 100 * alpha / 2)),
            "hi": float(np.percentile(boots, 100 * (1 - alpha / 2))),
            "sd": float(v.std(ddof=1)), "n": int(v.size)}


def paired_delta_ci(a: Sequence[float], b: Sequence[float],
                    n_boot: int = 2000, alpha: float = 0.05,
                    seed: int = 0) -> Dict[str, float]:
    """Paired bootstrap on a - b, for two methods run on the SAME seeds.

    Paired, because the seed controls the plant realisation: an unpaired
    interval over five seeds would be dominated by between-seed variance
    that both methods experienced identically.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = min(a.size, b.size)
    if n == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n": 0}
    d = a[:n] - b[:n]
    res = bootstrap_ci(d, n_boot=n_boot, alpha=alpha, seed=seed)
    res["positive_frac"] = float(np.mean(d > 0))
    return res
