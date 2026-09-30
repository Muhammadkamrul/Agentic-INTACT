"""Digital-twin calibration loop for INTACT-RA-Agentic.

WHY IT EXISTS
-------------
A safety-constrained arbiter does not excite its own knobs enough to learn
sensitivities from its own writes: the states where learning matters most
are exactly those where the robust safety floor refuses to act on an
uncertain slope, so the evidence never arrives.  Measured on S16: the
Kalman tracker learning only from live writes reached 0.740 IF against
0.818 for an oracle that measures true slopes every epoch.

The oracle measures slopes by paired finite differences on an exact COPY
of the live plant.  This module is the deployable counterpart: the same
measurement on a digital twin whose fidelity is deliberately imperfect,
and whose imperfections are explicit, configurable and reported:

  * no knowledge of the future -- the twin runs on its OWN random stream,
    not the plant's (an exact clone would replay the real future noise);
  * state-estimation error -- every UE position is perturbed by N(0, s_pos)
    per axis, since a real twin infers positions from measurement reports;
  * model mismatch -- the twin's thermal noise and every neighbour cell's
    load are wrong by errors drawn ONCE per run (a miscalibrated model does
    not re-draw its error every epoch);
  * lag -- the experiment folds twin measurements in at the END of an
    epoch, so they inform decisions only from the next one.

Twin compute runs in the slow calibration loop and is reported separately
from the near-real-time decision latency.

The ladder this creates is the point of the design:
    frozen table (INTACT-RA)  <  imperfect twin (Agentic)  <  exact copy (oracle)
and twin fidelity is a scope condition to be reported with any result.
"""
from __future__ import annotations

import time
from typing import Dict, Iterable, List, Tuple

import numpy as np

from ..arbiter.margins import margin


class DigitalTwin:
    def __init__(self, cfg: Dict, claims, intents, seed: int):
        tw = (cfg.get("agent", {}) or {}).get("twin", {}) or {}
        self.claims = claims
        self.intents = intents
        self.replicates = int(tw.get("replicates", 2))
        self.settle = int(tw.get("settle_slots", 8))
        self.measure = int(tw.get("measure_slots", 24))
        self.frac = float(tw.get("probe_frac", 0.5))
        self.pos_sigma = float(tw.get("pos_sigma_m", 25.0))
        rng = np.random.default_rng(int(seed) ^ 0x7A1)
        # model mismatch: drawn ONCE per run
        self.nf_err_db = float(rng.normal(0.0, float(tw.get("nf_err_sd_db", 1.0))))
        self.load_factor = float(np.exp(rng.normal(0.0, float(
            tw.get("load_err_sd", 0.2)))))
        self._rng = rng
        self.seconds = 0.0
        self.rollouts = 0
        self.calls = 0

    def describe(self) -> Dict:
        return {"pos_sigma_m": self.pos_sigma, "nf_err_db": self.nf_err_db,
                "neighbour_load_factor": self.load_factor,
                "replicates": self.replicates,
                "settle_slots": self.settle, "measure_slots": self.measure}

    # ------------------------------------------------------------------
    def _twin_of(self, ran, stream_seed: int):
        tw = ran.clone()
        tw.reseed(stream_seed)                       # its OWN future
        ch = tw.channel
        ch.noise_prb_dbm = float(ch.noise_prb_dbm) + self.nf_err_db
        for site in getattr(ch, "sites", [])[1:]:
            if hasattr(site, "load"):
                site.load = float(np.clip(site.load * self.load_factor,
                                          0.0, 1.0))
        return tw

    def _perturb_positions(self, tw, rs: np.random.Generator) -> None:
        for g in tw.mobility.groups.values():
            jit = rs.normal(0.0, self.pos_sigma, g.xy.shape)
            g.xy = g.xy + jit
            if getattr(g, "ref", None) is not None:
                g.ref = g.ref + jit

    def _rollout(self, base, param, value, rs_seed):
        tw = base.clone()
        tw.reseed(rs_seed)
        tw.apply(param, value)
        tw._reconfig_left = {t: 0 for t in tw.slices}
        tw._cell_reconfig_left = 0
        tw.step(self.settle, record=False)
        self.rollouts += 1
        return tw.step(self.measure, record=False)

    # ------------------------------------------------------------------
    def measure_slopes(self, ran, params: Iterable[str], regimes, epoch: int
                       ) -> List[Tuple[str, str, str, float, float]]:
        """[(regime, intent, param, slope, variance)] from the twin."""
        t0 = time.perf_counter()
        self.calls += 1
        base = self._twin_of(ran, 90_001 * (epoch + 1))
        self._perturb_positions(base, self._rng)
        cur = dict(base.commanded_controls())
        out = []
        for prm in sorted(set(params)):
            c = next((c for c in self.claims.values() if c.param == prm), None)
            if c is None:
                continue
            d = self.frac * c.max_step_frac * c.width
            lo, hi = c.domain
            x0 = float(cur.get(prm, 0.5 * (lo + hi)))
            plus, minus = min(x0 + d, hi), max(x0 - d, lo)
            if plus - minus < 1e-9:
                continue
            per = {i: [] for i in self.intents}
            for r in range(self.replicates):
                sd = 7_919 * (epoch + 1) + 104_729 * r
                kp = self._rollout(base, prm, plus, sd)
                km = self._rollout(base, prm, minus, sd)
                for iid, it in self.intents.items():
                    gp, gm = margin(it, kp), margin(it, km)
                    if gp is not None and gm is not None:
                        per[iid].append((gp - gm) / (plus - minus))
            for iid, v in per.items():
                if not v:
                    continue
                z = float(np.mean(v))
                var = float(np.var(v, ddof=1) / len(v)) if len(v) > 1 \
                    else float(z * z + 1e-6)
                out.append((regimes.of_intent(iid), iid, prm, z,
                            max(var, 1e-8)))
        self.seconds += time.perf_counter() - t0
        return out
