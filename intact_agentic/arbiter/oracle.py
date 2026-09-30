"""
intact_agentic/arbiter/oracle.py
================================
The ORACLE sensitivity model: the true local derivative, measured by
finite differences on a cloned RAN at the current operating point.

NOT A METHOD.  A YARDSTICK.
---------------------------
This is not deployable and is never presented as a competitor.  It reads
the simulator, clones it, perturbs one knob at a time with common random
numbers, and measures what actually happens.  No real RIC can do that:
it would need a perfect digital twin of the live cell and a free
maintenance window every epoch.

It exists for one reason.  When INTACT-RA-Agentic beats INTACT-RA, the
obvious next question is "how much of the remaining gap is estimation
error and how much is the arbiter's own design?".  Without an upper bound
that question cannot be answered, and a reviewer is entitled to ask it.
Running the identical arbiter with perfect slopes separates the two:

    ORACLE - INTACT-RA-Agentic   = what better estimation could still buy
    INTACT-RA-Agentic - INTACT-RA = what the online learning actually
                                    bought

COST CONTROL
------------
A full refresh is  |params| x 2  cloned rollouts.  It is therefore
refreshed every ``refresh_every`` epochs (default 10) and cached; the
cache staleness is itself bounded and reported, so the yardstick's own
error is visible rather than assumed away.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..types import Claim, Intent
from .margins import margin
from .sensitivity import Entry, Key, SensitivityBase


class OracleSensitivity(SensitivityBase):
    """True local slopes from paired finite differences on a RAN clone."""

    def __init__(self, cfg: Dict, claims: Dict[str, Claim],
                 intents: Dict[str, Intent]):
        o = (cfg.get("oracle", {}) or {})
        self.cfg = cfg
        self.claims = claims
        self.intents = intents
        self.refresh_every = int(o.get("refresh_every", 10))
        self.settle_slots = int(o.get("settle_slots", 8))
        self.measure_slots = int(o.get("measure_slots", 12))
        self.frac = float(o.get("probe_frac", 0.5))
        self.replicates = int(o.get("replicates", 2))
        self.default_sigma = float(o.get("default_sigma", 0.01))
        # (param, intent) -> median offline standard error across regimes;
        # set by attach_prior() so the oracle shares INTACT-RA's uncertainty
        self.prior_sigma = None

        self.sigma_min = float((cfg.get("arbiter", {}) or {})
                               .get("sigma_min", 5e-4))
        self.tab: Dict[Tuple[str, str], float] = {}     # (param, intent)
        self.last_refresh = -10 ** 9
        self.n_refresh = 0
        self.rollouts = 0

    # ------------------------------------------------------------------
    def attach_prior(self, prior) -> None:
        by = {}
        for (r, p, i), e in prior.tab.items():
            by.setdefault((p, i), []).append(float(e.sigma))
        self.prior_sigma = {k: float(np.median(v)) for k, v in by.items()}

    def refresh(self, ran, epoch: int, force: bool = False,
                params=None) -> bool:
        """Re-measure true local slopes on clones of the live plant.

        ``params`` restricts the measurement to the knobs that actually have
        a proposal this epoch: a slope for a knob nobody is asking to move
        cannot change the decision, so skipping it loses nothing and makes a
        PER-EPOCH oracle affordable.  The operating point is the COMMANDED
        control set, consistent with how the arbiter now reasons.
        """
        if not force and epoch - self.last_refresh < self.refresh_every:
            return False
        params = sorted(set(params) if params is not None else
                        {c.param for c in self.claims.values()})
        base = dict(ran.commanded_controls())
        for prm in params:
            c = next(c for c in self.claims.values() if c.param == prm)
            d = self.frac * c.max_step_frac * c.width
            lo, hi = c.domain
            cur = float(base.get(prm, 0.5 * (lo + hi)))
            plus = min(cur + d, hi)
            minus = max(cur - d, lo)
            if plus - minus < 1e-9:
                continue
            # Replicated paired finite differences.  Each replicate clones
            # the SAME live state and re-seeds both arms identically, so the
            # plus and minus arms share every random draw (common random
            # numbers) and only the knob differs.  Averaging replicates
            # reduces the residual noise of a local derivative measured on a
            # stochastic plant; a single replicate was noisy enough that the
            # oracle's "truth" was itself an estimate.
            diffs = {iid: [] for iid in self.intents}
            for r in range(self.replicates):
                seed = 1_000_003 * (epoch + 1) + 7919 * r
                kp = self._rollout(ran, prm, plus, seed)
                km = self._rollout(ran, prm, minus, seed)
                for iid, it in self.intents.items():
                    gp, gm = margin(it, kp), margin(it, km)
                    if gp is not None and gm is not None:
                        diffs[iid].append((gp - gm) / (plus - minus))
            for iid, v in diffs.items():
                if v:
                    self.tab[(prm, iid)] = float(np.mean(v))
        self.last_refresh = epoch
        self.n_refresh += 1
        return True

    def _rollout(self, ran, param: str, value: float,
                 seed: Optional[int] = None) -> Dict:
        probe = ran.clone()
        if seed is not None:
            probe.reseed(int(seed))     # a genuinely independent replicate
        probe.apply(param, value)
        # The reconfiguration transient is a cost of ACTING, not part of
        # the steady-state derivative, so it is cleared before measuring.
        probe._reconfig_left = {t: 0 for t in probe.slices}
        probe._cell_reconfig_left = 0
        probe.step(self.settle_slots, record=False)
        self.rollouts += 1
        return probe.step(self.measure_slots, record=False)

    # ------------------------------------------------------------------
    def get(self, regime: str, param: str, iid: str) -> float:
        v = self.tab.get((param, iid), 0.0)
        return v if abs(v) >= self.sigma_min else 0.0

    def sigma(self, regime: str, param: str, iid: str) -> float:
        """UNCERTAINTY PARITY with the method it is compared against.

        An earlier version returned 0.0.  That did not model "perfect
        knowledge of the current slope" -- it modelled perfect knowledge
        PLUS the belief that the realised outcome carries no uncertainty.
        With sigma = 0 the robust safety floor g_hat - beta * sigma >= eps
        provides no margin at all, so the oracle admitted every write whose
        point estimate cleared the floor.  In every scenario tested it then
        wrote more and caused more safety crossings than the frozen table
        (S1 45 vs 21.5, S7 46.5 vs 36, S8 242.5 vs 120) and scored BELOW
        it -- which was read as "current slopes are worthless" when it was
        really "overconfidence is harmful".

        Even an exact slope does not remove the plant's own noise from the
        realised margin change, and that noise is what the safety floor
        guards against.  So the oracle reports the SAME standard error the
        offline table reports for the same (parameter, intent), and the
        oracle-minus-frozen comparison differs ONLY in the slope point
        estimate.  That is the quantity the diagnostic is meant to isolate.
        """
        if self.prior_sigma is not None:
            return float(self.prior_sigma.get((param, iid), self.default_sigma))
        return self.default_sigma

    def known(self, regime: str, param: str, iid: str) -> bool:
        return (param, iid) in self.tab

    def affected(self, regime: str, param: str, iids: Iterable[str]
                 ) -> List[str]:
        return [i for i in iids if abs(self.get(regime, param, i))
                >= self.sigma_min]

    def diagnostics(self) -> Dict:
        return {"oracle_refreshes": self.n_refresh,
                "oracle_rollouts": self.rollouts,
                "oracle_entries": len(self.tab)}
