"""
intact_agentic/types.py
=======================
The six objects the whole framework is built from.

Everything downstream -- the RAN, the sensitivity model, the arbiter, the
DRL agent, the reporting -- is expressed in terms of ONLY these types.  If
you read this file you can read the rest of the package.

    Tenant      a customer of the neutral host.  Owns intents and an
                envelope of physical resource it may occupy.
    Intent      one contractual promise of one tenant, expressed as a
                target on a named KPI with a direction.
    Claim       "xApp x is permitted to write parameter p at scope s".
                THIS IS THE UNIT WE SCHEDULE.  Not the xApp -- an xApp
                that writes three knobs, only one of which clashes, must
                not lose the two harmless ones.
    Write       a concrete pending control message from a claim.
    Decision    the full audit record of what the arbiter did with a
                write and why.
    Portfolio   a set of claims authorised together in one epoch.

Design notes that matter and are easy to get wrong
--------------------------------------------------
* ``Claim.max_step_frac`` is the TRUST REGION.  The arbiter predicts a
  margin change with a first-order (linear) model whose error grows like
  |dnu|^2.  Without a cap the biggest writes -- exactly the ones that
  matter -- are judged by a linearisation far outside the neighbourhood
  it was fitted in.
* ``Intent.epsilon`` is the SAFETY FLOOR in margin units, not in KPI
  units.  g >= epsilon means "comfortably keeping the promise".
* ``Intent.clip`` bounds |g|.  "Three times worse than promised" is not
  meaningfully worse than "twice as bad" -- both are a total breach --
  but unclipped it would dominate every weighted sum in the framework.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Small enums that accept sloppy YAML casing without complaint.
# ---------------------------------------------------------------------------
class _CaseInsensitive(str, Enum):
    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            for m in cls:
                if m.value.lower() == value.lower():
                    return m
        return None


class Scope(_CaseInsensitive):
    """Who a write can possibly affect."""
    UE = "UE"          # one tenant's UEs only
    SLICE = "slice"    # one slice (can still leak through the shared PHY)
    CELL = "cell"      # the whole cell -> every tenant.  HOST-OWNED ONLY.


class Kind(_CaseInsensitive):
    """Does the value RESERVE capacity or BOUND it?

    REGULATIVE  a ceiling.  Two caps of 82 do not reserve 164 PRBs, so the
                per-tenant envelope constraint is vacuous for these.
    ALLOCATIVE  a reservation.  The value IS the commitment, so the sum
                over a tenant's co-authorised allocative claims must stay
                inside its envelope.
    """
    REGULATIVE = "regulative"
    ALLOCATIVE = "allocative"


class Direction(_CaseInsensitive):
    HIGHER_BETTER = "higher_better"
    LOWER_BETTER = "lower_better"


class Outcome(str, Enum):
    ADMIT = "admit"        # executed at the requested (grid-projected) value
    OVERRIDE = "override"  # executed at a different, safer value
    REJECT = "reject"      # not executed
    ABSTAIN = "abstain"    # not executed because the model could not judge it


# ---------------------------------------------------------------------------
@dataclass
class Tenant:
    tid: str
    omega: float = 1.0                 # commercial weight (contract term)
    rho_min: float = 0.8               # contracted fulfilment floor
    envelope: Dict[str, float] = field(default_factory=dict)   # {"PRB": 30}
    is_host: bool = False
    arrives_at_epoch: int = 0          # >0 => mid-run arrival

    def env(self, resource: str) -> float:
        return float(self.envelope.get(resource, float("inf")))


@dataclass
class Intent:
    iid: str
    tenant: str
    kpi: str                           # key into the RAN KPM dict
    target: float
    direction: Direction
    pi_class: float = 1.0              # contractual priority class
    epsilon: float = 0.02              # safety floor, in MARGIN units
    clip: float = 1.5                  # |g| bound
    weight: float = 1.0                # utility weight inside the arbiter
    arrives_at_epoch: int = 0

    @property
    def sign(self) -> float:
        return 1.0 if self.direction == Direction.HIGHER_BETTER else -1.0


@dataclass
class Claim:
    jid: str
    xapp: str
    tenant: str
    param: str
    scope: Scope
    kind: Kind
    domain: Tuple[float, float]
    step: float
    resource: Optional[str] = None     # allocative claims only, e.g. "PRB"
    d_bar: float = 0.0                 # max resource demand of this claim
    max_step_frac: float = 0.25        # trust region as a fraction of width
    r_j: float = 0.5                   # contracted actuation rate (audit only)
    arrives_at_epoch: int = 0

    # ---- derived helpers ------------------------------------------------
    @property
    def width(self) -> float:
        return float(self.domain[1] - self.domain[0])

    def grid(self) -> List[float]:
        lo, hi = self.domain
        n = int(round((hi - lo) / self.step))
        return [round(lo + k * self.step, 9) for k in range(n + 1)]

    def project(self, value: float) -> float:
        """Snap a requested value onto the legal grid inside the domain."""
        lo, hi = self.domain
        v = min(max(float(value), lo), hi)
        return round(round((v - lo) / self.step) * self.step + lo, 9)

    def trust_clip(self, value: float, nu_old: float) -> float:
        """Clamp a requested value to the one-epoch trust region."""
        cap = self.max_step_frac * self.width
        v = min(max(float(value), nu_old - cap), nu_old + cap)
        return self.project(v)


@dataclass
class Write:
    jid: str
    param: str
    nu_req: float
    nu_old: float
    epoch: int


@dataclass
class Decision:
    """Everything needed to audit one write after the fact."""
    jid: str
    param: str
    nu_old: float
    nu_req: float
    nu_star: float
    outcome: Outcome
    reason: str = ""
    implicated: List[str] = field(default_factory=list)
    predicted_dg: Dict[str, float] = field(default_factory=dict)
    sigma: Dict[str, float] = field(default_factory=dict)
    regime: Dict[str, str] = field(default_factory=dict)
    epoch: int = 0

    @property
    def dnu(self) -> float:
        return float(self.nu_star - self.nu_old)

    @property
    def executed(self) -> bool:
        return self.outcome in (Outcome.ADMIT, Outcome.OVERRIDE)

    def as_row(self) -> Dict:
        return {
            "epoch": self.epoch, "claim_id": self.jid, "param": self.param,
            "nu_old": self.nu_old, "nu_req": self.nu_req,
            "nu_star": self.nu_star, "dnu": self.dnu,
            "outcome": self.outcome.value, "reason": self.reason,
            "implicated": "|".join(self.implicated),
            "executed": int(self.executed),
        }


# A portfolio is just a frozen set of claim ids; a named type makes the
# signatures readable.
Portfolio = Tuple[str, ...]


def portfolio_key(claims: Sequence[str]) -> Portfolio:
    return tuple(sorted(claims))


# ---------------------------------------------------------------------------
@dataclass
class Registry:
    """The live set of tenants, intents and claims at the current epoch.

    A mid-run tenant arrival is handled by ACTIVATING rows that already
    exist in the configuration but whose ``arrives_at_epoch`` has not yet
    been reached.  Keeping them in one registry (rather than mutating
    dictionaries) means every method -- baseline or proposed -- sees the
    identical arrival schedule, which is what makes the comparison fair.
    """
    tenants: Dict[str, Tenant]
    intents: Dict[str, Intent]
    claims: Dict[str, Claim]

    def active_tenants(self, epoch: int) -> Dict[str, Tenant]:
        return {k: v for k, v in self.tenants.items()
                if epoch >= v.arrives_at_epoch}

    def active_intents(self, epoch: int) -> Dict[str, Intent]:
        return {k: v for k, v in self.intents.items()
                if epoch >= v.arrives_at_epoch
                and epoch >= self.tenants[v.tenant].arrives_at_epoch}

    def active_claims(self, epoch: int) -> Dict[str, Claim]:
        return {k: v for k, v in self.claims.items()
                if epoch >= v.arrives_at_epoch
                and epoch >= self.tenants[v.tenant].arrives_at_epoch}

    def arrival_epochs(self) -> List[int]:
        e = {t.arrives_at_epoch for t in self.tenants.values()}
        e |= {i.arrives_at_epoch for i in self.intents.values()}
        e |= {c.arrives_at_epoch for c in self.claims.values()}
        return sorted(x for x in e if x > 0)
