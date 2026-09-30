"""Method registry: every controller the benchmark compares, in one place.

A "method" is a small declarative record saying which of the four
interchangeable pieces a run uses:

    sensitivity   where the slope estimates s_{r,p,i} come from
    regime        how the operating regime that indexes them is labelled
    candidates    how the candidate portfolio set is produced
    supervisor    whether the slow agentic loop runs

Everything else -- the plant, the xApps, the proposal stream, the C1/C2
feasibility tests, the robust safety floor, the execution order -- is
IDENTICAL across methods by construction.  That is what makes the
comparison a comparison rather than a demonstration.

Two of the entries are deliberately not proposals:

  ALL_ACCEPT / ALL_REJECT  are the degenerate controllers.  If either of
      them wins, the scenario has not established that arbitration is
      necessary, and gates.py refuses the scenario.
  ORACLE                   uses true finite-difference slopes measured on
      RAN clones.  It is a YARDSTICK, not a method: it says how much of
      the remaining gap is attributable to estimation error at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Method:
    """One controller configuration."""

    name: str
    label: str                      # for figures and tables
    sensitivity: str                # static | online | oracle | none
    regime: str                     # cell | tenant_load | learned | oracle | fixed
    candidates: str                 # exhaustive | drl | all | none | priority | greedy
    supervisor: bool = False
    learn_context: bool = False     # run the GCN context head
    train_policy: bool = False      # run PPO updates
    channel_aware: bool = False     # two-axis regime space
    overrides: Dict = field(default_factory=dict)
    family: str = "controller"      # controller | baseline | ablation | bound
    note: str = ""

    @property
    def is_agentic(self) -> bool:
        return self.sensitivity == "online" or self.learn_context \
            or self.candidates == "drl"


# ---------------------------------------------------------------------------
# The two degenerate controllers.  Their whole purpose is to be beaten.
# ---------------------------------------------------------------------------
ALL_ACCEPT = Method(
    name="all-accept", label="All-accept", family="baseline",
    sensitivity="none", regime="fixed", candidates="all",
    note="Executes every proposal every epoch, conflicts resolved by claim "
         "id. Pays the full reconfiguration transient, so it loses whenever "
         "writing is not free.")

ALL_REJECT = Method(
    name="all-reject", label="All-reject", family="baseline",
    sensitivity="none", regime="fixed", candidates="none",
    note="Executes nothing. Wins only if the plant is stationary, which is "
         "exactly the degenerate case the scenario gates exclude.")

STATIC_PRIORITY = Method(
    name="static-priority", label="Static priority", family="baseline",
    sensitivity="none", regime="fixed", candidates="priority",
    note="Prior art: rank claims by tenant weight x claim priority r_j, "
         "admit the highest-ranked conflict-free set. No model, no safety "
         "prediction, no uncertainty.")

# B3, exactly as the original codebase defines it (intact/experiment.py,
# mode "value_only"; registry label "B3 value arbitration"):
#
#     Vq[j] = sum_i  s(regime_now, param_j, i) * dnu_j
#
# i.e. proposal-aware, UNWEIGHTED signed-margin value, read from the table
# at ONE CELL-LEVEL regime shared by every intent, followed by selection of
# an admissible (C1/C2) subset and direct ADMIT -- there is no second
# safety stage.  The single point of difference from INTACT-RA that
# matters is the regime: B3 reads one cell-wide column of slopes for every
# tenant, INTACT-RA reads each tenant's own column.
#
# An earlier revision of this package got B3 wrong twice: first by
# substituting static-priority ranking, then by giving B3 the per-tenant
# regime, which erased precisely the distinction that defines it.
B3_SIGNED_MARGIN = Method(
    name="b3", label="B3 value arbitration", family="baseline",
    sensitivity="static", regime="cell", candidates="b3",
    note="Unweighted signed-margin value at ONE cell-level regime; best "
         "C1/C2-admissible subset; admitted directly, no safety stage.")

B3_ORACLE = Method(
    name="b3-oracle", label="B3 + true slopes", family="bound",
    sensitivity="oracle", regime="oracle", candidates="b3",
    note="B3's selection rule reading the ORACLE table. Isolates slope "
         "quality at B3's write rate.")

B3_SCRAMBLED = Method(
    name="b3-scrambled", label="B3 + scrambled slopes", family="bound",
    sensitivity="scrambled", regime="cell", candidates="b3",
    note="B3's selection rule reading a table with the right magnitude "
         "distribution and the wrong assignments. If this matches B3, the "
         "scenario cannot discriminate sensitivity knowledge at all.")

GREEDY_VALUE = Method(
    name="greedy-value", label="Greedy value", family="baseline",
    sensitivity="static", regime="tenant_load", candidates="greedy",
    note="Greedy by predicted weighted gain, no robust safety bound. "
         "Isolates how much of INTACT-RA's benefit comes from the "
         "uncertainty-aware admissibility test rather than from ranking.")

# ---------------------------------------------------------------------------
# The frozen prior method.  This is INTACT-RA exactly as published:
# offline sensitivity table, per-tenant MEASURED LOAD regime, exhaustive
# enumeration, deterministic safety arbiter.
# ---------------------------------------------------------------------------
INTACT_RA = Method(
    name="intact-ra", label="INTACT-RA", family="controller",
    sensitivity="static", regime="tenant_load", candidates="exhaustive",
    note="Frozen prior method. The sensitivity table is calibrated offline "
         "with mobility disabled and never updated; the regime that indexes "
         "it is a per-tenant measured LOAD label, which is blind to a "
         "geometry change at constant load by construction.")

INTACT_RA_CELL = Method(
    name="intact-ra-cell", label="INTACT-RA (cell regime)", family="ablation",
    sensitivity="static", regime="cell", candidates="exhaustive",
    note="Ablation: one cell-average regime label instead of per-tenant. "
         "Quantifies what the per-tenant label was already worth.")

# ---------------------------------------------------------------------------
# The proposal.
# ---------------------------------------------------------------------------
INTACT_RA_AGENTIC = Method(
    name="intact-ra-agentic", label="INTACT-RA-Agentic", family="controller",
    sensitivity="kalman", regime="learned", candidates="drl",
    supervisor=True, learn_context=True, train_policy=True,
    channel_aware=True,
    note="Online recursive cross-sensitivity with held-out promotion and "
         "rollback; GCN context head estimating resource pressure and "
         "coverage index; claim-aware PPO top-K candidate selection; "
         "agentic supervisor issuing bounded calibration probes.")

# The first agentic estimator (joint RLS with held-out promotion), kept so
# its earlier results stay reproducible.  Superseded because it published
# uncertainty from plant-noise-dominated residuals without normalising by
# dose, which paralysed the robust safety floor (see KalmanSensitivity).
AG_RLS_V1 = Method(
    name="agentic-rls-v1", label="Agentic (RLS v1)", family="ablation",
    sensitivity="online", regime="learned", candidates="drl",
    supervisor=True, learn_context=True, train_policy=True,
    channel_aware=True,
    note="First agentic estimator; superseded by the Kalman tracker.")

AG_WITH_ANALYTIC = Method(
    name="agentic+analyticPrior", label="Agentic + analytic prior",
    family="ablation",
    sensitivity="kalman", regime="learned", candidates="drl",
    supervisor=True, learn_context=True, train_policy=True,
    channel_aware=True,
    overrides={"agent.kalman": {"analytic_prior": True}},
    note="Adds the physics-informed own-tenant served-ratio prior to the "
         "Kalman tracker. Tested during development and switched off by "
         "default because it lowered fulfilment (0.661 vs 0.740 on the "
         "development seeds, without the twin); kept so the choice is "
         "reproducible.")

AG_NO_TWIN = Method(
    name="agentic-noTwin", label="Agentic - digital twin", family="ablation",
    sensitivity="kalman", regime="learned", candidates="drl",
    supervisor=True, learn_context=True, train_policy=True,
    channel_aware=True,
    overrides={"agent.twin": {"enabled": False}},
    note="Kalman tracker learning from live writes only, without the "
         "digital-twin calibration loop.")

INTACT_RA_PERTENANT = Method(
    name="intact-ra-pertenant-sweep", label="INTACT-RA (per-tenant sweep)",
    family="ablation",
    sensitivity="static_pt", regime="tenant_load", candidates="exhaustive",
    note="Frozen INTACT-RA in every respect except its offline table, which "
         "is calibrated by scaling ONE tenant's load at a time (others at "
         "nominal) instead of all tenants together. Measures how much of "
         "INTACT-RA's shortfall is its calibration rather than its being "
         "frozen.")

AG_LEAN = Method(
    name="agentic-lean", label="INTACT-RA-Agentic (lean)", family="ablation",
    sensitivity="kalman", regime="tenant_load", candidates="exhaustive",
    supervisor=True, learn_context=False, train_policy=False,
    channel_aware=True,
    note="INTACT-RA-Agentic without BOTH the PPO top-K proposer (every "
         "portfolio scored exhaustively) and the learned GCN context (the "
         "per-tenant load regime instead). Keeps the Kalman sensitivity "
         "tracker, the digital-twin calibration loop and the supervisor's "
         "calibration probes -- the components the ablations found carry "
         "the gain. Tests whether a lighter agent loses anything.")

# ---- ablations: remove exactly one component at a time --------------------
AG_NO_ONLINE = Method(
    name="agentic-noOnline", label="Agentic - online table", family="ablation",
    sensitivity="static", regime="learned", candidates="drl",
    supervisor=False, learn_context=True, train_policy=True,
    channel_aware=True,
    note="Keeps the learned regime and the DRL proposer, restores the frozen "
         "table. Isolates the value of online slope estimation.")

AG_NO_LEARNED_REGIME = Method(
    name="agentic-noLearnedRegime", label="Agentic - learned regime",
    family="ablation",
    sensitivity="kalman", regime="tenant_load", candidates="drl",
    supervisor=True, learn_context=False, train_policy=True,
    note="Keeps online learning and the DRL proposer, restores the measured "
         "load label. Isolates the value of the channel-aware context head.")

AG_NO_DRL = Method(
    name="agentic-noDRL", label="Agentic - DRL top-K", family="ablation",
    sensitivity="kalman", regime="learned", candidates="exhaustive",
    supervisor=True, learn_context=True, train_policy=False,
    channel_aware=True,
    note="Keeps online learning and the learned regime, restores exhaustive "
         "enumeration. Isolates the LATENCY cost the DRL proposer buys down "
         "and whether it costs any decision quality to do so.")

AG_NO_PROBE = Method(
    name="agentic-noProbe", label="Agentic - probes", family="ablation",
    sensitivity="kalman", regime="learned", candidates="drl",
    supervisor=False, learn_context=True, train_policy=True,
    channel_aware=True,
    note="Keeps everything except the supervisor's active calibration "
         "probes. Isolates how much the agent needed to EXPLORE versus how "
         "much it could learn from writes it was going to make anyway.")

# ---------------------------------------------------------------------------
ORACLE = Method(
    name="oracle", label="Oracle slopes (bound)", family="bound",
    sensitivity="oracle", regime="oracle", candidates="exhaustive",
    note="Upper bound, not a method: true local slopes by paired finite "
         "differences on RAN clones with common random numbers. The gap "
         "between the agentic method and this bound is the part of the "
         "problem that better estimation could still buy.")


ORACLE_ROLLOUT = Method(
    name="oracle-rollout", label="Oracle-R (rollout bound)", family="bound",
    sensitivity="static", regime="tenant_load", candidates="rollout",
    note="Chooses among the SAME candidate portfolios INTACT-RA considers by "
         "simulating each forward on forks of the live plant, averaged over "
         "independent futures. The attainable value of perfect knowledge of "
         "the plant's current response, independent of any slope model.")

# ---------------------------------------------------------------------------
_ALL: List[Method] = [
    ALL_ACCEPT, ALL_REJECT, STATIC_PRIORITY, B3_SIGNED_MARGIN,
    B3_ORACLE, B3_SCRAMBLED, GREEDY_VALUE,
    INTACT_RA, INTACT_RA_CELL, INTACT_RA_PERTENANT,
    INTACT_RA_AGENTIC, AG_RLS_V1, AG_WITH_ANALYTIC, AG_NO_TWIN, AG_LEAN,
    AG_NO_ONLINE, AG_NO_LEARNED_REGIME, AG_NO_DRL, AG_NO_PROBE,
    ORACLE, ORACLE_ROLLOUT,
]

REGISTRY: Dict[str, Method] = {m.name: m for m in _ALL}

# The set used for the headline claim (requirement 3).
HEADLINE = ("all-accept", "all-reject", "static-priority", "b3",
            "intact-ra", "intact-ra-agentic", "oracle")

# The set used for the component analysis.
ABLATIONS = ("intact-ra-agentic", "agentic-noOnline",
             "agentic-noLearnedRegime", "agentic-noDRL", "agentic-noProbe")

# Everything worth running in a full benchmark.
FULL = tuple(m.name for m in _ALL)


def get(name: str) -> Method:
    if name not in REGISTRY:
        raise KeyError(f"unknown method {name!r}; known: "
                       f"{', '.join(sorted(REGISTRY))}")
    return REGISTRY[name]


def resolve(names: Optional[str]) -> List[Method]:
    """Turn a CLI string into a list of methods.

    Accepts a comma-separated list of method names, or one of the group
    aliases ``headline``, ``ablations``, ``full``.
    """
    if not names or names == "headline":
        return [get(n) for n in HEADLINE]
    if names == "ablations":
        return [get(n) for n in ABLATIONS]
    if names in ("full", "all"):
        return [get(n) for n in FULL]
    return [get(n.strip()) for n in names.split(",") if n.strip()]


def describe() -> str:
    lines = ["Methods available:", ""]
    for fam in ("baseline", "controller", "ablation", "bound"):
        members = [m for m in _ALL if m.family == fam]
        if not members:
            continue
        lines.append(f"  [{fam}]")
        for m in members:
            lines.append(f"    {m.name:<26} {m.label}")
            lines.append(f"    {'':<26} sens={m.sensitivity} "
                         f"regime={m.regime} cand={m.candidates} "
                         f"sup={int(m.supervisor)}")
        lines.append("")
    return "\n".join(lines)
