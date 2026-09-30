#!/usr/bin/env python3
"""Is the contested resource actually contested?

Run this BEFORE any controller experiment. It answers the one question
that decides whether a scenario can discriminate sensitivity models at
all: when one tenant's quota rises, does anyone pay?

    NET ~ 0      allocation is zero-sum. Granting one tenant's claim costs
                 another, so WHICH write is chosen is the decision, and
                 slope knowledge can have value.
    NET >> 0     free lunch. min(share*quota, demand) means unused quota
                 costs nobody anything, "raise everything" dominates, and
                 wIF will track WRITE RATE rather than slope quality. A
                 benchmark in this regime cannot support a claim about
                 estimation, however good the estimator is.

This probe is what identified the defect in S1 after two sessions of
fruitless controller tuning, so it is worth thirty seconds.
"""
from __future__ import annotations
import argparse
import numpy as np
from _common import ROOT, Log, common_args, load


def main():
    p = common_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--tenant", default=None, help="whose quota to raise")
    p.add_argument("--delta-prb", type=float, default=8.0)
    p.add_argument("--loads", default="1.0,1.3,1.6,2.0",
                   help="offered-load multipliers to sweep")
    a = p.parse_args()

    from intact_agentic.config import load_config, epoch_slots
    from intact_agentic.ran.simulator import RealisticRAN

    log = Log(None, a.quiet)
    base_cfg = load(a)
    tenants = sorted(base_cfg["ran"]["slices"])
    tgt = a.tenant or tenants[0]
    log(f"raising quota_{tgt} by {a.delta_prb:g} PRB and measuring who pays\n")
    log(f"{'load':>10s} {'util':>7s} " +
        " ".join(f"{'d'+t:>9s}" for t in tenants) + f" {'NET':>10s}  verdict")

    verdicts = []
    for m in [float(x) for x in a.loads.split(",")]:
        c = load(a)
        for t in c["ran"]["slices"]:
            c["ran"]["slices"][t]["load_mbps_per_ue"] *= m
            c["ran"]["slices"][t]["load_profile"] = {"levels": [1.0],
                                                     "hold_slots": 10 ** 9}
        c["ran"]["mobility"]["mode"] = "static"
        pre, post = epoch_slots(c)
        ran = RealisticRAN(c)
        ran.reset(77)
        ran.step(80, record=False)
        # PAIRED COUNTERFACTUAL.  Both arms are clones of the SAME state,
        # and clone() carries the RNG, so traffic arrivals, fading and
        # shadowing are identical in the two arms (common random numbers).
        # The ONLY difference is the quota write.  An earlier version of
        # this probe measured the baseline BEFORE cloning and the treated
        # arm 72 slots LATER, so it attributed 72 slots of ordinary plant
        # evolution to the write -- which is how it reported a +1.17 Mb/s
        # "free lunch" from a quota change that, at that load, cannot
        # affect PRB allocation at all.
        prm = f"quota_{tgt}"
        arms = {}
        for name, dv in (("base", 0.0), ("treat", a.delta_prb)):
            pb = ran.clone()
            if dv:
                pb.apply(prm, ran.current_controls()[prm] + dv,
                         scope="slice", tenant=tgt)
            pb._reconfig_left = {t: 0 for t in pb.slices}
            pb._cell_reconfig_left = 0
            pb.step(24, record=False)
            arms[name] = pb.step(48, record=False)
        k0, k1 = arms["base"], arms["treat"]
        d = {t: k1[t]["throughput_mbps"] - k0[t]["throughput_mbps"]
             for t in tenants}
        net = sum(d.values())
        util = k0["_cell"]["prb_util_pct"]
        own = d[tgt]
        # A reallocation is ZERO-SUM when the tenant that gained did so at
        # the others' expense: own gain positive, others' change negative,
        # and the net near zero.  A FREE LUNCH is everyone gaining.  INERT
        # means the write changed nothing, i.e. nobody was backlogged.
        others = net - own
        if abs(own) < 0.02 and abs(others) < 0.02:
            v = "inert (nobody backlogged)"
        elif own > 0 and others < -0.02:
            v = "zero-sum" if abs(net) <= 0.35 * abs(own) else "partly zero-sum"
        elif own > 0 and others >= -0.02:
            v = "FREE LUNCH"
        else:
            v = "harmful"
        verdicts.append((m, util, net, v))
        log(f"{m:10.2f} {util:6.1f}% " +
            " ".join(f"{d[t]:+9.4f}" for t in tenants) +
            f" {net:+10.4f}  {v}")

    log("")
    ok = [x for x in verdicts if x[3].startswith(("zero-sum","partly"))]
    if ok:
        log(f"Usable load multipliers (zero-sum): "
            f"{', '.join(f'{x[0]:g}x at {x[1]:.0f}% util' for x in ok)}")
        log("Set ran.slices.*.load_mbps_per_ue accordingly, then re-run "
            "the oracle diagnostic.")
    else:
        log("NO usable load multiplier in the swept range.")
        log("Below the range the pool does not bind (free lunch); above it "
            "the plant is so constrained that all actuation is harmful and "
            "all-reject wins. Widen --loads or change the plant.")


if __name__ == "__main__":
    main()
