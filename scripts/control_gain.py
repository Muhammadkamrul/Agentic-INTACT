#!/usr/bin/env python3
"""CEILING and FLOOR: is control necessary, and is good control possible?  [AUTO]

Two arbiter-free references, measured on the same plant with common random
numbers across several seeds:

  FLOOR    the provisioned allocation held for the whole run -- exactly
           what all-reject does.  Fulfilment here is what "doing nothing"
           buys.
  CEILING  a DEMAND-TRACKING reference: every epoch, each tenant's
           reservation is set from its ACTUAL current PRB need -- offered
           traffic divided by its current spectral efficiency, with HARQ
           overhead -- scaled to fit the pool and capped by its envelope.
           It knows both the load phase and the geometry, i.e. everything a
           perfect controller could know, and it pays the real
           reconfiguration transient whenever it rewrites a knob.

A benchmark is only convincing if the ceiling is HIGH (good control can
meet most intents) and the gap to the floor is LARGE (doing nothing is
clearly inadequate).  A small gap means every controller is fighting for
crumbs and a single bad write loses to leaving the default alone; a low
ceiling means the scenario is so infeasible that even perfect control
fails intents most of the time.  Neither number depends on INTACT-RA, the
oracle or any learned component, so calibrating a scenario against them
cannot favour any method.
"""
from __future__ import annotations
import argparse
import numpy as np
from _common import ROOT, Log, common_args, load, results_dir, write_json, write_text


def main():
    p = common_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--seeds", default="20260925,20260926,20260927")
    p.add_argument("--from-epoch", type=int, default=100)
    p.add_argument("--overhead", type=float, default=1.20,
                   help="HARQ/burst headroom on the estimated PRB need")
    p.add_argument("--min-change", type=float, default=3.0,
                   help="rewrite a knob only if it moves by at least this many PRB")
    p.add_argument("--drain-epochs", type=float, default=1.0,
                   help="horizon over which the reference drains a backlog")
    p.add_argument("--want-ceiling", type=float, default=0.85)
    p.add_argument("--want-gap", type=float, default=0.20)
    a = p.parse_args()

    from intact_agentic.config import build_registry, epoch_slots, scenario_name
    from intact_agentic.ran.simulator import RealisticRAN
    from intact_agentic.arbiter.margins import margin

    cfg0 = load(a)
    reg = build_registry(cfg0)
    pre, post = epoch_slots(cfg0)
    spe = pre + post
    E = int(cfg0["run"]["epochs"])
    N = float(cfg0["ran"]["n_prb"])
    log = Log(None, a.quiet)
    env = {t: float(v.get("prb", N)) for t, v in
           (cfg0["ran"].get("envelopes", {}) or {}).items()}
    prov = dict(cfg0["ran"]["initial_controls"])
    knobs, dom = {}, {}
    for c in cfg0["claims"]:
        if c["kind"] == "allocative":
            knobs.setdefault(c["tenant"], [])
            if c["param"] not in knobs[c["tenant"]]:
                knobs[c["tenant"]].append(c["param"])
            dom[c["param"]] = tuple(float(x) for x in c["domain"])
    tenants = sorted(knobs)

    def split(tid, total):
        """Spread a tenant's reservation over its knobs in provisioned ratio."""
        ks = knobs[tid]
        base = np.array([max(float(prov.get(k, 1.0)), 1e-6) for k in ks])
        vals = total * base / base.sum()
        return {k: float(np.clip(v, *dom[k])) for k, v in zip(ks, vals)}

    def run(seed, tracking):
        ran = RealisticRAN(cfg0)
        ran.reset(seed)
        per = {i: [] for i in reg.intents}
        ifs = []
        k = None
        for ep in range(E):
            ran.epoch = ep
            if tracking and k is not None:
                # PRB need from offered traffic and the throughput a PRB
                # actually carries after first-transmission errors
                need = {}
                for t in tenants:
                    se = max(float(k[t]["spectral_efficiency"]), 0.05)
                    bler = min(max(float(k[t].get("bler", 0.0)), 0.0), 0.9)
                    mbps_per_prb = se * (1.0 - bler) * float(ran.prb_hz) / 1e6
                    # BACKLOG-AWARE need: arrivals, plus enough to drain
                    # the queue built during a burst within --drain-epochs.
                    # Sizing from arrivals alone leaves backlogs undrained,
                    # which is exactly what drives delay tails.
                    n_ue = max(float(k[t].get("n_ue", 1.0)), 1.0)
                    backlog_mbps = (float(k[t].get("buffer_kb", 0.0)) * 8e3
                                    * n_ue / (a.drain_epochs * spe
                                              * float(ran.slot_s)) / 1e6)
                    need[t] = max((a.overhead * float(k[t]["offered_slice_mbps"])
                                   + backlog_mbps)
                                  / max(mbps_per_prb, 1e-3), 1.0)
                # PROPORTIONAL FILL: the whole pool is always handed out in
                # proportion to need, so when combined need is below the pool
                # every tenant gets headroom in proportion to what it needs
                # and its queue can drain.  Allocating each tenant exactly its
                # arrival rate left backlogs undrained and gave the surplus to
                # an equal split -- which made this "ceiling" lose to the
                # static default.  Envelopes cap each tenant; capacity freed
                # by a cap is redistributed to the others.
                alloc = {t: 0.0 for t in tenants}
                free, active = N, set(tenants)
                for _ in range(len(tenants) + 1):
                    if not active or free <= 1e-9:
                        break
                    ns = sum(need[t] for t in active)
                    capped = set()
                    for t in list(active):
                        give = free * need[t] / ns
                        room = env.get(t, N) - alloc[t]
                        if give >= room:
                            alloc[t] += room
                            capped.add(t)
                    if not capped:
                        for t in active:
                            alloc[t] += free * need[t] / ns
                        break
                    free = N - sum(alloc.values())
                    active -= capped
                need = alloc
                cur = ran.commanded_controls()
                for t in tenants:
                    for kk, v in split(t, need[t]).items():
                        if abs(v - float(cur.get(kk, v))) >= a.min_change:
                            ran.apply(kk, v, scope="slice", tenant=t)
            k = ran.step(spe, record=False)
            if ep < a.from_epoch:
                continue
            vals = []
            for i, it in reg.intents.items():
                g = margin(it, k)
                if g is not None:
                    ok = 1.0 if g >= 0 else 0.0
                    per[i].append(ok)
                    vals.append(ok)
            ifs.append(float(np.mean(vals)))
        return float(np.mean(ifs)), {i: float(np.mean(v)) for i, v in per.items()}

    seeds = [int(s) for s in a.seeds.split(",")]
    F, C, Fi, Ci = [], [], [], []
    for sd in seeds:
        f, fi = run(sd, False)
        c, ci = run(sd, True)
        F.append(f); C.append(c); Fi.append(fi); Ci.append(ci)
        log(f"  seed {sd}: FLOOR (provisioned, = all-reject) {f:.4f}   "
            f"CEILING (demand-tracking) {c:.4f}   gap {c - f:+.4f}")
    fm, cm = float(np.mean(F)), float(np.mean(C))
    iids = sorted(reg.intents)
    lines = [f"# Control gain: {scenario_name(cfg0)}", "",
             f"Seeds {seeds}; scored from epoch {a.from_epoch}.", "",
             "| | IF | " + " | ".join(iids) + " |",
             "|---|---|" + "---|" * len(iids),
             "| FLOOR: provisioned (all-reject) | " + f"{fm:.4f} | "
             + " | ".join(f"{np.mean([x[i] for x in Fi]):.2f}" for i in iids) + " |",
             "| CEILING: demand-tracking | " + f"{cm:.4f} | "
             + " | ".join(f"{np.mean([x[i] for x in Ci]):.2f}" for i in iids) + " |",
             "",
             f"- ceiling {cm:.4f} (want >= {a.want_ceiling}) -> "
             f"{'OK' if cm >= a.want_ceiling else 'TOO LOW: even ideal control fails intents too often'}",
             f"- gap {cm - fm:+.4f} (want >= {a.want_gap}) -> "
             f"{'OK' if cm - fm >= a.want_gap else 'TOO SMALL: doing nothing is nearly as good as ideal control'}"]
    out = results_dir(a, cfg0, sub="_control_gain")
    write_text(out / "CONTROL_GAIN.md", "\n".join(lines))
    write_json(out / "control_gain.json", {"floor": F, "ceiling": C})
    log("\n" + "\n".join(lines[4:]))


if __name__ == "__main__":
    main()
