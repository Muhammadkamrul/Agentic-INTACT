#!/usr/bin/env python3
"""Can every intent be met at all?  [AUTO]

For each tenant in turn, gives that tenant its ENTIRE contractual envelope
(all its allocative knobs at their maximum, capped by the envelope), holds
every other tenant at its provisioned allocation, and measures how often
the tenant's own intents are met.

An intent that cannot be met even at the tenant's contractual maximum is
unwinnable by construction.  Under a linear-margin objective it is worse
than useless: every extra PRB still IMPROVES its margin, so a controller
keeps granting capacity that never turns into fulfilment, while tenants
that could have been served starve.  That is exactly what drove S14's
INTACT-RA to push T3 to its envelope and drain T2.

Reports per-intent fulfilment at (a) the provisioned allocation and (b) the
tenant's envelope maximum, and flags any intent that stays below
--min-attainable at (b).
"""
from __future__ import annotations
import argparse
import numpy as np
from _common import ROOT, Log, common_args, load, results_dir, write_text


def main():
    p = common_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--seed", type=int, default=20260925)
    p.add_argument("--from-epoch", type=int, default=100)
    p.add_argument("--min-attainable", type=float, default=0.80)
    a = p.parse_args()

    from intact_agentic.config import build_registry, epoch_slots, scenario_name
    from intact_agentic.ran.simulator import RealisticRAN
    from intact_agentic.arbiter.margins import margin

    cfg0 = load(a)
    reg = build_registry(cfg0)
    pre, post = epoch_slots(cfg0)
    E = int(cfg0["run"]["epochs"])
    log = Log(None, a.quiet)
    env = {t: float(v.get("prb", 1e9)) for t, v in
           (cfg0["ran"].get("envelopes", {}) or {}).items()}
    knobs = {}
    for c in cfg0["claims"]:
        if c["kind"] == "allocative":
            knobs.setdefault(c["tenant"], {})[c["param"]] = float(c["domain"][1])

    def run(over):
        c = load(a)
        c["ran"]["initial_controls"].update(over)
        ran = RealisticRAN(c)
        ran.reset(a.seed)
        f = {i: [] for i in reg.intents}
        for ep in range(E):
            ran.epoch = ep
            k = ran.step(pre + post, record=False)
            if ep < a.from_epoch:
                continue
            for i, it in reg.intents.items():
                g = margin(it, k)
                if g is not None:
                    f[i].append(1.0 if g >= 0 else 0.0)
        return {i: float(np.mean(v)) for i, v in f.items() if v}

    base = run({})
    lines = [f"# Intent attainability: {scenario_name(cfg0)}", "",
             "| intent | tenant | provisioned | at envelope max | whole cell | verdict |",
             "|---|---|---|---|---|---|"]
    bad = []
    for tid, ks in knobs.items():
        # scale this tenant's knobs to its envelope, keep the knob ratios
        tot = sum(ks.values())
        scale = min(1.0, env.get(tid, tot) / max(tot, 1e-9))
        over = {k: v * scale for k, v in ks.items()}
        # every OTHER tenant yields to its knob minima, so the pool can
        # actually honour this tenant's envelope instead of normalising it
        # back down -- the question is whether the intent can be met at all
        mins = {c["param"]: float(c["domain"][0]) for c in cfg0["claims"]
                if c["kind"] == "allocative" and c["tenant"] != tid}
        res = run({**mins, **over})
        # reference: this tenant with the WHOLE cell, envelope ignored.  If
        # an intent fails even here, it is unmeetable for a reason no
        # allocation can fix -- e.g. a served-ratio target that bursty
        # traffic cannot hit over a short scoring window
        whole = {k: v * (float(cfg0["ran"]["n_prb"]) / max(tot, 1e-9))
                 for k, v in ks.items()}
        res_whole = run({**mins, **whole})
        for i, it in reg.intents.items():
            if it.tenant != tid:
                continue
            ok = res[i] >= a.min_attainable
            if not ok:
                bad.append(i)
            why = ("attainable" if ok else
                   "**unmeetable even with the whole cell**"
                   if res_whole[i] < a.min_attainable else
                   "**UNATTAINABLE within its envelope**")
            lines.append(f"| {i} ({it.kpi}) | {tid} | {base[i]:.2f} | "
                         f"{res[i]:.2f} | {res_whole[i]:.2f} | {why} |")
            log(f"  {i:4s} {tid} {it.kpi:14s} provisioned {base[i]:.2f}  "
                f"envelope {res[i]:.2f}  whole cell {res_whole[i]:.2f}  "
                f"{'ok' if ok else ('WHOLE-CELL FAIL' if res_whole[i] < a.min_attainable else 'envelope too small')}")
    lines += ["", ("All intents are attainable within their envelopes."
                   if not bad else
                   f"**{len(bad)} intent(s) cannot be met even at the tenant's "
                   f"contractual maximum: {', '.join(bad)}.** Resize the "
                   f"tenant's load, envelope or target before benchmarking.")]
    out = results_dir(a, cfg0, sub="_attainability")
    write_text(out / "ATTAINABILITY.md", "\n".join(lines))
    log("\n" + lines[-1])


if __name__ == "__main__":
    main()
