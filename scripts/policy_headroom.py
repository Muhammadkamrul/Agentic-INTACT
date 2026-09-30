#!/usr/bin/env python3
"""Attainable value of ADAPTIVE policies, with no arbiter.  [AUTO]

A constant-control sweep cannot show the value of adapting to load that
varies over time: any single constant split only half-fits alternating
phases.  This script instead drives the plant with SCHEDULED policies --
control settings that change at chosen epochs -- and pays the real
reconfiguration transient on every change.  No sensitivity model, oracle
or arbiter is involved, so the numbers bound what ANY controller could
achieve by following that schedule.

The comparison that matters for the headline is between two adaptive
policies that differ in exactly one respect:

  track, T1 held      follows the T2/T3 load phases but keeps T1's
                      provisioned reservation for the whole run -- what a
                      controller trusting a table calibrated at the edge does
  track, T1 released  identical, except that once T1 has moved inward it
                      releases T1's now-unneeded reservation to T2/T3 -- what
                      current sensitivity knowledge makes possible

Their difference after the migration is the attainable value of knowing
T1's CURRENT need.  Runs several seeds and reports the spread, because a
headroom smaller than the seed-to-seed noise is not evidence of anything.
"""
from __future__ import annotations
import argparse, json
import numpy as np
from _common import ROOT, Log, common_args, load, results_dir, write_json, write_text


def main():
    p = common_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--seeds", default="20260925,20260926,20260927")
    p.add_argument("--release-epoch", type=int, default=250)
    p.add_argument("--pre", default="10:95")
    p.add_argument("--post", default="300:500")
    p.add_argument("--policies", default=None,
                   help='JSON {name: [[from_epoch, [q_T1, q_T2, q_T3]], ...]}: '
                        'piecewise-constant schedules. Replaces the built-in '
                        'phase-tracking set.')
    p.add_argument("--compare", default=None,
                   help='JSON [[a, b, label, window], ...]: paired a-minus-b '
                        'differences to report; window is "pre" or "post"')
    a = p.parse_args()

    from intact_agentic.config import build_registry, epoch_slots, scenario_name
    from intact_agentic.ran.simulator import RealisticRAN
    from intact_agentic.arbiter.margins import margin

    cfg0 = load(a)
    scen = scenario_name(cfg0)
    out = results_dir(a, cfg0, sub="_policy_headroom")
    log = Log(out / "policy_headroom.log", a.quiet)
    reg = build_registry(cfg0)
    pre_s, post_s = epoch_slots(cfg0)
    spe = pre_s + post_s
    E = int(cfg0["run"]["epochs"])
    hold = int(cfg0["ran"]["slices"]["T2"]["load_profile"]["hold_slots"])
    R = a.release_epoch

    def t2_high(ep):
        return ((ep * spe) // hold) % 2 == 0

    held = (50.0, 38.0, 30.0)
    T2H_held, T3H_held = (50.0, 46.0, 22.0), (50.0, 30.0, 38.0)
    T2H_rel, T3H_rel = (26.0, 54.0, 38.0), (26.0, 40.0, 44.0)
    def piecewise(segs):
        segs = sorted((int(e), tuple(float(x) for x in q)) for e, q in segs)
        def f(ep):
            cur = segs[0][1]
            for e, q in segs:
                if ep >= e:
                    cur = q
            return cur
        return f

    POL = {
        "held (all-reject)":   lambda ep: held,
        "track, T1 held":      lambda ep: T2H_held if t2_high(ep) else T3H_held,
        "track, T1 released":  lambda ep: ((T2H_held if t2_high(ep) else T3H_held)
                                           if ep < R else
                                           (T2H_rel if t2_high(ep) else T3H_rel)),
        "release only":        lambda ep: held if ep < R else (26.0, 50.0, 42.0),
        "track, released EARLY": lambda ep: T2H_rel if t2_high(ep) else T3H_rel,
    }
    if a.policies:
        POL = {n: piecewise(s) for n, s in json.loads(a.policies).items()}
    win = lambda s: tuple(int(x) for x in s.split(":"))
    PRE, POST = win(a.pre), win(a.post)
    seeds = [int(s) for s in a.seeds.split(",")]
    res = {}
    for name, pol in POL.items():
        pre_v, post_v = [], []
        for sd in seeds:
            ran = RealisticRAN(cfg0)
            ran.reset(sd)
            cur = None
            f = {}
            for ep in range(E):
                ran.epoch = ep
                want = pol(ep)
                if want != cur:
                    for tid, v in zip(("T1", "T2", "T3"), want):
                        if cur is None or abs(v - cur[("T1", "T2", "T3")
                                                     .index(tid)]) > 1e-9:
                            ran.apply(f"quota_{tid}", v, scope="slice",
                                      tenant=tid)
                    if cur is None:     # provisioning is not a live write
                        ran._reconfig_left = {t: 0 for t in ran.slices}
                        ran._cell_reconfig_left = 0
                    cur = want
                k = ran.step(spe, record=False)
                vals = []
                for iid, it in reg.intents.items():
                    g = margin(it, k)
                    if g is not None:
                        vals.append(1.0 if g >= 0 else 0.0)
                f[ep] = float(np.mean(vals))
            pre_v.append(np.mean([f[e] for e in range(*PRE)]))
            post_v.append(np.mean([f[e] for e in range(*POST)]))
        res[name] = {"pre": pre_v, "post": post_v}
        log(f"  {name:24s} IF pre {np.mean(pre_v):.4f} (sd {np.std(pre_v, ddof=1):.4f})"
            f"   post {np.mean(post_v):.4f} (sd {np.std(post_v, ddof=1):.4f})")

    def paired_pre(x, y):
        # the early-release check is about the PRE window: after the change
        # both policies have released, so a post-window comparison would
        # duplicate the headline row by construction
        d = np.array(res[x]["pre"]) - np.array(res[y]["pre"])
        return float(d.mean()), float(d.std(ddof=1)), int((d > 0).sum()), len(d)

    def paired(x, y):
        d = np.array(res[x]["post"]) - np.array(res[y]["post"])
        return float(d.mean()), float(d.std(ddof=1)), int((d > 0).sum()), len(d)

    lines = [f"# Attainable value of adaptive policies: {scen}", "",
             f"No arbiter. Seeds {seeds}. Pre window {PRE}, post window "
             f"{POST}. T1 released at epoch {R}.", "",
             "| policy | IF pre | IF post |", "|---|---|---|"]
    for n, r in res.items():
        lines.append(f"| {n} | {np.mean(r['pre']):.4f} | "
                     f"{np.mean(r['post']):.4f} |")
    lines += ["", "## Paired differences after the change (same seeds)", ""]
    comps = (json.loads(a.compare) if a.compare else [
        ["track, T1 held", "held (all-reject)",
         "value of tracking the load phases at all", "post"],
        ["track, T1 released", "track, T1 held",
         "value of knowing T1's CURRENT need  <-- the headline quantity", "post"],
        ["track, released EARLY", "track, T1 held",
         "releasing BEFORE the migration (should hurt: T1 needed it)", "pre"]])
    for x, y, what, w in comps:
        m, s, pos, n = (paired_pre(x, y) if w == "pre" else paired(x, y))
        lines.append(f"- {what}: **{m:+.4f}** (sd {s:.4f}, positive on "
                     f"{pos}/{n} seeds)")
    write_text(out / "POLICY_HEADROOM.md", "\n".join(lines))
    write_json(out / "policy_headroom.json", res)
    log("\n" + "\n".join(lines[lines.index("## Paired differences after the change (same seeds)"):]))
    log.close()


if __name__ == "__main__":
    main()
