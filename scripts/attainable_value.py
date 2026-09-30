#!/usr/bin/env python3
"""STEP 0. Is there anything for ANY controller to win?  [AUTO]

Holds the controls at fixed settings with NO arbiter and measures intent
fulfilment separately before and after the scheduled plant change.  It
answers, independently of every sensitivity model and every oracle:

  * does the plant change break contracts when nothing is done?
  * is there a setting that repairs them?
  * is that setting DIFFERENT from the best setting before the change?

Only if all three hold does adapting the controls have attainable value.
If the best setting is the same in both phases, a frozen table that keeps
the pre-change preference is already optimal, and no estimator -- however
current -- can beat it.  This instrument found that S9's tilt optimum did
not move at all (8 deg both before and after), which explained why every
controller there lost to leaving the knob alone.

Also sweeps the SEVERITY of the change (drift rate), because a change too
mild breaks nothing and a change too severe breaks what no setting can fix.

Writes ATTAINABLE_VALUE.md and attainable_value.json.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from _common import ROOT, Log, common_args, load, results_dir, write_json, write_text

PRESET = {
    "default":        {},
    "quota_T1+":      {"quota_T1": 58.0},
    "tilt 6":         {"tilt": 6.0},
    "tilt 6+quota+":  {"tilt": 6.0, "quota_T1": 58.0},
    "power+":         {"txpower": 43.0},
    "all three":      {"tilt": 6.0, "quota_T1": 58.0, "txpower": 43.0},
}


def main():
    p = common_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--rates", default=None,
                   help="comma list of drift rates (m/slot); default: scenario's")
    p.add_argument("--seeds", default="20260925")
    p.add_argument("--pre", default="10:95", help="pre-change epoch window")
    p.add_argument("--post", default=None, help="post-change window a:b")
    p.add_argument("--min-headroom", type=float, default=0.05)
    p.add_argument("--settings", default=None,
                   help='JSON dict {label: {control: value}}; the first '
                        'label is treated as the default')
    a = p.parse_args()

    from intact_agentic import methods as M
    from intact_agentic.config import build_registry, scenario_name
    from intact_agentic.experiment import Experiment
    from intact_agentic.arbiter.sensitivity import StaticSensitivity

    cfg0 = load(a)
    scen = scenario_name(cfg0)
    out = results_dir(a, cfg0, sub="_attainable_value")
    log = Log(out / "attainable.log", a.quiet)
    reg = build_registry(cfg0)
    E = int(cfg0["run"]["epochs"])
    pre = tuple(int(x) for x in a.pre.split(":"))
    post = tuple(int(x) for x in (a.post or f"{int(0.6*E)}:{E}").split(":"))
    rates = [float(r) for r in a.rates.split(",")] if a.rates else \
        [float(cfg0["ran"]["mobility"]["drift"].get("rate_m_per_slot", 0.0))]
    seeds = [int(s) for s in a.seeds.split(",")]
    null_prior = StaticSensitivity(cfg0, {})
    iids = sorted(reg.intents)
    presets = json.loads(a.settings) if a.settings else PRESET
    default_label = next(iter(presets))

    def fulfil(recs, lo, hi, iid=None):
        v = [1.0 if r.g_after[i] >= 0 else 0.0 for rs in recs for r in rs
             if lo <= r.epoch < hi for i in r.g_after if iid in (None, i)]
        return float(np.mean(v)) if v else float("nan")

    res = {}
    for rate in rates:
        for name, over in presets.items():
            recs = []
            for sd in seeds:
                c = load(a)
                c["run"].update({"log_every": 0, "checkpoint_every": 0,
                                 "counterfactual": False})
                c["ran"]["mobility"]["drift"]["rate_m_per_slot"] = rate
                for k, v in over.items():
                    c["ran"]["initial_controls"][k] = v
                ex = Experiment(c, M.get("all-reject"), reg,
                                out / f"r{rate}" / name.replace(" ", "_")
                                / f"s{sd}", seed=sd, prior=null_prior,
                                telemetry=False, log=lambda *z: None)
                ex.run()
                recs.append(ex.metrics.records)
            res[(rate, name)] = {
                "pre": fulfil(recs, *pre), "post": fulfil(recs, *post),
                "post_intent": {i: fulfil(recs, *post, i) for i in iids}}
            r = res[(rate, name)]
            log(f"  rate {rate:.4f}  {name:15s} IF pre {r['pre']:.4f}  "
                f"post {r['post']:.4f}  |  " + " ".join(
                    f"{i}:{r['post_intent'][i]:.2f}" for i in iids))

    lines = [f"# Attainable value of adaptation: {scen}", "",
             f"Controls held CONSTANT, no arbiter. Pre window {pre}, post "
             f"window {post}, seeds {seeds}.", ""]
    verdicts = {}
    for rate in rates:
        rows = {n: res[(rate, n)] for n in presets}
        best_pre = max(rows, key=lambda n: rows[n]["pre"])
        best_post = max(rows, key=lambda n: rows[n]["post"])
        held = rows[default_label]
        change = held["post"] - held["pre"]
        # What decides whether adaptation has value is NOT whether the plant
        # change degrades service -- an inward migration makes a tenant
        # EASIER to serve, and passive fulfilment then rises.  It is whether
        # moving the controls after the change beats holding the provisioned
        # ones, and whether the best setting is a different one from before.
        head = rows[best_post]["post"] - held["post"]
        shifted = best_post != best_pre
        ok = head >= a.min_headroom and shifted
        verdicts[rate] = ok
        lines += [f"## Drift rate {rate:.4f} m/slot", "",
                  "| setting | IF pre | IF post |", "|---|---|---|"]
        for n in presets:
            lines.append(f"| {n} | {rows[n]['pre']:.4f} | "
                         f"{rows[n]['post']:.4f} |")
        lines += ["",
                  f"- effect of the change with controls held: "
                  f"{held['pre']:.4f} -> {held['post']:.4f} "
                  f"(**{change:+.4f}**; informative, not a gate)",
                  f"- best setting before: `{best_pre}`; best after: "
                  f"`{best_post}`",
                  f"- headroom from adapting: **{head:+.4f}** "
                  f"(need >= {a.min_headroom}) -> "
                  f"{'yes' if head >= a.min_headroom else 'NO'}",
                  f"- optimum SHIFTS with the change: "
                  f"{'yes' if shifted else 'NO'}"
                  + ("" if shifted else "  -- the provisioned preference is "
                     "already right after the change, so a frozen table that "
                     "keeps it cannot be beaten"),
                  f"- **verdict: {'ATTAINABLE VALUE' if ok else 'no attainable value'}**",
                  ""]
    good = [r for r, ok in verdicts.items() if ok]
    lines += ["## Summary", "",
              (f"Adaptation has attainable value at drift rate(s) "
               f"{', '.join(f'{r:.4f}' for r in good)}. Use one of these "
               f"for the benchmark, then run the oracle diagnostic."
               if good else
               "**No swept setting gives adaptation attainable value.** Do "
               "not tune a controller against this scenario: either no "
               "setting beats holding the provisioned controls by the "
               "required margin, or the best setting is unchanged, in which "
               "case a frozen preference is already right.")]
    write_text(out / "ATTAINABLE_VALUE.md", "\n".join(lines))
    write_json(out / "attainable_value.json",
               {f"{r}|{n}": v for (r, n), v in res.items()})
    log("\n" + "\n".join(lines[lines.index("## Summary"):]))
    log.close()


if __name__ == "__main__":
    main()
