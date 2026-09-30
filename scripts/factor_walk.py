#!/usr/bin/env python3
"""FACTOR WALK: which property of a plant makes a per-tenant regime win?  [AUTO]

Starts from a VALID base plant (default S16, where a cell-level regime wins)
and transplants properties of a DONOR plant (default S8, where a per-tenant
regime wins) one factor at a time -- or in combinations joined by '+' -- so
the property that flips the ordering can be isolated while validity stays
measurable.  Each variant is written as configs/scenarios/FW_<factors>.yaml,
gets its own offline table, and is run with the FROZEN methods only (cheap).

Factors (donor -> base):
  shared_knobs    the donor's cell-wide and scheduler knobs with their xApps:
                  transmit power (two writers: C1), tilt, scheduler weights,
                  MCS cap.  Their effect on a tenant depends on THAT tenant's
                  coverage and backlog -- the lead hypothesis.
  host_intent     the donor's host transmit-power intent
  loads           the donor's per-tenant loads and load profiles
  migration_out   the donor's mobility (T1 drifting OUTWARD) and placements
  no_burnin       no unscored burn-in
  percentile      the donor's percentile-calibrated tenant intents

Per variant it reports, with self-interpretation:
  * INTACT-RA (per-tenant) − INTACT-RA (cell), paired over seeds
      PER-TENANT WINS / CELL WINS / NO CLEAR ORDER   (threshold --margin)
  * best controller − all-reject: does control beat doing nothing?

Development seeds only.  A variant that looks promising must then pass the
arbiter-free validity checks (control_gain.py, intent_attainability.py) and
be evaluated on held-out seeds before any claim is made.

Usage
    python scripts/factor_walk.py --factors shared_knobs,loads,migration_out
    python scripts/factor_walk.py --factors shared_knobs+host_intent --seeds 20260925,20260926
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import yaml

from _common import ROOT

LIST_SECTIONS = ["tenants", "intents", "claims", "xapps"]


def f_shared_knobs(base, donor):
    keep = [c for c in donor["claims"] if not c["param"].startswith("quota_")]
    xn = {c["xapp"] for c in keep}
    used = {c["jid"] for c in base["claims"]}
    for k, c in enumerate(keep):
        c = copy.deepcopy(c)
        c["jid"] = f"x{k + 1}"
        assert c["jid"] not in used
        base["claims"].append(c)
    base["xapps"] += [copy.deepcopy(x) for x in donor["xapps"] if x["name"] in xn]
    for c in keep:
        base["ran"]["initial_controls"][c["param"]] = \
            donor["ran"]["initial_controls"].get(c["param"],
                                                 base["ran"]["initial_controls"].get(c["param"]))


def f_host_intent(base, donor):
    base["intents"] += [copy.deepcopy(i) for i in donor["intents"]
                        if i["tenant"] == "HOST"]


def f_loads(base, donor):
    for t, s in donor["ran"]["slices"].items():
        if t in base["ran"]["slices"]:
            for k in ("load_mbps_per_ue", "load_profile"):
                if k in s:
                    base["ran"]["slices"][t][k] = copy.deepcopy(s[k])


def f_migration_out(base, donor):
    base["ran"]["mobility"] = copy.deepcopy(donor["ran"]["mobility"])
    for t, s in donor["ran"]["slices"].items():
        if t in base["ran"]["slices"] and "placement" in s:
            base["ran"]["slices"][t]["placement"] = copy.deepcopy(s["placement"])


def f_no_burnin(base, donor):
    base["run"]["burn_in_epochs"] = 0


def f_percentile(base, donor):
    host = [i for i in base["intents"] if i["tenant"] == "HOST"]
    base["intents"] = [copy.deepcopy(i) for i in donor["intents"]
                       if i["tenant"] != "HOST"] + host


FACTORS = {"shared_knobs": f_shared_knobs, "host_intent": f_host_intent,
           "loads": f_loads, "migration_out": f_migration_out,
           "no_burnin": f_no_burnin, "percentile": f_percentile}


def build_variant(base_name, donor_name, combo):
    from intact_agentic.config import load_config
    base = load_config(scenario=base_name)
    donor = load_config(scenario=donor_name)
    v = copy.deepcopy(base)
    for f in combo.split("+"):
        FACTORS[f](v, donor)
    name = "FW_" + combo.replace("+", "__")
    v["_replace_sections"] = LIST_SECTIONS
    v["_scenario_note"] = (f"Factor-walk variant: {base_name} with {combo} "
                           f"transplanted from {donor_name}. Diagnostic only.")
    v.pop("_scenario", None)
    path = ROOT / "configs" / "scenarios" / f"{name}.yaml"
    path.write_text(f"# {v['_scenario_note']}\n"
                    + yaml.safe_dump(v, sort_keys=False, default_flow_style=None))
    return name


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="S16_high_ceiling")
    ap.add_argument("--donor", default="S8_dedicated_contest")
    ap.add_argument("--factors", default="shared_knobs,loads,migration_out")
    ap.add_argument("--seeds", default="20260925")
    ap.add_argument("--methods", default="all-reject,b3,intact-ra,intact-ra-cell")
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--include-base", action="store_true", default=True)
    ap.add_argument("--out", default=str(ROOT / "results" / "_factor_walk"))
    a = ap.parse_args()

    from intact_agentic import methods as M
    from intact_agentic.config import load_config, build_registry
    from intact_agentic.experiment import Experiment, build_prior
    import tempfile

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs_path = out / "runs.jsonl"
    done = {}
    if runs_path.exists():
        for line in runs_path.read_text().splitlines():
            d = json.loads(line)
            done[(d["variant"], d["m"], d["s"])] = d
    combos = [c.strip() for c in a.factors.split(",") if c.strip()]
    variants = ([("BASE", a.base)] if a.include_base else []) + \
        [(c, build_variant(a.base, a.donor, c)) for c in combos]
    seeds = [int(s) for s in a.seeds.split(",")]

    for label, scen in variants:
        cfg0 = load_config(scenario=scen)
        reg = build_registry(cfg0)
        prior = build_prior(cfg0, reg, log=lambda *z: None,
                            cache=ROOT / "artifacts" / f"prior_{scen}.json")
        for sd in seeds:
            for m in a.methods.split(","):
                if (label, m, sd) in done:
                    continue
                c = load_config(scenario=scen)
                c["run"].update({"log_every": 0, "checkpoint_every": 0,
                                 "counterfactual": False})
                with tempfile.TemporaryDirectory() as td:
                    ex = Experiment(c, M.get(m), reg, td, seed=sd, prior=prior,
                                    telemetry=False, log=lambda *z: None)
                    s = ex.run()
                d = {"variant": label, "scenario": scen, "m": m, "s": sd,
                     "IF": s["IF"], "wpe": s["writes_per_epoch"]}
                done[(label, m, sd)] = d
                with open(runs_path, "a") as fh:
                    fh.write(json.dumps(d) + "\n")
                print(f"{label:28s} seed {sd}  {m:15s} IF={s['IF']:.4f}  "
                      f"w/ep={s['writes_per_epoch']:.2f}", flush=True)

    L = ["# Factor walk", "",
         f"Base `{a.base}` (cell-level regime wins), donor `{a.donor}` "
         f"(per-tenant regime wins). Frozen methods, development seeds "
         f"{seeds}. Margin for a clear ordering: {a.margin}.", "",
         "| variant | all-reject | B3 | INTACT-RA (per-tenant) | INTACT-RA (cell) "
         "| per-tenant − cell | ordering | best controller − all-reject |",
         "|---|---|---|---|---|---|---|---|"]
    found = []
    for label, scen in variants:
        get = lambda m: [done[(label, m, s)]["IF"] for s in seeds
                         if (label, m, s) in done]
        vals = {m: get(m) for m in a.methods.split(",")}
        if not all(vals.values()):
            continue
        d = np.array(vals["intact-ra"]) - np.array(vals["intact-ra-cell"])
        if np.all(d > a.margin):
            order = "**PER-TENANT WINS**"
        elif np.all(d < -a.margin):
            order = "CELL WINS"
        else:
            order = "no clear order"
        best_ctl = max(np.mean(vals[m]) for m in vals if m != "all-reject")
        gain = best_ctl - np.mean(vals["all-reject"])
        L.append(f"| {label} | {np.mean(vals['all-reject']):.3f} | "
                 f"{np.mean(vals.get('b3', [np.nan])):.3f} | "
                 f"{np.mean(vals['intact-ra']):.3f} | "
                 f"{np.mean(vals['intact-ra-cell']):.3f} | {d.mean():+.3f} | "
                 f"{order} | {gain:+.3f} |")
        if order.startswith("**PER") and gain >= 0.03:
            found.append(label)
    L += ["", "## Interpretation", ""]
    if found:
        L.append(f"**Per-tenant regime wins AND control beats doing nothing in: "
                 f"{', '.join(found)}.** These are candidates for the "
                 f"per-tenant plant. Next: run control_gain.py and "
                 f"intent_attainability.py on each (validity), then more "
                 f"development seeds, then held-out evaluation with every "
                 f"method (run_heldout.py).")
    else:
        L.append("No variant here shows the per-tenant regime winning while "
                 "control beats doing nothing. Try combinations of factors "
                 "(join with '+'), or factors not yet transplanted.")
    (out / "FACTOR_WALK.md").write_text("\n".join(L))
    print("\n" + "\n".join(L))


if __name__ == "__main__":
    main()
