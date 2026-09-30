#!/usr/bin/env python3
"""Trace the MECHANISM behind the headline.  [AUTO]  (~25 min)

Runs frozen INTACT-RA and INTACT-RA-Agentic on the same development seed and
records, through the plant change:

  * the TRUE local slopes of key (knob -> intent) pairs, measured every
    --every epochs by paired finite differences on clones of the agent's
    live plant (the same measurement the oracle uses);
  * INTACT-RA-Agentic's Kalman estimate and its standard deviation for the
    same pairs, in the regime it would use at that moment;
  * the frozen table's value that INTACT-RA reads for the same pairs;
  * both methods' committed reservations and per-epoch fulfilment;
  * the migrating tenant's mean distance from the site.

This is the evidence for criterion 9 of the acceptance gates: that
sensitivities change for a physical reason, and that tracking them changes
the decisions.  Development seeds only -- never a held-out seed.

Writes mechanism.json; scripts/make_paper_figures.py plots it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _common import ROOT, common_args, load


def main():
    p = common_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--seed", type=int, default=20260926)
    p.add_argument("--every", type=int, default=10)
    p.add_argument("--pairs", default="quota_T1:i1,quota_T2:i3,quota_T3:i5,"
                                      "quota_T2:i5")
    a = p.parse_args()

    from intact_agentic import methods as M
    from intact_agentic.config import build_registry, scenario_name
    from intact_agentic.experiment import Experiment
    from intact_agentic.arbiter.sensitivity import StaticSensitivity
    from intact_agentic.arbiter.oracle import OracleSensitivity
    from intact_agentic.arbiter.margins import margin
    from intact_agentic.arbiter.regime import build_regime_estimator

    cfg0 = load(a)
    scen = scenario_name(cfg0)
    reg = build_registry(cfg0)
    prior = StaticSensitivity.load(cfg0, ROOT / "artifacts"
                                   / f"prior_{scen}.json")
    pairs = [tuple(x.split(":")) for x in a.pairs.split(",")]
    out = Path(a.out) if a.out else ROOT / "results" / scen / "mechanism"
    out.mkdir(parents=True, exist_ok=True)
    rec = {"scenario": scen, "seed": a.seed, "pairs": [list(x) for x in pairs],
           "burn_in": int(cfg0["run"].get("burn_in_epochs", 0))}

    for name in ("intact-ra", "intact-ra-agentic"):
        c = load(a)
        c["run"].update({"log_every": 0, "checkpoint_every": 0,
                         "counterfactual": False})
        ex = Experiment(c, M.get(name), reg, out / name, seed=a.seed,
                        prior=prior, telemetry=False, log=lambda *z: None)
        co = load(a)
        co["oracle"].update({"replicates": 3, "settle_slots": 8,
                             "measure_slots": 24})
        orc = OracleSensitivity(co, reg.claims, reg.intents)
        R = ex.regime_est
        orig = R.estimate
        last = {}

        # INTACT-RA's OWN regime estimator, evaluated on the same state, so
        # the frozen value plotted is the one INTACT-RA would actually read
        load_regime = build_regime_estimator(ex.cfg, "tenant_load")

        def spy(ran, kpm, tenants, intents, _o=orig, _l=last):
            rep = _o(ran, kpm, tenants, intents)
            _l["rep"] = rep
            _l["load_rep"] = load_regime.estimate(ran, kpm, tenants, intents)
            return rep
        R.estimate = spy
        series = {"epoch": [], "IF": [], "res": {t: [] for t in ("T1", "T2", "T3")},
                  "radius_T1": []}
        slopes = {"epoch": [], "true": {f"{k}|{i}": [] for k, i in pairs},
                  "est": {f"{k}|{i}": [] for k, i in pairs},
                  "sd": {f"{k}|{i}": [] for k, i in pairs},
                  "frozen": {f"{k}|{i}": [] for k, i in pairs},
                  "regime": {f"{k}|{i}": [] for k, i in pairs}}
        E = int(c["run"]["epochs"])
        for ep in range(E):
            ex.step_epoch()
            r = ex.metrics.records[-1]
            series["epoch"].append(ep)
            series["IF"].append(float(np.mean([1.0 if g >= 0 else 0.0
                                               for g in r.g_after.values()])))
            cmd = ex.ran.commanded_controls()
            for t in ("T1", "T2", "T3"):
                series["res"][t].append(float(ex.ran.quota_of(t, cmd)))
            series["radius_T1"].append(float(np.mean(
                ex.ran.mobility.groups["T1"].distance())))
            if name == "intact-ra-agentic" and ep >= rec["burn_in"] \
                    and ep % a.every == 0 and "rep" in last:
                orc.refresh(ex.ran, ep, force=True,
                            params={k for k, _ in pairs})
                slopes["epoch"].append(ep)
                for k, i in pairs:
                    key = f"{k}|{i}"
                    rg = last["rep"].of_intent(i)
                    slopes["true"][key].append(float(orc.tab.get((k, i), 0.0)))
                    slopes["est"][key].append(float(ex.sens.get(rg, k, i)))
                    slopes["sd"][key].append(float(ex.sens.sigma(rg, k, i)))
                    lr = last["load_rep"].of_intent(i)
                    slopes["frozen"][key].append(float(prior.get(lr, k, i)))
                    slopes["regime"][key].append(rg)
        rec[name] = {"series": series, "summary_IF": ex.metrics.summary()["IF"]}
        if name == "intact-ra-agentic":
            rec["slopes"] = slopes
        print(f"{name}: IF {rec[name]['summary_IF']:.4f}", flush=True)

    (out / "mechanism.json").write_text(json.dumps(rec))
    print(f"wrote {out / 'mechanism.json'}")


if __name__ == "__main__":
    main()
