#!/usr/bin/env python3
"""How good are a method's writes, causally?  [AUTO]

For every epoch in which a method writes, compares the margins after the
write with those of the PAIRED no-write copy of the plant (identical random
draws), so plant noise cancels.  Per write epoch:

  causal fulfilment effect = (# intents fulfilled with the write)
                           - (# intents fulfilled without it), / # intents

and classifies each write epoch as helpful (> 0), harmful (< 0) or neutral.
It separates HOW OFTEN a method acts from HOW WELL: a method can score
badly because it acts often on a wrong model, or because it acts rarely.
"""
from __future__ import annotations
import argparse, json
import numpy as np
from _common import ROOT, common_args, load


def main():
    p = common_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--methods", default="intact-ra,intact-ra-cell")
    p.add_argument("--seeds", default="31001,31002,31003,31004,31005")
    a = p.parse_args()
    from intact_agentic import methods as M
    from intact_agentic.config import build_registry, scenario_name
    from intact_agentic.experiment import Experiment
    from intact_agentic.arbiter.sensitivity import StaticSensitivity
    import tempfile
    cfg0 = load(a); reg = build_registry(cfg0); scen = scenario_name(cfg0)
    prior = StaticSensitivity.load(cfg0, ROOT / "artifacts" / f"prior_{scen}.json")
    out = {}
    for m in a.methods.split(","):
        rows = []
        for sd in [int(s) for s in a.seeds.split(",")]:
            c = load(a); c["run"].update({"log_every": 0, "checkpoint_every": 0,
                                          "counterfactual": True})
            burn = int(c["run"].get("burn_in_epochs", 0))
            quotas = sorted(k for k in c["ran"]["initial_controls"]
                            if k.startswith("quota_"))
            n_up = n_down = 0
            reserved = []
            with tempfile.TemporaryDirectory() as td:
                ex = Experiment(c, M.get(m), reg, td, seed=sd, prior=prior,
                                telemetry=False, log=lambda *z: None)
                # step manually so the RESERVATION trajectory is visible:
                # a write that is harmless in its own epoch can still drain
                # the pool over many epochs, which a per-epoch counterfactual
                # cannot see
                for ep in range(int(c["run"]["epochs"])):
                    before = ex.ran.commanded_controls()
                    ex.step_epoch()
                    after = ex.ran.commanded_controls()
                    if ep >= burn:
                        for k in quotas:
                            d = after.get(k, 0.0) - before.get(k, 0.0)
                            n_up += d > 1e-9
                            n_down += d < -1e-9
                        reserved.append(sum(after.get(k, 0.0) for k in quotas))
                s = ex.metrics.summary()
            recs = [r for r in ex.metrics.records if r.epoch >= burn]
            eff = []
            for r in recs:
                if r.n_writes == 0 or not r.g_cf:
                    continue
                ids = [i for i in r.g_after if i in r.g_cf]
                e = (sum(r.g_after[i] >= 0 for i in ids)
                     - sum(r.g_cf[i] >= 0 for i in ids)) / max(len(ids), 1)
                eff.append(e)
            eff = np.array(eff)
            rows.append({"seed": sd, "IF": s["IF"], "epochs": len(recs),
                         "write_epochs": int(len(eff)),
                         "helpful": int((eff > 0).sum()),
                         "harmful": int((eff < 0).sum()),
                         "neutral": int((eff == 0).sum()),
                         "net_causal_IF": float(eff.sum() / max(len(recs), 1)),
                         "raises": int(n_up), "releases": int(n_down),
                         "reserved_mean": float(np.mean(reserved)),
                         "reserved_last100": float(np.mean(reserved[-100:]))})
            r = rows[-1]
            print(f"{m:15s} seed {sd}: IF {r['IF']:.4f}  write epochs {r['write_epochs']:3d}  "
                  f"helpful {r['helpful']:3d}  harmful {r['harmful']:3d}  neutral {r['neutral']:3d}  "
                  f"net causal IF {r['net_causal_IF']:+.4f}  raises {r['raises']:3d}  "
                  f"releases {r['releases']:3d}  PRB reserved mean {r['reserved_mean']:.1f} "
                  f"(last 100 epochs {r['reserved_last100']:.1f})", flush=True)
        out[m] = rows
    path = ROOT / "results" / scen / "write_quality.json"
    path.write_text(json.dumps(out, indent=2))
    n_seeds = len(a.seeds.split(","))
    print(f"\nsummary ({n_seeds} seed{'s' if n_seeds != 1 else ''}):")
    for m, rows in out.items():
        W = sum(r["write_epochs"] for r in rows); H = sum(r["helpful"] for r in rows)
        X = sum(r["harmful"] for r in rows)
        print(f"  {m:15s} write epochs {W:4d}  helpful {H/W if W else 0:5.1%}  harmful {X/W if W else 0:5.1%}  "
              f"net causal IF from writes {np.mean([r['net_causal_IF'] for r in rows]):+.4f}  "
              f"raises {sum(r['raises'] for r in rows)}  releases {sum(r['releases'] for r in rows)}  "
              f"PRBs reserved in the last 100 epochs {np.mean([r['reserved_last100'] for r in rows]):.1f}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
