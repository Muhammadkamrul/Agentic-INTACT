#!/usr/bin/env python3
"""STEP 2. Screen scenarios BEFORE spending time on a full benchmark.

Runs a short version of each scenario with the minimum set of methods
needed to evaluate the gates, then REFUSES the ones that cannot support
the claim. Exits non-zero if any requested scenario is refused, so it can
gate a longer job in a shell script:

    python scripts/validate_scenarios.py --scenarios S1_edge_drift \\
        && python scripts/run_benchmark.py --scenario S1_edge_drift
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
from _common import ROOT, Log, write_json, write_text


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenarios", default="S0_stationary,S1_edge_drift,"
                                          "S2_traffic_shift,S3_new_tenant,"
                                          "S4_interference")
    p.add_argument("--base", default=str(ROOT / "configs" / "base.yaml"))
    p.add_argument("--epochs", type=int, default=None,
                   help="override; default is gates.probe_epochs")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", default=str(ROOT / "results" / "_screening"))
    p.add_argument("--skip-drift", action="store_true",
                   help="skip the slope-drift measurement (much faster)")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args()

    from intact_agentic import gates as G, methods as M
    from intact_agentic.config import load_config, build_registry
    from intact_agentic.experiment import Experiment, build_prior
    from intact_agentic.report import interpret as I

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = Log(out / "screening.log", a.quiet)
    NEEDED = ("all-reject", "all-accept", "static-priority",
              "intact-ra", "intact-ra-agentic")
    verdicts = {}

    for name in [s.strip() for s in a.scenarios.split(",") if s.strip()]:
        log("")
        log("=" * 68)
        log(f"SCREENING {name}")
        log("=" * 68)
        cfg = load_config(scenario=name, base=a.base)
        g = cfg.get("gates", {}) or {}
        cfg["run"]["epochs"] = int(a.epochs or g.get("probe_epochs", 220))
        cfg["run"]["log_every"] = 0
        cfg["run"]["checkpoint_every"] = 0
        seed = a.seed if a.seed is not None else int(cfg["run"]["seed"])
        reg = build_registry(cfg)
        prior = build_prior(cfg, reg, log=lambda *x: None,
                            cache=ROOT / (cfg.get("calibration", {}) or {})
                            .get("cache", "artifacts/sensitivity_prior.json"))
        summaries, runs = {}, {}
        for mn in NEEDED:
            t0 = time.time()
            ex = Experiment(cfg, M.get(mn), reg, out / name / mn, seed=seed,
                            prior=prior, telemetry=False,
                            log=lambda *x: None)
            s = ex.run()
            summaries[mn] = s
            runs[mn] = ex
            log(f"  {mn:24s} wIF={s['wIF']:.4f} "
                f"cross={s['safety_crossings']:4d} "
                f"w/ep={s['writes_per_epoch']:.2f}  [{time.time()-t0:.0f}s]")

        drift = None
        if not a.skip_drift and \
                (cfg["ran"].get("mobility", {}) or {}).get("mode") == "group_drift":
            log("  measuring true slope drift on RAN clones...")
            ratios, flips, se, sl = G.measure_slope_drift(
                cfg, reg, early_epoch=15,
                late_epoch=max(cfg["run"]["epochs"] - 30, 40),
                reps=2, log=log)
            drift = (ratios, flips)
            write_text(out / name / "DRIFT.md", I.interpret_drift(
                ratios, flips, se, sl,
                float(g.get("drift_materiality_min", 2.0))))

        rep = G.run_gates(cfg, name, summaries, runs, drift)
        log("")
        log(rep.text())
        rep.save(out / name / "gates.json")
        verdicts[name] = rep.passed

    log("")
    log("=" * 68)
    log("SCREENING SUMMARY")
    for n, ok in verdicts.items():
        log(f"  {n:24s} {'ACCEPTED' if ok else 'REFUSED'}")
    write_json(out / "verdicts.json", verdicts)
    log.close()
    if not all(verdicts.values()):
        sys.exit(2)


if __name__ == "__main__":
    main()
