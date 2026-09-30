#!/usr/bin/env python3
"""STEP 3. Run the benchmark: every method, every seed, one scenario.

Resumable. Each (method, seed) run checkpoints every `run.checkpoint_every`
epochs and a completed run is skipped on a re-invocation unless --force is
given, so an interrupted overnight job is restarted with the same command.

Automated diagnostics run at the end without being asked for: the scenario
validity gates, the paired bootstrap over seeds, and the written
interpretation. If a critical gate fails the script says so loudly and
exits non-zero, because a benchmark on a degenerate scenario is worse than
no benchmark.
"""
from __future__ import annotations
import argparse, json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
from _common import ROOT, Log, common_args, load, results_dir, write_json, write_text


def main():
    p = common_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--methods", default="headline",
                   help="comma-separated names, or headline|ablations|full")
    p.add_argument("--seeds", default=None,
                   help="comma-separated ints (default: run.seeds)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--force", action="store_true", help="ignore completed runs")
    p.add_argument("--no-telemetry", action="store_true",
                   help="skip per-slot CSVs (much faster, no RAN figures)")
    p.add_argument("--no-gates", action="store_true")
    a = p.parse_args()

    from intact_agentic import methods as M
    from intact_agentic.config import build_registry, scenario_name
    from intact_agentic.experiment import Experiment, build_prior
    from intact_agentic.report.metrics import bootstrap_ci, paired_delta_ci
    from intact_agentic import gates as G
    from intact_agentic.report import interpret as I

    cfg = load(a)
    if a.epochs:
        cfg["run"]["epochs"] = int(a.epochs)
    root = results_dir(a, cfg)
    log = Log(root / "benchmark.log", a.quiet)
    reg = build_registry(cfg)
    meths = M.resolve(a.methods)
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds \
        else list(cfg["run"].get("seeds", [cfg["run"]["seed"]]))

    log(f"scenario   {scenario_name(cfg)}")
    log(f"methods    {', '.join(m.name for m in meths)}")
    log(f"seeds      {seeds}")
    log(f"epochs     {cfg['run']['epochs']}")
    log("")

    cache = ROOT / (cfg.get("calibration", {}) or {}).get(
        "cache", "artifacts/sensitivity_prior.json")
    prior = build_prior(cfg, reg, log=log, cache=cache)
    log("")

    per_seed = defaultdict(dict)     # method -> seed -> summary
    runs_seed0 = {}
    t_all = time.time()
    for m in meths:
        for sd in seeds:
            rd = root / m.name / f"seed{sd}"
            done = rd / "summary.json"
            if done.exists() and not a.force:
                per_seed[m.name][sd] = json.loads(done.read_text())
                log(f"  skip {m.name:24s} seed {sd}  (already complete)")
                continue
            t0 = time.time()
            ex = Experiment(cfg, m, reg, rd, seed=sd, prior=prior,
                            telemetry=not a.no_telemetry, log=log)
            ex.load_checkpoint()
            s = ex.run()
            per_seed[m.name][sd] = s
            if sd == seeds[0]:
                runs_seed0[m.name] = ex
            log(f"  {m.name:24s} seed {sd}  wIF={s['wIF']:.4f} "
                f"cross={s['safety_crossings']:4d} "
                f"w/ep={s['writes_per_epoch']:.2f} "
                f"p95lat={s['latency_ms_p95']:.2f}ms "
                f"[{time.time()-t0:.0f}s]")

    # ---- aggregate across seeds -----------------------------------------
    NUM = ("wIF", "worst_intent_fulfilment", "safety_crossings",
           "writes_per_epoch", "prediction_mae", "latency_ms_mean",
           "latency_ms_p95", "candidates_mean", "search_regret",
           "override_frac", "reject_frac", "mean_margin", "probes",
           "c1_violations", "c2_violations", "zero_crossings")
    summaries, cis = {}, {}
    for name, bysd in per_seed.items():
        rows = list(bysd.values())
        agg = {"method": name, "label": M.get(name).label,
               "family": M.get(name).family, "n_seeds": len(rows)}
        for k in NUM:
            v = [r.get(k) for r in rows if isinstance(r.get(k), (int, float))]
            if v:
                agg[k] = float(np.mean(v))
                cis.setdefault(name, {})[k] = bootstrap_ci(v, seed=7)
        for k in rows[0]:
            if k.startswith("fulfilment_"):
                agg[k] = float(np.mean([r.get(k, np.nan) for r in rows]))
        summaries[name] = agg

    paired = {}
    if "intact-ra" in per_seed and "intact-ra-agentic" in per_seed:
        common = sorted(set(per_seed["intact-ra"]) &
                        set(per_seed["intact-ra-agentic"]))
        for k in ("wIF", "safety_crossings", "prediction_mae"):
            paired[k] = paired_delta_ci(
                [per_seed["intact-ra-agentic"][s].get(k, np.nan) for s in common],
                [per_seed["intact-ra"][s].get(k, np.nan) for s in common],
                seed=11)

    write_json(root / "summaries.json", summaries)
    write_json(root / "per_seed.json", per_seed)
    write_json(root / "bootstrap_ci.json", cis)
    write_json(root / "paired.json", paired)

    # ---- gates -----------------------------------------------------------
    rep = None
    if not a.no_gates:
        log("")
        log("running scenario validity gates...")
        drift = None
        if (cfg["ran"].get("mobility", {}) or {}).get("mode") == "group_drift":
            ratios, flips, se, sl = G.measure_slope_drift(
                cfg, reg, early_epoch=20,
                late_epoch=max(cfg["run"]["epochs"] - 60, 60), log=log)
            drift = (ratios, flips)
            write_json(root / "slope_drift.json",
                       {"ratios": ratios, "sign_flips": flips,
                        "early": {f"{k[0]}->{k[1]}": v for k, v in se.items()},
                        "late": {f"{k[0]}->{k[1]}": v for k, v in sl.items()}})
            write_text(root / "DRIFT.md", I.interpret_drift(
                ratios, flips, se, sl,
                float((cfg.get("gates", {}) or {}).get(
                    "drift_materiality_min", 2.0))))
        rep = G.run_gates(cfg, scenario_name(cfg), summaries,
                          runs_seed0 or None, drift)
        log("")
        log(rep.text())
        rep.save(root / "gates.json")
        write_text(root / "GATES.md", "```\n" + rep.text() + "\n```\n")

    # ---- interpretation ---------------------------------------------------
    text = I.interpret_benchmark(summaries, cfg, scenario_name(cfg),
                                 ci=cis, paired=paired, gate_report=rep)
    write_text(root / "INTERPRETATION.md", text)
    log("")
    log(text)
    log("")
    log(f"total wall time {time.time()-t_all:.0f}s   results in {root}")
    log.close()
    if rep is not None and not rep.passed:
        sys.exit(2)


if __name__ == "__main__":
    main()
