#!/usr/bin/env python3
"""Run ONE method on one scenario with one seed. Resumable.

Use this while developing or debugging. For the actual benchmark use
run_benchmark.py, which runs every method on the same plant and seeds and
then applies the validity gates.
"""
from __future__ import annotations
import argparse, time
from _common import ROOT, Log, common_args, load, results_dir


def main():
    p = common_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--method", default="intact-ra-agentic")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-telemetry", action="store_true")
    a = p.parse_args()

    from intact_agentic import methods as M
    from intact_agentic.config import build_registry
    from intact_agentic.experiment import Experiment, build_prior

    cfg = load(a)
    if a.epochs:
        cfg["run"]["epochs"] = int(a.epochs)
    seed = a.seed if a.seed is not None else int(cfg["run"]["seed"])
    rd = results_dir(a, cfg, sub=f"{a.method}/seed{seed}")
    log = Log(rd / "run.log", a.quiet)
    reg = build_registry(cfg)
    prior = build_prior(cfg, reg, log=log,
                        cache=ROOT / (cfg.get("calibration", {}) or {}).get(
                            "cache", "artifacts/sensitivity_prior.json"))
    ex = Experiment(cfg, M.get(a.method), reg, rd, seed=seed, prior=prior,
                    telemetry=not a.no_telemetry, log=log)
    if a.resume:
        ex.load_checkpoint()
    t0 = time.time()
    s = ex.run()
    log("")
    for k in ("wIF", "worst_intent_fulfilment", "safety_crossings",
              "writes_per_epoch", "prediction_mae", "latency_ms_p95",
              "candidates_mean", "probes"):
        if k in s:
            log(f"  {k:26s} {s[k]}")
    log(f"  wall {time.time()-t0:.0f}s   -> {rd}")
    log.close()


if __name__ == "__main__":
    main()
