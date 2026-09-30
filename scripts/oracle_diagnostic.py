#!/usr/bin/env python3
"""THE gate that decides whether a scenario is worth benchmarking.

Runs the identical arbiter with four different sensitivity models and
reports whether knowing the CURRENT slopes is worth anything:

    oracle        true local slopes, refreshed from cloned paired finite
                  differences with common random numbers
    frozen        the offline table (INTACT-RA)
    scrambled     the offline table's magnitude distribution with the
                  assignments permuted: calibrated aggregate beliefs,
                  zero specific knowledge
    agentic       the online estimator

CASE A  oracle - frozen < oracle_gain_min
        Sensitivity evolution is not decision-relevant. Do NOT tune the
        agent; diagnose the scenario. Start with scripts/probe_zerosum.py.

CASE B  oracle - frozen >= oracle_gain_min
        Exploitable. The problem is estimation, and agent work is
        justified.

The scrambled control is the one that cannot be argued with: if a
controller scores the same on permuted slopes as on true ones, the
scenario is not measuring sensitivity knowledge and any reported win for
a better estimator is an artefact of something else.
"""
from __future__ import annotations
import argparse, json
import numpy as np
from _common import ROOT, Log, common_args, load, results_dir, write_json, write_text


def main():
    p = common_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--seeds", default=None)
    p.add_argument("--epochs", type=int, default=None,
                   help="default: the scenario's run.epochs")
    p.add_argument("--refresh-every", type=int, default=5)
    p.add_argument("--per-epoch", action="store_true",
                   help="refresh the oracle's slopes EVERY scored epoch, only "
                        "for knobs with a proposal (2 replicates, 24-slot "
                        "windows). This is the setting that separates slope "
                        "staleness from myopia.")
    a = p.parse_args()

    from intact_agentic import methods as M
    from intact_agentic.config import build_registry, scenario_name
    from intact_agentic.experiment import Experiment, build_prior

    cfg = load(a)
    scen = scenario_name(cfg)
    out = results_dir(a, cfg, sub="_oracle_diagnostic")
    log = Log(out / "diagnostic.log", a.quiet)
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds \
        else list(cfg["run"].get("seeds", [cfg["run"]["seed"]]))[:2]
    g = cfg.get("gates", {}) or {}
    reg = build_registry(cfg)
    prior = build_prior(cfg, reg, log=log,
                        cache=ROOT / "artifacts" / f"prior_{scen}.json")

    NAMES = ("all-reject", "all-accept", "b3", "intact-ra",
             "intact-ra-agentic", "oracle", "b3-oracle", "b3-scrambled")
    R, X, W = {}, {}, {}
    log(f"\n{'method':24s} {'wIF':>8s} {'cross':>7s} {'w/ep':>7s}")
    for name in NAMES:
        w, x, wr = [], [], []
        for sd in seeds:
            c = load(a)
            c["run"].update({"log_every": 0, "checkpoint_every": 0})
            if a.epochs:
                c["run"]["epochs"] = a.epochs
            c["oracle"]["refresh_every"] = a.refresh_every
            if a.per_epoch:
                c["oracle"].update({"refresh_every_eval": 1, "replicates": 2,
                                    "settle_slots": 8, "measure_slots": 24})
            ex = Experiment(c, M.get(name), reg, out / name / f"seed{sd}",
                            seed=sd, prior=prior, telemetry=False,
                            log=lambda *z: None)
            s = ex.run()
            w.append(s["wIF"]); x.append(s["safety_crossings"])
            wr.append(s["writes_per_epoch"])
        R[name], X[name], W[name] = np.mean(w), np.mean(x), np.mean(wr)
        log(f"{name:24s} {R[name]:8.4f} {X[name]:7.1f} {W[name]:7.2f}")

    og = R["oracle"] - R["intact-ra"]
    kg = R["b3-oracle"] - R["b3-scrambled"]
    rate_gap = abs(W["b3-oracle"] - W["b3-scrambled"])
    t_og = float(g.get("oracle_gain_min", 0.030))
    t_kg = float(g.get("knowledge_gain_min", 0.050))
    ctrl = max(R[m] for m in ("b3", "intact-ra", "intact-ra-agentic"))

    lines = [f"# Oracle diagnostic: {scen}", "",
             f"{cfg['run']['epochs'] if not a.epochs else a.epochs} epochs, "
             f"seeds {seeds}, oracle refreshed "
             f"{'every scored epoch' if a.per_epoch else f'every {a.refresh_every} epochs'}.",
             "",
             "| method | wIF | crossings | writes/ep |", "|---|---|---|---|"]
    for n in NAMES:
        lines.append(f"| {M.get(n).label} | {R[n]:.4f} | {X[n]:.1f} | "
                     f"{W[n]:.2f} |")
    lines += ["", "## Gates", "",
              f"- oracle - frozen INTACT-RA = **{og:+.4f}** "
              f"(need >= {t_og:.3f}) -> "
              f"{'PASS' if og >= t_og else 'FAIL'}",
              f"- true - scrambled slopes = **{kg:+.4f}** "
              f"(need >= {t_kg:.3f}) -> "
              f"{'PASS' if kg >= t_kg else 'FAIL'}; write-rate gap "
              f"{rate_gap:.2f}/epoch"
              + ("" if rate_gap < 0.15 else
                 "  **CONFOUNDED: the two arms do not write at the same "
                 "rate, so this comparison mixes knowledge with volume.**"),
              f"- best controller - all-reject = "
              f"**{ctrl - R['all-reject']:+.4f}**",
              f"- best controller - all-accept = "
              f"**{ctrl - R['all-accept']:+.4f}**", ""]
    if og < t_og:
        lines += ["## VERDICT: CASE A -- scenario rejected", "",
                  "Perfect knowledge of the current sensitivities buys "
                  f"{og:+.4f} wIF over a frozen table. Sensitivity "
                  "evolution is not decision-relevant here, so tuning the "
                  "agentic controller against this scenario would be "
                  "manufacturing a gain rather than measuring one.", "",
                  "Diagnose the scenario instead. Run "
                  "`scripts/probe_zerosum.py` first: if the marginal quota "
                  "increase nets a large positive, the pool does not bind, "
                  "and wIF will track write rate rather than slope "
                  "quality."]
        if R["oracle"] < R["all-reject"]:
            lines += ["", "Additionally, **the oracle loses to doing "
                      f"nothing** ({R['oracle']:.4f} vs "
                      f"{R['all-reject']:.4f}). With perfect slopes the "
                      "arbiter still cannot beat all-reject, so this is "
                      "not an estimation problem at all -- the plant is so "
                      "constrained that actuation is net-harmful."]
    else:
        lines += ["## VERDICT: CASE B -- exploitable", "",
                  f"Perfect slopes are worth {og:+.4f} wIF. The gap "
                  f"between the oracle and the agentic method "
                  f"({R['oracle'] - R['intact-ra-agentic']:+.4f}) is what "
                  "better estimation could still buy. Agent work is "
                  "justified."]
    write_text(out / "ORACLE_DIAGNOSTIC.md", "\n".join(lines))
    write_json(out / "diagnostic.json",
               {"wIF": R, "crossings": X, "writes_per_epoch": W,
                "oracle_gain": og, "knowledge_gain": kg,
                "rate_gap": rate_gap,
                "case": "B" if og >= t_og else "A"})
    log("")
    log("\n".join(lines[lines.index("## Gates"):]))
    log(f"\nwritten to {out / 'ORACLE_DIAGNOSTIC.md'}")
    log.close()


if __name__ == "__main__":
    main()
