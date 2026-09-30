#!/usr/bin/env python3
"""HELD-OUT evaluation: every method on seeds never used in development.  [AUTO]

Resumable.  Each (method, seed) result is appended to runs.jsonl as soon as
it finishes, so an interrupted evaluation continues from where it stopped
when the same command is re-run.

Two safeguards, both hard refusals rather than warnings:

  * it refuses to start if any held-out seed is also a development seed;
  * on the first run it writes manifest.json with the configuration
    fingerprint, and on every later run it refuses to resume if the
    fingerprint has changed -- so results from two configurations can never
    be mixed into one table.

Freeze the method and the scenario BEFORE running this.  Changing either in
response to held-out numbers turns the held-out seeds into development
seeds.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from _common import ROOT, common_args, load

DEFAULT_METHODS = ("all-reject,all-accept,b3,intact-ra,intact-ra-cell,"
                   "intact-ra-pertenant-sweep,intact-ra-agentic,oracle")


def code_fingerprint() -> str:
    """Hash of every source file in the package.  The configuration
    fingerprint cannot see a code change, and a code change can alter a
    frozen method's behaviour just as surely as a configuration change."""
    import hashlib
    h = hashlib.sha256()
    for f in sorted((ROOT / "intact_agentic").rglob("*.py")):
        h.update(str(f.relative_to(ROOT)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def main():
    p = common_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--seeds", default="31001,31002,31003,31004,31005",
                   help="held-out seeds (must not overlap --dev-seeds)")
    p.add_argument("--dev-seeds", default="20260925,20260926,20260927",
                   help="seeds used during development")
    p.add_argument("--methods", default=DEFAULT_METHODS)
    p.add_argument("--allow-code-change", action="store_true",
                   help="resume even though the package source changed; only "
                        "after verifying completed runs reproduce exactly")
    p.add_argument("--oracle-per-epoch", action="store_true", default=True,
                   help="refresh the oracle's slopes every scored epoch")
    a = p.parse_args()

    from intact_agentic import methods as M
    from intact_agentic.config import (build_registry, config_fingerprint,
                                       scenario_name)
    from intact_agentic.experiment import Experiment
    from intact_agentic.arbiter.sensitivity import StaticSensitivity

    held = [int(s) for s in a.seeds.split(",")]
    dev = {int(s) for s in a.dev_seeds.split(",")}
    clash = set(held) & dev
    if clash:
        sys.exit(f"REFUSING: held-out seeds overlap development seeds: "
                 f"{sorted(clash)}")

    cfg0 = load(a)
    scen = scenario_name(cfg0)
    out = Path(a.out) if a.out else ROOT / "results" / scen / "heldout"
    out.mkdir(parents=True, exist_ok=True)
    runs_path, man_path = out / "runs.jsonl", out / "manifest.json"
    fp = config_fingerprint(cfg0)
    if man_path.exists():
        man = json.loads(man_path.read_text())
        if man["fingerprint"] != fp:
            sys.exit(f"REFUSING TO RESUME: configuration fingerprint changed "
                     f"({man['fingerprint']} -> {fp}). Start a new output "
                     f"directory instead of mixing configurations.")
        if sorted(man["seeds"]) != sorted(held):
            sys.exit("REFUSING TO RESUME: a different held-out seed list was "
                     "given than the one recorded in the manifest.")
        cfp = code_fingerprint()
        if "code_fingerprint" in man and man["code_fingerprint"] != cfp:
            if not a.allow_code_change:
                sys.exit(f"REFUSING TO RESUME: the package source changed "
                         f"({man['code_fingerprint']} -> {cfp}). A code "
                         f"change can alter a frozen method. Verify the "
                         f"completed runs reproduce exactly with the new code, "
                         f"then re-run with --allow-code-change.")
            print(f"WARNING: code changed ({man['code_fingerprint']} -> {cfp}); "
                  f"continuing because --allow-code-change was given")
    else:
        man_path.write_text(json.dumps({
            "scenario": scen, "fingerprint": fp,
            "code_fingerprint": code_fingerprint(), "seeds": held,
            "dev_seeds": sorted(dev), "methods": a.methods.split(","),
            "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))

    reg = build_registry(cfg0)
    cache = ROOT / "artifacts" / f"prior_{scen}.json"
    if not cache.exists():
        from intact_agentic.experiment import build_prior
        prior = build_prior(cfg0, reg, log=print, cache=cache)
    else:
        prior = StaticSensitivity.load(cfg0, cache)

    done = set()
    if runs_path.exists():
        for line in runs_path.read_text().splitlines():
            d = json.loads(line)
            done.add((d["m"], d["s"]))

    for sd in held:                      # seed-major: each seed is complete
        for name in a.methods.split(","):
            if (name, sd) in done:
                continue
            c = load(a)
            c["run"].update({"log_every": 0, "checkpoint_every": 50})
            if name == "oracle" and a.oracle_per_epoch:
                c["oracle"].update({"refresh_every": 20,
                                    "refresh_every_eval": 1,
                                    "replicates": 2, "settle_slots": 8,
                                    "measure_slots": 24})
            t0 = time.time()
            ex = Experiment(c, M.get(name), reg, out / name / f"seed{sd}",
                            seed=sd, prior=prior, telemetry=False,
                            log=lambda *z: None)
            ex.load_checkpoint()
            s = ex.run()
            rec = {"m": name, "s": sd, "IF": s["IF"],
                   "short": s["mean_shortfall"],
                   "c1": s["c1_violations"], "c2": s["c2_violations"],
                   "cx": s["causal_crossings"],
                   "wpe": s["writes_per_epoch"],
                   "lat95": s["latency_ms_p95"],
                   "twin_s_per_call": s.get("twin_seconds_per_call"),
                   "wall_s": time.time() - t0,
                   "per": {k[11:]: round(v, 3) for k, v in s.items()
                           if k.startswith("fulfilment_")}}
            with open(runs_path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"seed {sd}  {name:18s} IF={rec['IF']:.4f}  "
                  f"C1={rec['c1']}  C2={rec['c2']}  causal={rec['cx']}  "
                  f"w/ep={rec['wpe']:.2f}  [{rec['wall_s']:.0f}s]", flush=True)

    print(f"\nall runs complete. Next:\n  python scripts/heldout_report.py "
          f"--runs {runs_path} --manifest {man_path} --out {out}")


if __name__ == "__main__":
    main()
