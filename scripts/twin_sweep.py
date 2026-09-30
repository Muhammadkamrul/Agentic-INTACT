#!/usr/bin/env python3
"""TWIN SWEEP: how good, and how fresh, must the digital twin be?  [AUTO]

INTACT-RA-Agentic's gain comes from its digital-twin calibration loop (the
no-twin ablation scores no better than frozen INTACT-RA).  A reviewer will
therefore ask two things, and this script answers both on held-out seeds:

  FIDELITY  how wrong can the twin be?  It re-runs INTACT-RA-Agentic with
            larger UE-position errors, noise-figure errors and neighbour-
            load errors, and all three degraded together.
  CADENCE   how often must it calibrate?  It re-runs with the twin called
            every 1, 3, 6 and 12 epochs (160 ms each).  Calibrating every 6
            epochs (~1 s) approximates hosting the twin in the non-RT RIC
            and pushing sensitivities over A1.  NOTE: cadence is a PROXY
            for lag -- between calls the table is up to N epochs stale --
            not an explicit delay queue.

Each setting is a separate held-out evaluation (run_heldout.py with --set
overrides), so each has its own fingerprinted manifest and can never be
mixed with another.  Baselines do not depend on the twin, so they are
reused from the main S16 held-out runs.

Output: results/<scenario>/twin_sweep/TWIN_SWEEP.md with, per setting,
INTACT-RA-Agentic's fulfilment and its paired difference (bootstrap 95% CI)
from INTACT-RA (per-tenant), INTACT-RA (cell regime) and the default twin,
plus an automatic statement of where, if anywhere, the advantage is lost.

Time: ~7 min per setting per seed on one core.  The full sweep (9 settings
x 5 seeds) is about 5 hours; --quick uses 3 seeds.  Resumable.

Usage
    python scripts/twin_sweep.py
    python scripts/twin_sweep.py --quick
    python scripts/twin_sweep.py --only pos100,every6
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from _common import ROOT
from heldout_report import paired

SETTINGS = {
    # name: (description, overrides)
    "pos50":    ("UE position error 50 m (default 25 m)",
                 ["agent.twin.pos_sigma_m=50"]),
    "pos100":   ("UE position error 100 m",
                 ["agent.twin.pos_sigma_m=100"]),
    "nf3":      ("noise-figure error s.d. 3 dB (default 1 dB)",
                 ["agent.twin.nf_err_sd_db=3.0"]),
    "load05":   ("neighbour-load error s.d. 0.5 (default 0.2)",
                 ["agent.twin.load_err_sd=0.5"]),
    "poor":     ("all three degraded: 100 m, 3 dB, 0.5",
                 ["agent.twin.pos_sigma_m=100", "agent.twin.nf_err_sd_db=3.0",
                  "agent.twin.load_err_sd=0.5"]),
    "every3":   ("twin every 3 epochs (0.48 s)",
                 ["agent.twin.every_eval=3"]),
    "every6":   ("twin every 6 epochs (~1 s; non-RT RIC / A1-like)",
                 ["agent.twin.every_eval=6"]),
    "every12":  ("twin every 12 epochs (~2 s)",
                 ["agent.twin.every_eval=12"]),
}


def load_runs(dirs):
    runs = {}
    for d in dirs:
        p = Path(d) / "runs.jsonl"
        if p.exists():
            for line in p.read_text().splitlines():
                r = json.loads(line)
                runs.setdefault(r["m"], {})[r["s"]] = r
    return runs


def fmt(q):
    return ("—" if q is None else
            f"{q['mean']:+.3f} [{q['lo']:+.3f}, {q['hi']:+.3f}], {q['wins']}/{q['n']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="S16_high_ceiling")
    ap.add_argument("--seeds", default="31001,31002,31003,31004,31005")
    ap.add_argument("--quick", action="store_true", help="3 seeds only")
    ap.add_argument("--only", default=None, help="comma list of setting names")
    ap.add_argument("--reference", default=None,
                    help="comma list of dirs with baseline + default-twin runs "
                         "(default: the main held-out, cell and ablation dirs)")
    ap.add_argument("--allow-code-change", action="store_true")
    a = ap.parse_args()
    seeds = a.seeds.split(",")[:3] if a.quick else a.seeds.split(",")
    names = a.only.split(",") if a.only else list(SETTINGS)
    res = ROOT / "results" / a.scenario
    out = res / "twin_sweep"
    out.mkdir(parents=True, exist_ok=True)
    ref_dirs = (a.reference.split(",") if a.reference else
                [res / "heldout", res / "heldout_cell", res / "heldout_ablations"])
    ref = load_runs(ref_dirs)
    for need in ("intact-ra-agentic", "intact-ra", "intact-ra-cell"):
        if not ref.get(need):
            print(f"WARNING: reference runs for {need} not found in {ref_dirs}")

    for n in names:
        desc, ov = SETTINGS[n]
        d = out / n
        cmd = [sys.executable, str(ROOT / "scripts" / "run_heldout.py"),
               "--scenario", a.scenario, "--seeds", ",".join(seeds),
               "--methods", "intact-ra-agentic", "--out", str(d), "--set", *ov]
        if a.allow_code_change:
            cmd.append("--allow-code-change")
        print(f"\n=== {n}: {desc}\n$ " + " ".join(cmd), flush=True)
        subprocess.call(cmd, cwd=ROOT)

    L = ["# Twin sweep: how good and how fresh must the digital twin be?", "",
         f"Scenario `{a.scenario}`, held-out seeds {seeds}. Each row re-runs "
         f"INTACT-RA-Agentic with one twin setting changed; baselines are the "
         f"main held-out runs. Paired differences: mean [95% CI], seeds ahead.",
         "", "| setting | description | IF | vs INTACT-RA (per-tenant) | "
         "vs INTACT-RA (cell) | vs default twin |", "|---|---|---|---|---|---|"]
    base_if = [ref["intact-ra-agentic"][int(s)]["IF"] for s in seeds
               if int(s) in ref.get("intact-ra-agentic", {})]
    L.append(f"| default | 25 m, 1 dB, 0.2, every epoch | "
             f"{np.mean(base_if):.3f} | "
             f"{fmt(paired(ref, 'intact-ra-agentic', 'intact-ra'))} | "
             f"{fmt(paired(ref, 'intact-ra-agentic', 'intact-ra-cell'))} | — |")
    lost_ra, lost_cell = [], []
    for n in names:
        runs = load_runs([out / n])
        if not runs.get("intact-ra-agentic"):
            L.append(f"| {n} | {SETTINGS[n][0]} | not run | | | |")
            continue
        merged = {**ref, "tw": runs["intact-ra-agentic"]}
        q_ra = paired(merged, "tw", "intact-ra")
        q_cell = paired(merged, "tw", "intact-ra-cell")
        q_def = paired(merged, "tw", "intact-ra-agentic")
        v = [r["IF"] for r in runs["intact-ra-agentic"].values()]
        L.append(f"| {n} | {SETTINGS[n][0]} | {np.mean(v):.3f} | {fmt(q_ra)} | "
                 f"{fmt(q_cell)} | {fmt(q_def)} |")
        if q_ra is None or q_ra["lo"] <= 0:
            lost_ra.append(n)
        if q_cell is not None and q_cell["hi"] < 0:
            lost_cell.append(n)
    L += ["", "## Interpretation", ""]
    ran = [n for n in names if load_runs([out / n]).get("intact-ra-agentic")]
    L.append("- **Advantage over INTACT-RA as published** (CI above 0) is "
             + ("kept at every tested setting." if ran and not lost_ra else
                f"**lost** at: {', '.join(lost_ra)}." if lost_ra else "not yet measured."))
    L.append("- **Significantly worse than the cell-regime variant** at: "
             + (", ".join(lost_cell) if lost_cell else "none of the tested settings."))
    L.append("- Cadence rows show how stale the table may become before the "
             "advantage erodes; `every6` approximates a twin hosted in the "
             "non-RT RIC pushing sensitivities over A1 (a proxy for lag).")
    (out / "TWIN_SWEEP.md").write_text("\n".join(L))
    print("\n" + "\n".join(L))


if __name__ == "__main__":
    main()
