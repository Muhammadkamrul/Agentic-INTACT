#!/usr/bin/env python3
"""ROBUSTNESS STUDY: does INTACT-RA-Agentic match the best frozen INTACT-RA
configuration on plants where DIFFERENT frozen configurations win?  [AUTO]

The claim being tested
----------------------
No single frozen INTACT-RA configuration is safe across plants: on one
plant a per-tenant regime wins, on another a cell-level regime wins, and
choosing wrong in advance is costly.  INTACT-RA-Agentic, which keeps its
sensitivities current, should match the best frozen configuration on BOTH
without being told which one fits.

What it does, in order (each stage is automatic; nothing is edited for you)
  1. PREFLIGHT   for each scenario, the arbiter-free validity checks:
                 floor/ceiling (control_gain.py) and attainability
                 (intent_attainability.py).  No controller runs here.
  2. HELD-OUT    every method on the held-out seeds (run_heldout.py,
                 resumable, fingerprinted): all-reject, B0, B3, INTACT-RA
                 (per-tenant, as published), INTACT-RA (cell regime),
                 INTACT-RA (per-tenant sweep), INTACT-RA-Agentic, oracle.
  3. VERDICT     ROBUSTNESS_REPORT.md: per-scenario results, paired
                 comparisons with bootstrap 95% intervals, and a verdict on
                 four criteria FIXED IN THIS FILE BEFORE ANY RESULT:

  R1  each scenario is a valid benchmark (preflight passes)
  R2  the frozen ordering genuinely flips: in one scenario per-tenant
      INTACT-RA beats the cell variant with a CI above 0, and in the other
      the cell variant beats per-tenant INTACT-RA with a CI above 0
  R3  INTACT-RA-Agentic is NOT significantly worse than the best frozen
      configuration in ANY scenario (CI upper bound >= 0)
  R4  in EACH scenario INTACT-RA-Agentic beats, with a CI above 0, the frozen
      configuration you would have chosen by calibrating on the OTHER
      scenario -- the concrete cost of a wrong a-priori choice

  SUPPORTED only if R1-R4 all pass.  Any failure is reported as such, with
  the criterion and the numbers.  Too few seeds -> INCONCLUSIVE.

Usage (defaults run S16 and S19 on five held-out seeds, ~4-5 h on one core;
S19's floor/ceiling gap is +0.139, below the default 0.20 -- see --min-gap)
    python scripts/robustness_study.py
    # reuse S16 results already produced (the package ships them):
    python scripts/robustness_study.py --reuse S16_high_ceiling=results/S16_high_ceiling/heldout,results/S16_high_ceiling/heldout_cell,results/S16_high_ceiling/heldout_ablations
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from _common import ROOT
from heldout_report import paired

METHODS = ("all-reject,all-accept,b3,intact-ra,intact-ra-cell,"
           "intact-ra-pertenant-sweep,intact-ra-agentic,oracle")
FROZEN = ("intact-ra", "intact-ra-cell", "intact-ra-pertenant-sweep", "b3")
LABEL = {"all-reject": "All-reject", "all-accept": "B0 all-admit", "b3": "B3",
         "intact-ra": "INTACT-RA (per-tenant, published)",
         "intact-ra-cell": "INTACT-RA (cell regime)",
         "intact-ra-pertenant-sweep": "INTACT-RA (per-tenant sweep)",
         "intact-ra-agentic": "INTACT-RA-Agentic", "oracle": "Oracle*"}
# thresholds, fixed before any robustness result was seen
WANT_CEILING, WANT_GAP, MIN_SEEDS = 0.85, 0.20, 3


def sh(cmd):
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    return subprocess.call([str(c) for c in cmd], cwd=ROOT)


def preflight(scen, dev_seeds, min_gap=WANT_GAP):
    """Arbiter-free validity: floor/ceiling and attainability."""
    sh([sys.executable, ROOT / "scripts" / "control_gain.py", "--scenario",
        scen, "--seeds", dev_seeds])
    sh([sys.executable, ROOT / "scripts" / "intent_attainability.py",
        "--scenario", scen])
    res = ROOT / "results" / scen
    cg = json.loads((res / "_control_gain" / "control_gain.json").read_text())
    floor, ceil = float(np.mean(cg["floor"])), float(np.mean(cg["ceiling"]))
    att = (res / "_attainability" / "ATTAINABILITY.md").read_text()
    attainable = "All intents are attainable" in att
    ok = ceil >= WANT_CEILING and ceil - floor >= min_gap and attainable
    return {"floor": floor, "ceiling": ceil, "gap": ceil - floor,
            "attainable": attainable, "valid": ok}


def load_runs(dirs):
    runs = {}
    for d in dirs:
        p = Path(d) / "runs.jsonl"
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            r = json.loads(line)
            runs.setdefault(r["m"], {})[r["s"]] = r
    return runs


def mean_if(runs, m):
    v = [r["IF"] for r in runs.get(m, {}).values()]
    return float(np.mean(v)) if v else float("nan")


def fmt(q):
    return ("—" if q is None else
            f"{q['mean']:+.3f} [{q['lo']:+.3f}, {q['hi']:+.3f}], {q['wins']}/{q['n']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", default="S16_high_ceiling,S19_persistent_drift")
    ap.add_argument("--seeds", default="31001,31002,31003,31004,31005")
    ap.add_argument("--dev-seeds", default="20260925,20260926,20260927")
    ap.add_argument("--reuse", action="append", default=[],
                    help="SCENARIO=dir1,dir2,... of existing runs.jsonl to merge")
    ap.add_argument("--skip-preflight", action="store_true")
    ap.add_argument("--min-gap", type=float, default=WANT_GAP,
                    help="validity gap threshold (default 0.20). Changing it is a "
                         "validity decision and is RECORDED in the report.")
    ap.add_argument("--allow-code-change", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results" / "_robustness"))
    a = ap.parse_args()
    scen = [s.strip() for s in a.scenarios.split(",") if s.strip()]
    if len(scen) != 2:
        sys.exit("the robustness verdict compares exactly two scenarios")
    reuse = {k: v.split(",") for k, v in (x.split("=", 1) for x in a.reuse)}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    pre, data = {}, {}
    for s in scen:
        print(f"\n{'=' * 70}\n{s}\n{'=' * 70}", flush=True)
        if a.skip_preflight:
            pre[s] = None
        else:
            pre[s] = preflight(s, a.dev_seeds, a.min_gap)
            p = pre[s]
            print(f"PREFLIGHT {s}: floor {p['floor']:.3f}, ceiling "
                  f"{p['ceiling']:.3f}, gap {p['gap']:+.3f}, attainable "
                  f"{p['attainable']} -> {'VALID' if p['valid'] else 'NOT VALID'}")
        dirs = reuse.get(s, [])
        if not dirs:
            d = ROOT / "results" / s / "robustness_heldout"
            cmd = [sys.executable, ROOT / "scripts" / "run_heldout.py",
                   "--scenario", s, "--seeds", a.seeds, "--methods", METHODS,
                   "--out", d]
            if a.allow_code_change:
                cmd.append("--allow-code-change")
            rc = sh(cmd)
            if rc != 0:
                print(f"  run_heldout exited {rc}; reporting what exists")
            dirs = [d]
        data[s] = load_runs(dirs)

    # ---------------------------------------------------------------- report
    L = ["# Robustness study", "",
         f"Scenarios: {scen}. Held-out seeds: {a.seeds}. Paired differences "
         f"with bootstrap 95% intervals (difference [CI], seeds ahead/seeds).",
         "", "## 1. Validity (arbiter-free, no controller involved)", "",
         "| scenario | floor | ceiling | gap | attainable | valid |",
         "|---|---|---|---|---|---|"]
    for s in scen:
        p = pre[s]
        L.append(f"| {s} | " + (" | ".join([f"{p['floor']:.3f}",
                 f"{p['ceiling']:.3f}", f"{p['gap']:+.3f}",
                 str(p["attainable"]), "**yes**" if p["valid"] else "**NO**"])
                 if p else "not run | — | — | — | —") + " |")
    L += ["", f"Thresholds: ceiling >= {WANT_CEILING}, gap >= {a.min_gap}, every "
          f"intent attainable within its envelope."
          + (f" **The gap threshold was changed from the default {WANT_GAP} to "
             f"{a.min_gap}; this is a validity decision and must be reported "
             f"with any result.**" if a.min_gap != WANT_GAP else ""), "",
          "## 2. Fulfilment", "",
          "| method | " + " | ".join(scen) + " |", "|---|" + "---|" * len(scen)]
    for m in METHODS.split(","):
        L.append(f"| {LABEL[m]} | " + " | ".join(
            f"{mean_if(data[s], m):.3f} (n={len(data[s].get(m, {}))})"
            if data[s].get(m) else "—" for s in scen) + " |")

    best = {}
    for s in scen:
        c = [(mean_if(data[s], m), m) for m in FROZEN if data[s].get(m)]
        best[s] = max(c)[1] if c else None
    L += ["", "## 3. Key paired comparisons", "",
          "| comparison | " + " | ".join(scen) + " |", "|---|" + "---|" * len(scen)]
    rows = [("INTACT-RA (per-tenant) − INTACT-RA (cell)", "intact-ra", "intact-ra-cell"),
            ("INTACT-RA-Agentic − INTACT-RA (per-tenant)", "intact-ra-agentic", "intact-ra"),
            ("INTACT-RA-Agentic − INTACT-RA (cell)", "intact-ra-agentic", "intact-ra-cell"),
            ("INTACT-RA-Agentic − B3", "intact-ra-agentic", "b3"),
            ("oracle − best frozen", "oracle", None)]
    for name, x, y in rows:
        cells = []
        for s in scen:
            yy = y or best[s]
            cells.append(fmt(paired(data[s], x, yy)) if yy else "—")
        L.append(f"| {name} | " + " | ".join(cells) + " |")

    # ---------------------------------------------------------------- verdict
    V = {}
    V["R1"] = all(pre[s] and pre[s]["valid"] for s in scen)
    q_pt = {s: paired(data[s], "intact-ra", "intact-ra-cell") for s in scen}
    enough = all(q and q["n"] >= MIN_SEEDS for q in q_pt.values())
    pt_wins = [s for s in scen if q_pt[s] and q_pt[s]["lo"] > 0]
    cell_wins = [s for s in scen if q_pt[s] and q_pt[s]["hi"] < 0]
    V["R2"] = bool(pt_wins) and bool(cell_wins)
    r3, r4, lines3, lines4 = True, True, [], []
    for s in scen:
        other = [o for o in scen if o != s][0]
        qb = paired(data[s], "intact-ra-agentic", best[s]) if best[s] else None
        ok3 = qb is not None and qb["hi"] >= 0
        r3 &= ok3
        lines3.append(f"  - {s}: best frozen is {LABEL.get(best[s], best[s])}; "
                      f"Agentic − it = {fmt(qb)} -> "
                      f"{'not worse' if ok3 else 'SIGNIFICANTLY WORSE'}")
        wrong = best[other]
        qw = paired(data[s], "intact-ra-agentic", wrong) if wrong else None
        ok4 = qw is not None and qw["lo"] > 0
        r4 &= ok4
        lines4.append(f"  - {s}: calibrating on {other} would pick "
                      f"{LABEL.get(wrong, wrong)}; Agentic − it = {fmt(qw)} -> "
                      f"{'beats it' if ok4 else 'does NOT beat it'}")
    V["R3"], V["R4"] = r3, r4
    L += ["", "## 4. Verdict on the robustness claim", "",
          f"- **R1 valid benchmarks:** {'PASS' if V['R1'] else 'FAIL'}",
          f"- **R2 the frozen ordering flips:** {'PASS' if V['R2'] else 'FAIL'} "
          f"(per-tenant significantly better in {pt_wins or 'none'}; cell "
          f"significantly better in {cell_wins or 'none'})",
          f"- **R3 never significantly worse than the best frozen:** "
          f"{'PASS' if V['R3'] else 'FAIL'}", *lines3,
          f"- **R4 beats the wrong a-priori choice in each plant:** "
          f"{'PASS' if V['R4'] else 'FAIL'}", *lines4, ""]
    if not enough:
        head = ("**ROBUSTNESS CLAIM INCONCLUSIVE** — fewer than "
                f"{MIN_SEEDS} paired seeds in at least one scenario.")
    elif all(V.values()):
        head = ("**ROBUSTNESS CLAIM SUPPORTED.** Different frozen configurations "
                "win on the two plants, and INTACT-RA-Agentic matches the best "
                "on both while beating the wrong a-priori choice.")
    else:
        failed = [k for k, v in V.items() if not v]
        head = (f"**ROBUSTNESS CLAIM NOT SUPPORTED** (failed: {', '.join(failed)}). "
                f"Report the S16 result on its own terms; see FINDINGS.md.")
        if "R2" in failed:
            head += (" R2 failing means the two plants do not favour different "
                     "frozen configurations, so they cannot test the claim at all.")
    L += [head, "",
          "*Oracle: INTACT-RA's own rule with true current slopes; a "
          "perfect-knowledge reference, not an upper bound."]
    (out / "ROBUSTNESS_REPORT.md").write_text("\n".join(L))
    (out / "robustness.json").write_text(json.dumps(
        {"verdict": V, "preflight": pre, "best_frozen": best}, indent=2, default=str))
    print("\n" + "\n".join(L))

    # figures via the suite script, on the merged runs
    try:
        args = [sys.executable, ROOT / "scripts" / "run_suite.py", "--scenarios",
                ",".join(scen), "--aggregate-only", "--out", out]
        for s in scen:
            merged = out / f"merged_{s}"
            merged.mkdir(exist_ok=True)
            with open(merged / "runs.jsonl", "w") as fh:
                for m, per in data[s].items():
                    for r in per.values():
                        fh.write(json.dumps(r) + "\n")
            args += ["--reuse", f"{s}={merged}"]
        sh(args)
    except Exception as e:
        print(f"figures skipped: {e}")


if __name__ == "__main__":
    main()
