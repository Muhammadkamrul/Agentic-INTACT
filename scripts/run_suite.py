#!/usr/bin/env python3
"""Cross-scenario suite: the same methods, the same held-out seeds, several
scenarios, one report.  [AUTO]

For each scenario it runs (resumably) every method on the held-out seeds via
run_heldout.py, then writes a single SUITE_REPORT.md and figures that state,
PER SCENARIO, with paired bootstrap 95% intervals:

  * INTACT-RA-Agentic vs INTACT-RA      (the headline)
  * INTACT-RA-Agentic vs B3
  * INTACT-RA vs B3                     (per-tenant vs cell-level regime)
  * oracle vs INTACT-RA                 (are current sensitivities decision-
                                         relevant here at all?)

Nothing is assumed.  A scenario where INTACT-RA-Agentic does not win is
reported as such, and a scenario where even the oracle cannot beat the
frozen table (CASE A) is flagged: there, equal performance is the correct
expectation and no method should be credited with a gain.

Results from a DIFFERENT simulator -- for example the original INTACT
experiments -- can be added with --external NAME=path.csv (columns:
method,seed,IF).  They appear in their own clearly labelled column, only for
the methods actually run there; nothing is imputed for the others.

Usage
    python scripts/run_suite.py --scenarios S8_dedicated,S16_high_ceiling \\
        --seeds 31001,31002,31003,31004,31005
    # reuse an existing held-out run instead of re-running it:
    python scripts/run_suite.py --scenarios S16_high_ceiling \\
        --reuse S16_high_ceiling=results/S16_high_ceiling/heldout \\
        --aggregate-only
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from _common import ROOT
from heldout_report import paired

METHODS = "all-reject,all-accept,b3,intact-ra,intact-ra-cell,intact-ra-pertenant-sweep,intact-ra-agentic,oracle"
LABEL = {"oracle": "Oracle*", "intact-ra-agentic": "INTACT-RA-Agentic",
         "b3": "B3", "intact-ra": "INTACT-RA",
         "intact-ra-cell": "INTACT-RA (cell regime)",
         "intact-ra-pertenant-sweep": "INTACT-RA (per-tenant sweep)",
         "all-reject": "All-reject",
         "all-accept": "B0 all-admit"}
FROZEN = ("intact-ra", "intact-ra-cell", "intact-ra-pertenant-sweep", "b3")
COMPARISONS = [("intact-ra-agentic", "intact-ra", "Agentic − INTACT-RA"),
               ("intact-ra-agentic", "intact-ra-cell", "Agentic − INTACT-RA (cell)"),
               ("intact-ra-cell", "intact-ra", "INTACT-RA (cell) − INTACT-RA"),
               ("intact-ra-agentic", "b3", "Agentic − B3"),
               ("intact-ra", "b3", "INTACT-RA − B3"),
               ("oracle", "intact-ra", "oracle − INTACT-RA")]
CASE_A_THRESHOLD = 0.03


def load_runs(path: Path):
    runs = {}
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            runs.setdefault(r["m"], {})[r["s"]] = r
    return runs


def load_external(path: Path):
    runs = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            runs.setdefault(row["method"], {})[int(row["seed"])] = {
                "IF": float(row["IF"])}
    return runs


def verdict(q):
    if q is None:
        return "not run"
    if q["n"] < 3:
        return f"too few seeds ({q['n']})"
    if q["lo"] > 0:
        return "positive (CI above 0)"
    if q["hi"] < 0:
        return "negative (CI below 0)"
    return "no detectable difference"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", required=True)
    ap.add_argument("--seeds", default="31001,31002,31003,31004,31005")
    ap.add_argument("--methods", default=METHODS)
    ap.add_argument("--reuse", action="append", default=[],
                    help="SCENARIO=dir holding runs.jsonl from an earlier run")
    ap.add_argument("--external", action="append", default=[],
                    help="NAME=path.csv of results from another simulator")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--allow-code-change", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results" / "_suite"))
    a = ap.parse_args()

    scen = [s.strip() for s in a.scenarios.split(",") if s.strip()]
    reuse = dict(x.split("=", 1) for x in a.reuse)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    dirs = {}
    for s in scen:
        d = Path(reuse[s]) if s in reuse else ROOT / "results" / s / "suite_heldout"
        dirs[s] = d
        if a.aggregate_only or s in reuse:
            continue
        cmd = [sys.executable, str(ROOT / "scripts" / "run_heldout.py"),
               "--scenario", s, "--seeds", a.seeds, "--methods", a.methods,
               "--out", str(d)]
        if a.allow_code_change:
            cmd.append("--allow-code-change")
        print(f"\n=== {s} ===\n" + " ".join(cmd), flush=True)
        rc = subprocess.call(cmd, cwd=ROOT)
        if rc != 0:
            print(f"  {s}: run_heldout.py exited with {rc}; reporting what exists")

    data = {s: load_runs(dirs[s] / "runs.jsonl") for s in scen}
    for x in a.external:
        name, path = x.split("=", 1)
        data[f"{name} (external simulator)"] = load_external(Path(path))
    cols = list(data)

    L = ["# Cross-scenario suite", "",
         f"Held-out seeds: {a.seeds}. Paired differences with bootstrap 95% "
         f"intervals, over seeds both methods completed. Note: the "
         f"oracle is INTACT-RA's own decision rule reading true current "
         f"slopes -- a perfect-knowledge reference, not an upper bound.", "",
         "## Fulfilment by scenario", "",
         "| method | " + " | ".join(cols) + " |",
         "|---|" + "---|" * len(cols)]
    for m in METHODS.split(","):
        cells = []
        for c in cols:
            v = [r["IF"] for r in data[c].get(m, {}).values()]
            cells.append(f"{np.mean(v):.3f} (n={len(v)})" if v else "—")
        L.append(f"| {LABEL.get(m, m)} | " + " | ".join(cells) + " |")
    L += ["", "## Paired comparisons by scenario", "",
          "| comparison | " + " | ".join(cols) + " |",
          "|---|" + "---|" * len(cols)]
    summary = {}
    for x, y, name in COMPARISONS:
        cells = []
        for c in cols:
            q = paired(data[c], x, y)
            summary[(name, c)] = q
            cells.append("—" if q is None else
                         f"{q['mean']:+.3f} [{q['lo']:+.3f}, {q['hi']:+.3f}] "
                         f"{q['wins']}/{q['n']} — {verdict(q)}")
        L.append(f"| {name} | " + " | ".join(cells) + " |")

    # THE comparison a reviewer asks for: against the BEST frozen
    # configuration in each scenario, chosen after the fact by mean IF.
    # Choosing it post hoc is deliberately biased AGAINST the agent.
    L += ["", "## INTACT-RA-Agentic vs the best frozen configuration", "",
          "The best of INTACT-RA (per-tenant), INTACT-RA (cell regime) and B3 "
          "in each scenario, chosen AFTER seeing the results -- a comparison "
          "deliberately biased against the agent.", "",
          "| scenario | best frozen | its IF | Agentic − best frozen | verdict |",
          "|---|---|---|---|---|"]
    for c in cols:
        cands = [(np.mean([r["IF"] for r in data[c][m].values()]), m)
                 for m in FROZEN if data[c].get(m)]
        if not cands or not data[c].get("intact-ra-agentic"):
            L.append(f"| {c} | — | — | — | not run |")
            continue
        bf, bm = max(cands)
        q = paired(data[c], "intact-ra-agentic", bm)
        summary[("Agentic − best frozen", c)] = q
        L.append(f"| {c} | {LABEL[bm]} | {bf:.3f} | "
                 + (f"{q['mean']:+.3f} [{q['lo']:+.3f}, {q['hi']:+.3f}] "
                    f"{q['wins']}/{q['n']}" if q else "—")
                 + f" | {verdict(q)} |")
    L += ["", "## What each scenario shows", ""]
    for c in cols:
        qo = summary.get(("oracle − INTACT-RA", c))
        qh = summary.get(("Agentic − INTACT-RA", c))
        qb = summary.get(("INTACT-RA − B3", c))
        if qo is not None and qo["n"] >= 3 and qo["mean"] < CASE_A_THRESHOLD:
            L.append(f"- **{c}: CASE A.** Even perfect current slopes beat the "
                     f"frozen table by only {qo['mean']:+.3f}; current "
                     f"sensitivities are not decision-relevant here, and no "
                     f"method should be credited with a gain over INTACT-RA.")
        elif qh is not None:
            L.append(f"- **{c}:** INTACT-RA-Agentic vs INTACT-RA "
                     f"{verdict(qh)}; INTACT-RA vs B3 {verdict(qb)}.")
        else:
            L.append(f"- **{c}:** INTACT-RA-Agentic not run here; "
                     f"INTACT-RA vs B3 {verdict(qb)}.")
    n_win = sum(1 for c in cols if summary.get(("Agentic − INTACT-RA", c))
                and summary[("Agentic − INTACT-RA", c)]["n"] >= 3
                and summary[("Agentic − INTACT-RA", c)]["lo"] > 0)
    n_run = sum(1 for c in cols if summary.get(("Agentic − INTACT-RA", c))
                and summary[("Agentic − INTACT-RA", c)]["n"] >= 3)
    L += ["", f"**INTACT-RA-Agentic beats INTACT-RA with a CI above zero in "
              f"{n_win} of {n_run} scenario(s) where both were run.** Any "
              f"claim of winning 'in every scenario' requires this to read "
              f"{n_run} of {n_run}."]
    (out / "SUITE_REPORT.md").write_text("\n".join(L))
    print("\n".join(L))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(max(6, 2.2 * len(cols)), 3.6))
        ms = [m for m in METHODS.split(",")]
        w = 0.8 / len(ms)
        for k, m in enumerate(ms):
            xs, ys, es = [], [], []
            for i, c in enumerate(cols):
                v = [r["IF"] for r in data[c].get(m, {}).values()]
                if v:
                    xs.append(i + (k - len(ms) / 2) * w + w / 2)
                    ys.append(np.mean(v))
                    es.append(np.std(v) / max(np.sqrt(len(v)), 1))
            if xs:
                ax.bar(xs, ys, width=w, yerr=es, capsize=2,
                       label=LABEL.get(m, m))
        ax.set_xticks(range(len(cols)), cols, fontsize=8)
        ax.set_ylabel("intent fulfilment (held-out)")
        ax.legend(fontsize=7, ncol=3)
        ax.set_ylim(0.4, 1.0)
        fig.tight_layout()
        fig.savefig(out / "fig_suite_if.png", dpi=150)

        fig, ax = plt.subplots(figsize=(7, 0.6 + 0.5 * len(cols) * len(COMPARISONS)))
        rows, y = [], 0
        for x_, y_, name in COMPARISONS:
            for c in cols:
                q = summary.get((name, c))
                if q is None:
                    continue
                ax.errorbar(q["mean"], y, xerr=[[q["mean"] - q["lo"]],
                                                [q["hi"] - q["mean"]]],
                            fmt="o", capsize=3)
                rows.append(f"{name} | {c}")
                y += 1
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(range(len(rows)), rows, fontsize=7)
        ax.set_xlabel("paired difference in IF (bootstrap 95% CI)")
        fig.tight_layout()
        fig.savefig(out / "fig_suite_forest.png", dpi=150)
        print(f"\nwrote {out/'fig_suite_if.png'} and {out/'fig_suite_forest.png'}")
    except Exception as e:                     # figures are a convenience
        print(f"figures skipped: {e}")


if __name__ == "__main__":
    main()
