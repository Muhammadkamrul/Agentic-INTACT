#!/usr/bin/env python3
"""STEP 5. Assemble the written report for one scenario.

Produces a single self-contained REPORT.md holding: the plant description
with its measured operating point, the drift evidence, the benchmark
tables, the gate verdicts, the auto-interpretation and an index of every
figure. One document, no cross-referenced addenda.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from _common import ROOT, Log, common_args, load, results_dir, write_text


def main():
    p = common_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--telemetry-from", default="intact-ra-agentic")
    a = p.parse_args()

    from intact_agentic.config import scenario_name, config_fingerprint
    from intact_agentic.report import interpret as I
    from intact_agentic.telemetry import write_metric_catalogue

    cfg = load(a)
    root = results_dir(a, cfg)
    log = Log(root / "report.log", a.quiet)
    sp = root / "summaries.json"
    if not sp.exists():
        log("no summaries.json; run run_benchmark.py first")
        return
    summaries = json.loads(sp.read_text())
    paired = json.loads((root / "paired.json").read_text()) \
        if (root / "paired.json").exists() else None
    cis = json.loads((root / "bootstrap_ci.json").read_text()) \
        if (root / "bootstrap_ci.json").exists() else None

    parts = [f"# INTACT-RA-Agentic results: {scenario_name(cfg)}", "",
             f"Config fingerprint `{config_fingerprint(cfg)}`. "
             f"{cfg['run']['epochs']} epochs, seeds "
             f"{list(cfg['run'].get('seeds', []))}.", ""]

    # ---- headline table ---------------------------------------------------
    parts += ["## Headline results", "",
              "| method | wIF | worst intent | safety crossings | "
              "writes/epoch | pred. MAE | p95 latency (ms) | candidates |",
              "|---|---|---|---|---|---|---|---|"]
    order = sorted(summaries, key=lambda m: -summaries[m].get("wIF", 0))
    for m in order:
        s = summaries[m]
        def f(k, nd=4):
            v = s.get(k)
            return "n/a" if v is None or not np.isfinite(float(v)) \
                else f"{float(v):.{nd}f}"
        parts.append(f"| {s.get('label', m)} | **{f('wIF')}** | "
                     f"{f('worst_intent_fulfilment', 3)} | "
                     f"{f('safety_crossings', 0)} | "
                     f"{f('writes_per_epoch', 2)} | {f('prediction_mae')} | "
                     f"{f('latency_ms_p95', 2)} | "
                     f"{f('candidates_mean', 1)} |")
    parts.append("")

    if cis:
        parts += ["### Bootstrap confidence intervals (over seeds)", "",
                  "| method | wIF mean | 95% CI | n |", "|---|---|---|---|"]
        for m in order:
            c = (cis.get(m) or {}).get("wIF")
            if c:
                parts.append(f"| {summaries[m].get('label', m)} | "
                             f"{c['mean']:.4f} | "
                             f"[{c['lo']:.4f}, {c['hi']:.4f}] | {c['n']} |")
        parts.append("")

    # ---- the plant --------------------------------------------------------
    tel = root / a.telemetry_from / \
        f"seed{a.seed if a.seed is not None else cfg['run']['seed']}" / "csv"
    kpm = tel / "epoch_kpm.csv"
    if kpm.exists():
        df = pd.read_csv(kpm)
        tstats = {}
        for t in sorted(cfg["ran"]["slices"]):
            d = df[df["tenant"] == t]
            if len(d):
                tstats[t] = {c: float(d[c].mean())
                             for c in d.columns
                             if d[c].dtype.kind in "fi"}
        cstats = {}
        dc = df[df["tenant"] == "_cell"]
        if len(dc):
            cstats = {c: float(dc[c].mean()) for c in dc.columns
                      if dc[c].dtype.kind in "fi"}
        parts += [I.interpret_ran(cstats, tstats, cfg), ""]

    for extra in ("DRIFT.md", "INTERPRETATION.md", "GATES.md"):
        f = root / extra
        if f.exists():
            parts += [f.read_text(), ""]

    figs = sorted((root / "figures").glob("*.png"))
    if figs:
        parts += ["## Figures", ""]
        for f in figs:
            parts.append(f"![{f.stem}](figures/{f.name})")
            parts.append("")

    write_text(root / "REPORT.md", "\n".join(parts))
    write_metric_catalogue(ROOT / "docs" / "METRICS.md")
    log(f"wrote {root / 'REPORT.md'}")
    log(f"wrote {ROOT / 'docs' / 'METRICS.md'}")
    log.close()


if __name__ == "__main__":
    main()
