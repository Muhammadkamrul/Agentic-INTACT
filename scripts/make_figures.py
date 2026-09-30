#!/usr/bin/env python3
"""STEP 4. Build every figure from a completed results directory.

Reads only what the run wrote. Figures that need telemetry CSVs are
skipped with a message if the benchmark was run with --no-telemetry,
rather than being drawn from invented data.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from _common import ROOT, Log, common_args, load, results_dir


def _read(p: Path):
    return pd.read_csv(p) if p.exists() else None


def main():
    p = common_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--telemetry-from", default="intact-ra-agentic",
                   help="which method's CSVs to draw the RAN figures from")
    p.add_argument("--seed", type=int, default=None)
    a = p.parse_args()

    from intact_agentic.config import scenario_name
    from intact_agentic.report import figures as F

    cfg = load(a)
    root = results_dir(a, cfg)
    figs = root / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    log = Log(root / "figures.log", a.quiet)
    F.apply_style(int((cfg.get("report", {}) or {}).get("dpi", 160)))
    made = []

    sp = root / "summaries.json"
    if not sp.exists():
        log(f"no summaries.json in {root}; run run_benchmark.py first")
        return
    summaries = json.loads(sp.read_text())
    cis = json.loads((root / "bootstrap_ci.json").read_text()) \
        if (root / "bootstrap_ci.json").exists() else None

    made.append(F.fig_headline(summaries, figs / "fig1_headline.png", cis,
                               f"{scenario_name(cfg)}"))
    iids = sorted({k[len("fulfilment_"):] for s in summaries.values()
                   for k in s if k.startswith("fulfilment_")})
    if iids:
        made.append(F.fig_per_intent(summaries, figs / "fig2_per_intent.png",
                                     iids))
    made.append(F.fig_safety_vs_value(summaries,
                                      figs / "fig3_safety_vs_value.png"))
    if any(m.startswith("agentic-") for m in summaries):
        made.append(F.fig_ablation(summaries, figs / "fig4_ablation.png"))

    # --- time series need the per-epoch CSVs ------------------------------
    seed = a.seed if a.seed is not None else int(cfg["run"]["seed"])
    recs = {}
    for m in summaries:
        f = root / m / f"seed{seed}" / "epochs.csv"
        if f.exists():
            recs[m] = _epochs_to_records(pd.read_csv(f))
    if recs:
        ch = _change_epoch(cfg)
        made.append(F.fig_drift_response(recs, figs / "fig5_drift_response.png",
                                         change_epoch=ch))
        made.append(F.fig_latency(recs, figs / "fig6_latency.png"))

    sd = root / "slope_drift.json"
    if sd.exists():
        blob = json.loads(sd.read_text())
        se = {tuple(k.split("->")): v for k, v in blob["early"].items()}
        sl = {tuple(k.split("->")): v for k, v in blob["late"].items()}
        made.append(F.fig_slope_drift(se, sl, figs / "fig7_slope_drift.png"))

    # --- RAN figures -------------------------------------------------------
    tel = root / a.telemetry_from / f"seed{seed}" / "csv"
    if tel.exists():
        perf = _read(tel / "epoch_kpm.csv")
        cell = _read(tel / "cell.csv")
        ue = _read(tel / "ue_state.csv")
        chan = _read(tel / "channel.csv")
        tenants = sorted(cfg["ran"]["slices"])
        if perf is not None:
            perf = perf[perf["tenant"] != "_cell"]
            made.append(F.fig_causal_chain(cell, perf,
                                           figs / "fig8_ran_causal_chain.png",
                                           tenants))
        if cell is not None and "epoch" not in cell:
            cell = cell.assign(epoch=np.arange(len(cell)))
        if perf is not None:
            cp = perf[perf["tenant"] == tenants[0]]
            made.append(F.fig_cell_overview(
                _read(tel / "epoch_kpm.csv").query("tenant == '_cell'"),
                figs / "fig9_cell_overview.png"))
        src = ue if ue is not None and "sinr_db" in (ue.columns if ue is not None else []) else chan
        if src is not None:
            made.append(F.fig_ran_distributions(
                src, figs / "fig10_ran_distributions.png", tenants))
        ag = _read(tel / "agent.csv")
        sens = _read(tel / "sensitivity_log.csv")
        if ag is not None:
            made.append(F.fig_agent_internals(
                ag, sens, figs / "fig11_agent_internals.png"))
    else:
        log(f"no telemetry at {tel}; RAN figures skipped "
            f"(re-run the benchmark without --no-telemetry)")

    for m in made:
        log(f"  wrote {m}")
    log(f"{len(made)} figures in {figs}")
    log.close()


def _change_epoch(cfg):
    d = ((cfg.get("ran", {}) or {}).get("mobility", {}) or {}).get("drift", {})
    if d and d.get("tenants"):
        spe = (cfg["ran"].get("pre_slots", 8) + cfg["ran"].get("post_slots", 8))
        return int(d.get("start_slot", 0)) / max(spe, 1)
    for t, s in (cfg["ran"].get("slices") or {}).items():
        if int(s.get("arrives_at_epoch", 0)) > 0:
            return int(s["arrives_at_epoch"])
    return None


class _R:
    __slots__ = ("epoch", "g_after", "g_before", "predicted", "pi_class",
                 "n_writes", "latency_ms", "candidates")


def _epochs_to_records(df):
    gcols = [c for c in df.columns if c.startswith("g_")]
    pcols = [c for c in df.columns if c.startswith("gpre_")]
    out = []
    for _, row in df.iterrows():
        r = _R()
        r.epoch = int(row["epoch"])
        r.g_after = {c[2:]: float(row[c]) for c in gcols
                     if np.isfinite(row[c])}
        r.g_before = {c[5:]: float(row[c]) for c in pcols
                      if np.isfinite(row[c])}
        r.predicted = {}
        r.pi_class = {k: 1.0 for k in r.g_after}
        r.n_writes = int(row.get("n_writes", 0))
        r.latency_ms = float(row.get("latency_ms", 0.0))
        r.candidates = float(row.get("candidates", 0.0))
        out.append(r)
    return out


if __name__ == "__main__":
    main()
