#!/usr/bin/env python3
"""STEP 1. Offline calibration.

Runs, in order, and writes everything it finds to disk:

  1. the passive-plant KPI survey, which is where the intent targets come
     from -- a target set by eye produces intents that are always met or
     never met, and both carry no information about the controller;
  2. the resource-pressure band edges, matched to the load scales the
     sweep uses for its three table columns;
  3. the cold-start pressure-proxy affine fit;
  4. the OFFLINE SENSITIVITY SWEEP that produces INTACT-RA's frozen table.

Everything here is automated. The only manual step is deciding whether to
copy the suggested targets and edges into configs/base.yaml, and the
script prints them in copy-paste form with a diff against what is
currently configured.
"""
from __future__ import annotations
import argparse
import numpy as np
from _common import ROOT, Log, common_args, load, write_json, write_text


def main():
    p = common_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--survey-epochs", type=int, default=300)
    p.add_argument("--survey-seeds", type=int, default=3)
    p.add_argument("--force", action="store_true",
                   help="rebuild the sensitivity table even if cached")
    p.add_argument("--skip-sweep", action="store_true")
    a = p.parse_args()

    from intact_agentic.config import build_registry, epoch_slots, scenario_name
    from intact_agentic.ran.simulator import RealisticRAN
    from intact_agentic.experiment import build_prior

    cfg = load(a)
    out = ROOT / "artifacts" / scenario_name(cfg)
    out.mkdir(parents=True, exist_ok=True)
    log = Log(out / "calibrate.log", a.quiet)
    reg = build_registry(cfg)
    pre, post = epoch_slots(cfg)
    tenants = sorted(cfg["ran"]["slices"])
    report = {}

    # -- 1. passive KPI survey ------------------------------------------
    log("STEP 1/4  passive-plant KPI survey "
        f"({a.survey_seeds} seeds x {a.survey_epochs} epochs)")
    KEYS = ("throughput_mbps", "delay_p50_ms", "delay_ms", "delivery_pct",
            "bler", "sinr_db", "cqi", "mcs", "spectral_efficiency",
            "prb_alloc", "buffer_kb", "offered_slice_mbps")
    acc = {t: {k: [] for k in KEYS} for t in tenants}
    cell, rho_true, rho_proxy = [], [], []
    for s in range(a.survey_seeds):
        ran = RealisticRAN(cfg)
        ran.reset(int(cfg["run"]["seed"]) + 13 * s)
        for ep in range(a.survey_epochs):
            ran.epoch = ep
            k = ran.step(pre + post, record=False)
            for t in tenants:
                if t not in k:
                    continue
                for kk in KEYS:
                    if kk in k[t]:
                        acc[t][kk].append(k[t][kk])
                rho_true.append(ran.resource_pressure(t, k))
                row = k[t]
                cap = (max(row["prb_alloc"], 1e-6) * ran.prb_hz
                       * max(row["spectral_efficiency"], 1e-3) / 1e6)
                rho_proxy.append(min(row["offered_slice_mbps"]
                                     / max(cap, 1e-6), 3.0))
            cell.append(k["_cell"]["prb_util_pct"])
    log(f"          mean PRB utilisation {np.mean(cell):5.1f}% "
        f"(p10 {np.percentile(cell,10):.1f}, p90 {np.percentile(cell,90):.1f})")
    survey = {t: {kk: {"p10": float(np.percentile(v, 10)),
                       "p35": float(np.percentile(v, 35)),
                       "p50": float(np.percentile(v, 50)),
                       "p65": float(np.percentile(v, 65)),
                       "p90": float(np.percentile(v, 90)),
                       "mean": float(np.mean(v))}
                  for kk, v in d.items() if v}
              for t, d in acc.items()}
    report["survey"] = survey
    report["cell_prb_util_pct"] = float(np.mean(cell))

    log("")
    log("  SUGGESTED INTENT TARGETS (measured median of the passive plant)")
    log("  An intent targeted at the median is contestable: a do-nothing")
    log("  controller sits near 50% fulfilment and can be moved either way.")
    sugg = []
    for it in cfg["intents"]:
        t, kpi = it["tenant"], it["kpi"]
        if t in survey and kpi in survey[t]:
            med = survey[t][kpi]["p50"]
            cur = float(it["target"])
            flag = "  <-- CHANGE" if abs(med - cur) > 0.12 * max(abs(cur), 1e-9) else ""
            log(f"    {it['iid']:>4}  {t:>4} {kpi:<16} "
                f"configured {cur:9.2f}   measured median {med:9.2f}{flag}")
            sugg.append({"iid": it["iid"], "configured": cur,
                         "suggested": round(med, 2)})
    report["suggested_targets"] = sugg

    # -- 2. pressure band edges ------------------------------------------
    log("")
    log("STEP 2/4  resource-pressure band edges")
    scales = (cfg.get("calibration", {}) or {}).get(
        "sweep_load_scales", [0.70, 1.00, 1.35])
    cuts = {}
    for sc in scales:
        c2 = {**cfg, "ran": {**cfg["ran"],
                             "slices": {t: {**v} for t, v in
                                        cfg["ran"]["slices"].items()}}}
        for t in c2["ran"]["slices"]:
            c2["ran"]["slices"][t]["load_mbps_per_ue"] *= float(sc)
            c2["ran"]["slices"][t]["load_profile"] = {"levels": [1.0],
                                                      "hold_slots": 10 ** 9}
        r = RealisticRAN(c2)
        r.step(64, record=False)
        v = []
        for _ in range(40):
            k = r.step(pre + post, record=False)
            v.append(np.mean([r.resource_pressure(t, k)
                              for t in c2["ran"]["slices"]]))
        cuts[float(sc)] = float(np.mean(v))
        log(f"          load scale {sc:.2f} -> resource pressure "
            f"{cuts[float(sc)]:.3f}")
    ks = sorted(cuts)
    edges = [round((cuts[ks[i]] + cuts[ks[i + 1]]) / 2, 3)
             for i in range(len(ks) - 1)]
    log(f"          => arbiter.regime_pressure_edges: {edges}   "
        f"(configured: {list(cfg['arbiter'].get('regime_pressure_edges', []))})")
    report["regime_pressure_edges"] = edges

    # -- 3. cold-start proxy fit -----------------------------------------
    log("")
    log("STEP 3/4  cold-start pressure-proxy affine fit")
    rt, rp = np.array(rho_true), np.array(rho_proxy)
    A = np.column_stack([rp, np.ones_like(rp)])
    coef, *_ = np.linalg.lstsq(A, rt, rcond=None)
    corr = float(np.corrcoef(rt, rp)[0, 1])
    log(f"          rho_true ~ {coef[0]:.3f} * proxy + {coef[1]:.3f}   "
        f"(corr {corr:.3f}, n={rt.size})")
    log(f"          => agent.context.rho_proxy_scale:  {coef[0]:.3f}")
    log(f"             agent.context.rho_proxy_offset: {coef[1]:.3f}")
    report["rho_proxy"] = {"scale": float(coef[0]), "offset": float(coef[1]),
                           "corr": corr}

    # -- 4. the offline sweep ---------------------------------------------
    if not a.skip_sweep:
        log("")
        log("STEP 4/4  offline sensitivity sweep (this is the slow one)")
        cache = ROOT / (cfg.get("calibration", {}) or {}).get(
            "cache", "artifacts/sensitivity_prior.json")
        prior = build_prior(cfg, reg, log=log, cache=cache, force=a.force)
        ses = np.array([e.sigma for e in prior.tab.values()])
        sl = np.array([abs(e.slope) for e in prior.tab.values()])
        nz = sl > 1e-6
        log(f"          {len(prior.tab)} entries; |slope| median "
            f"{np.median(sl):.5f}; sigma median {np.median(ses):.5f}")
        if nz.any():
            ratio = float(np.median(ses[nz] / sl[nz]))
            log(f"          median sigma/|slope| = {ratio:.2f}")
            if ratio > 3.0:
                log("          WARNING: the typical slope is not "
                    "distinguishable from noise. Raise calibration."
                    "replicates or calibration.measure_slots, or the robust "
                    "arbiter will refuse almost every write.")
        report["prior_entries"] = len(prior.tab)
        report["prior_sigma_median"] = float(np.median(ses))

    write_json(out / "calibration.json", report)
    log("")
    log(f"Done. Full numbers in {out / 'calibration.json'}")
    log("MANUAL STEP: if any line above is flagged <-- CHANGE, copy the")
    log("suggested value into configs/base.yaml and re-run this script.")
    log.close()


if __name__ == "__main__":
    main()
