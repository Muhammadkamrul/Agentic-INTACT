#!/usr/bin/env python3
"""ACCOUNTING: what does each method's allocation cost, and what overhead
does it create?  Measured, never optimised.  [AUTO]

No method -- proposed or baseline -- was designed to reduce a bill, and none
is changed here.  This script only MEASURES what each method already does,
by replaying recorded runs deterministically and observing the live plant.

How it stays honest
-------------------
* Runs are deterministic (verified bit-identical elsewhere), so each
  recorded (method, seed) is re-run and its fulfilment must MATCH the
  recorded value exactly.  A run that does not reproduce is reported as
  such and excluded -- never silently used.
* Observation is attached to the plant CLASS and filtered to the live plant
  object, so the digital twin's clones and the counterfactual copy are
  never touched.  No method code, configuration or random stream changes.
* A lower bill is NOT automatically better.  A method that strands capacity
  holds LESS reservation and so looks cheaper while failing more intents.
  Every cost is therefore reported next to fulfilment, including cost per
  fulfilled intent-epoch.

Billing proxies, per tenant (scored epochs only; one epoch = 160 ms)
  reserved PRB-seconds   capacity leased (hard slicing: you pay for what you
                         reserve, used or not)
  used PRB-seconds       capacity actually scheduled (pay-per-use)
  delivered gigabits     traffic carried (pay-per-traffic)
  idle leased PRB-s      reserved minus used: paid for, carried nothing
  unsold pool PRB-s      pool capacity no tenant reserved (host's idle stock)
  PRB-s per fulfilled intent-epoch   cost-effectiveness

Overhead proxies, per method
  E2 control messages    one RIC Control request per executed write
                         (signalling overhead; acks would double it)
  reconfiguration loss   PRB capacity lost to reconfiguration transients --
                         measured by the simulator, not a proxy
  candidates scored      arbitration work per epoch (computation)
  decision latency       measured near-RT decision time (computation)
  twin calls, seconds    slow-loop calibration work (INTACT-RA-Agentic only)

Usage
    python scripts/accounting.py --scenario S16_high_ceiling \\
        --runs-dirs results/S16_high_ceiling/heldout,results/S16_high_ceiling/heldout_cell
    python scripts/accounting.py ... --methods intact-ra,intact-ra-cell --seeds 31004
Time: frozen methods ~1 min per run; INTACT-RA-Agentic and the oracle
~6-9 min per run.  Resumable (replays are cached in accounting.jsonl).
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from _common import ROOT, load

EPOCH_S = 0.16
LABEL = {"all-reject": "All-reject", "all-accept": "B0 all-admit", "b3": "B3",
         "intact-ra": "INTACT-RA", "intact-ra-cell": "INTACT-RA (cell)",
         "intact-ra-pertenant-sweep": "INTACT-RA (per-tenant sweep)",
         "intact-ra-agentic": "INTACT-RA-Agentic", "oracle": "Oracle*",
         "agentic-lean": "Agentic (lean)"}


def recorded(dirs):
    rec = {}
    for d in dirs:
        p = Path(d) / "runs.jsonl"
        if p.exists():
            for line in p.read_text().splitlines():
                r = json.loads(line)
                rec[(r["m"], r["s"])] = r
    return rec


def replay(scen, method, seed, overrides):
    """Re-run one (method, seed), observing the live plant each epoch."""
    from intact_agentic import methods as M
    from intact_agentic.config import build_registry, scenario_name
    from intact_agentic.experiment import Experiment
    from intact_agentic.arbiter.sensitivity import StaticSensitivity
    from intact_agentic.ran.simulator import RealisticRAN

    class A:
        scenario, base, set = scen, str(ROOT / "configs" / "base.yaml"), overrides
    cfg = load(A)
    reg = build_registry(cfg)
    prior = StaticSensitivity.load(cfg, ROOT / "artifacts" /
                                   f"prior_{scenario_name(cfg)}.json")
    tenants = sorted(t for t in cfg["ran"]["slices"])
    n_prb = float(cfg["ran"]["n_prb"])
    burn = int(cfg["run"].get("burn_in_epochs", 0))
    from intact_agentic.config import epoch_slots
    spe = sum(epoch_slots(cfg))              # slots per epoch
    last = {}
    orig_step = RealisticRAN.step

    def spy(self, n, record=True, **kw):
        out = orig_step(self, n, record=record, **kw)
        # identity against the experiment's CURRENT plant: the burn-in
        # boundary swaps in a fresh plant mid-epoch, and the twin and the
        # counterfactual step their own copies, which must be ignored
        ex_ = last.get("ex")
        if ex_ is not None and self is ex_.ran:
            last["kpm"] = out
        return out

    RealisticRAN.step = spy
    try:
        with tempfile.TemporaryDirectory() as td:
            c = load(A)
            c["run"].update({"log_every": 0, "checkpoint_every": 0})
            ex = Experiment(c, M.get(method), reg, td, seed=seed, prior=prior,
                            telemetry=False, log=lambda *z: None)
            last["ex"] = ex
            rows = []
            for ep in range(int(c["run"]["epochs"])):
                ex.step_epoch()
                if ep < burn:
                    continue
                cmd = ex.ran.commanded_controls()
                k = last.get("kpm", {})
                r = ex.metrics.records[-1]
                row = {"writes": r.n_writes, "candidates": r.candidates,
                       "latency_ms": r.latency_ms,
                       "reconfig_prb": float(r.kpm_cell.get("reconfig_prb", 0.0)),
                       "fulfilled": {i: int(g >= 0) for i, g in r.g_after.items()}}
                for t in tenants:
                    row[f"res_{t}"] = float(ex.ran.quota_of(t, cmd))
                    kt = k.get(t, {})
                    row[f"used_{t}"] = float(kt.get("prb_alloc", 0.0))
                    row[f"mbps_{t}"] = float(kt.get("throughput_mbps", 0.0)) * \
                        float(kt.get("n_ue", 1.0))
                rows.append(row)
            s = ex.metrics.summary()
            tw = {"twin_calls": getattr(ex.twin, "calls", 0) if ex.twin else 0,
                  "twin_seconds": getattr(ex.twin, "seconds", 0.0) if ex.twin else 0.0}
    finally:
        RealisticRAN.step = orig_step
    intent_tenant = {i.iid: i.tenant for i in reg.intents.values()}
    return {"IF": s["IF"], "rows": rows, "tenants": tenants, "n_prb": n_prb,
            "slots_per_epoch": spe,
            "intent_tenant": intent_tenant, **tw}


def summarise(rep):
    rows, T, N = rep["rows"], rep["tenants"], rep["n_prb"]
    out = {"IF": rep["IF"], "epochs": len(rows)}
    fulfilled_by_t = {t: 0 for t in T}
    for r in rows:
        for i, ok in r["fulfilled"].items():
            t = rep["intent_tenant"].get(i)
            if t in fulfilled_by_t:
                fulfilled_by_t[t] += ok
    for t in T:
        res = sum(r[f"res_{t}"] for r in rows) * EPOCH_S
        used = sum(min(r[f"used_{t}"], r[f"res_{t}"] if r[f"res_{t}"] > 0
                       else r[f"used_{t}"]) for r in rows) * EPOCH_S
        out[f"reserved_prbs_{t}"] = res
        out[f"used_prbs_{t}"] = sum(r[f"used_{t}"] for r in rows) * EPOCH_S
        out[f"idle_leased_prbs_{t}"] = max(res - used, 0.0)
        out[f"delivered_gbit_{t}"] = sum(r[f"mbps_{t}"] for r in rows) * EPOCH_S / 1e3
        out[f"prbs_per_fulfilled_{t}"] = res / max(fulfilled_by_t[t], 1)
    tot_res = sum(out[f"reserved_prbs_{t}"] for t in T)
    out["reserved_prbs_total"] = tot_res
    out["unsold_pool_prbs"] = max(N * len(rows) * EPOCH_S - tot_res, 0.0)
    out["host_sold_fraction"] = tot_res / max(N * len(rows) * EPOCH_S, 1e-9)
    out["idle_leased_prbs_total"] = sum(out[f"idle_leased_prbs_{t}"] for t in T)
    n_ful = sum(sum(r["fulfilled"].values()) for r in rows)
    out["prbs_per_fulfilled_total"] = tot_res / max(n_ful, 1)
    out["e2_msgs_per_epoch"] = float(np.mean([r["writes"] for r in rows]))
    # The simulator records reconfiguration loss as PRB-SLOTS summed over the
    # measurement window, so it is expressed against the whole epoch's
    # capacity (n_prb x slots per epoch), not against n_prb alone -- which
    # would overstate it by the number of slots.
    lost = float(np.mean([r["reconfig_prb"] for r in rows]))
    out["reconfig_prb_slots_per_epoch"] = lost
    out["reconfig_loss_pct"] = 100.0 * lost / (N * rep["slots_per_epoch"])
    out["candidates_per_epoch"] = float(np.mean([r["candidates"] for r in rows]))
    out["latency_p95_ms"] = float(np.percentile([r["latency_ms"] for r in rows], 95))
    out["twin_calls"] = rep["twin_calls"]
    out["twin_seconds"] = rep["twin_seconds"]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="S16_high_ceiling")
    ap.add_argument("--runs-dirs", required=True,
                    help="comma list of dirs with runs.jsonl (to verify against)")
    ap.add_argument("--methods", default=None)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rec = recorded(a.runs_dirs.split(","))
    ms = a.methods.split(",") if a.methods else sorted({m for m, _ in rec})
    sds = [int(s) for s in a.seeds.split(",")] if a.seeds else \
        sorted({s for _, s in rec})
    out = Path(a.out) if a.out else ROOT / "results" / a.scenario / "accounting"
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "accounting.jsonl"
    done = {}
    if cache.exists():
        for line in cache.read_text().splitlines():
            d = json.loads(line)
            done[(d["m"], d["s"])] = d
    for m in ms:
        for sd in sds:
            if (m, sd) not in rec or (m, sd) in done:
                continue
            rep = replay(a.scenario, m, sd, a.set)
            same = abs(rep["IF"] - rec[(m, sd)]["IF"]) < 1e-12
            d = {"m": m, "s": sd, "verified": same,
                 "recorded_IF": rec[(m, sd)]["IF"], **summarise(rep)}
            done[(m, sd)] = d
            with open(cache, "a") as fh:
                fh.write(json.dumps(d) + "\n")
            print(f"{m:26s} seed {sd}: replay IF {rep['IF']:.10f} vs recorded "
                  f"{rec[(m, sd)]['IF']:.10f} -> "
                  f"{'VERIFIED' if same else 'DOES NOT REPRODUCE (excluded)'}",
                  flush=True)
    ok = [d for d in done.values() if d["verified"] and d["m"] in ms]
    bad = [d for d in done.values() if not d["verified"]]
    if not ok:
        print("no verified replays yet")
        return
    tenants = sorted({k[len("reserved_prbs_"):] for d in ok for k in d
                      if k.startswith("reserved_prbs_") and k != "reserved_prbs_total"})
    mean = lambda m, k: float(np.mean([d[k] for d in ok if d["m"] == m]))
    present = [m for m in LABEL if any(d["m"] == m for d in ok)]
    L = ["# Accounting: allocation cost and overhead (measured, not optimised)",
         "", f"Scenario `{a.scenario}`. Verified replays: {len(ok)}"
         + (f"; {len(bad)} did NOT reproduce and are excluded." if bad else "."),
         "No method was designed to minimise any of these quantities, and none "
         "was changed. **A lower bill is not automatically better**: stranding "
         "capacity lowers the lease while failing intents. Read cost next to "
         "fulfilment.", "",
         "## Billing proxies (per run, scored epochs; PRB-s = PRB-seconds)", "",
         "| method | IF | reserved PRB-s (tenant lease) | host sold fraction | used PRB-s | "
         "idle leased PRB-s | unsold pool PRB-s | delivered Gbit | PRB-s per fulfilled intent-epoch* |",
         "|---|---|---|---|---|---|---|---|---|"]
    for m in present:
        L.append(f"| {LABEL[m]} | {mean(m, 'IF'):.3f} | {mean(m, 'reserved_prbs_total'):.0f} | "
                 f"{mean(m, 'host_sold_fraction'):.1%} | "
                 f"{sum(mean(m, f'used_prbs_{t}') for t in tenants):.0f} | "
                 f"{mean(m, 'idle_leased_prbs_total'):.0f} | {mean(m, 'unsold_pool_prbs'):.0f} | "
                 f"{sum(mean(m, f'delivered_gbit_{t}') for t in tenants):.1f} | "
                 f"{mean(m, 'prbs_per_fulfilled_total'):.2f} |")
    L += ["", "\\* **Do not rank methods by this ratio alone.** It falls when a method "
          "leases LESS, so a method that strands capacity looks cheaper while "
          "fulfilling fewer intents. Compare methods on all of: fulfilment, "
          "tenant lease, and host sold fraction (Pareto): one method is better "
          "only if it is no worse on any of them.", "",
          "### Per-tenant lease (reserved PRB-s)", "",
          "| method | " + " | ".join(tenants) + " |", "|---|" + "---|" * len(tenants)]
    for m in present:
        L.append(f"| {LABEL[m]} | " + " | ".join(
            f"{mean(m, f'reserved_prbs_{t}'):.0f}" for t in tenants) + " |")
    L += ["", "## Overhead proxies (per epoch unless stated)", "",
          "| method | E2 control msgs | reconfiguration loss (% of epoch capacity) | "
          "candidates scored | decision p95 (ms) | twin calls / run | twin s / run |",
          "|---|---|---|---|---|---|---|"]
    for m in present:
        L.append(f"| {LABEL[m]} | {mean(m, 'e2_msgs_per_epoch'):.2f} | "
                 f"{mean(m, 'reconfig_loss_pct'):.2f} | {mean(m, 'candidates_per_epoch'):.1f} | "
                 f"{mean(m, 'latency_p95_ms'):.2f} | {mean(m, 'twin_calls'):.0f} | "
                 f"{mean(m, 'twin_seconds'):.0f} |")
    L += ["", "## How to read this", "",
          "- **Reserved PRB-s** is what a tenant would pay under hard slicing. A "
          "method with a low value may simply be stranding capacity; check "
          "**unsold pool PRB-s** and IF beside it.",
          "- **PRB-s per fulfilled intent-epoch** is the cost-effectiveness "
          "figure: capacity leased per unit of fulfilment delivered.",
          "- **E2 control messages** equal executed writes (one control request "
          "each); **reconfiguration loss** is capacity the simulator actually "
          "removed during reconfiguration transients.",
          "- Twin work runs in the slow loop and is reported separately from "
          "the near-RT decision latency."]
    (out / "ACCOUNTING.md").write_text("\n".join(L))
    print("\n" + "\n".join(L))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = np.arange(len(present))
        fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
        bottom = np.zeros(len(present))
        for t in tenants:
            v = np.array([mean(m, f"reserved_prbs_{t}") for m in present])
            axs[0].bar(x, v, bottom=bottom, label=f"{t} lease")
            bottom += v
        axs[0].bar(x, [mean(m, "unsold_pool_prbs") for m in present], bottom=bottom,
                   color="lightgrey", label="unsold pool")
        axs[0].set_xticks(x, [LABEL[m] for m in present], rotation=30, ha="right", fontsize=7)
        axs[0].set_ylabel("PRB-seconds per run")
        axs[0].set_title("Capacity lease by tenant, and unsold pool")
        axs[0].legend(fontsize=7)
        for m in present:
            axs[1].scatter(100 * mean(m, "host_sold_fraction"), mean(m, "IF"), s=40)
            axs[1].annotate(LABEL[m], (100 * mean(m, "host_sold_fraction"), mean(m, "IF")),
                            fontsize=7, xytext=(4, 2), textcoords="offset points")
        axs[1].set_xlabel("host: share of the PRB pool sold (%)")
        axs[1].set_ylabel("tenants: intent fulfilment")
        axs[1].set_title("Both sides at once (up and right is better for both)")
        fig.tight_layout()
        fig.savefig(out / "fig_billing.png", dpi=150)
        fig, axs = plt.subplots(1, 4, figsize=(13, 3.2))
        for ax, k, ttl in zip(axs, ["e2_msgs_per_epoch", "reconfig_loss_pct",
                                    "candidates_per_epoch", "latency_p95_ms"],
                              ["E2 control msgs / epoch", "reconfig. loss (% capacity)",
                               "candidates scored / epoch", "decision p95 (ms)"]):
            ax.bar(x, [mean(m, k) for m in present])
            ax.set_xticks(x, [LABEL[m] for m in present], rotation=40, ha="right", fontsize=6)
            ax.set_title(ttl, fontsize=9)
        fig.tight_layout()
        fig.savefig(out / "fig_overhead.png", dpi=150)
        print(f"\nwrote {out/'fig_billing.png'} and {out/'fig_overhead.png'}")
    except Exception as e:
        print(f"figures skipped: {e}")


if __name__ == "__main__":
    main()
