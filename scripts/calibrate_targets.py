#!/usr/bin/env python3
"""Set intent targets AND xApp setpoints from the passive plant. [AUTO]

Two separate failures in this project came from setpoints calibrated for a
different plant than the one being run:

  * INTENT targets set by eye produce intents that are always met or never
    met, and either way they carry no information about the controller.
  * xAPP setpoints calibrated for a lighter plant leave every xApp
    PERMANENTLY unsatisfied, so each one asks for more of its knob every
    epoch.  Under perfect slopes that drove every control into a domain
    boundary within 200 epochs and left the oracle 0.25 below all-reject.

This script measures the passive plant (no controller) and places every
target and setpoint at the MEDIAN, so each is satisfied about half the
time and pushes in both directions.  It rewrites the block between the
markers `# >>> CALIBRATED` and `# <<< CALIBRATED` in the scenario file and
leaves everything else untouched.

A delay KPI whose interquartile range is degenerate (pinned at the
processing floor or at the reporting clip) is replaced automatically by a
KPI that actually varies, and the substitution is written into the file.
"""
from __future__ import annotations
import argparse, re
from pathlib import Path
import numpy as np
from _common import ROOT, Log, common_args, load

MARK_A, MARK_B = "# >>> CALIBRATED", "# <<< CALIBRATED"


def main():
    p = common_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--seeds", default="11,12")
    p.add_argument("--compliance", type=float, default=0.50,
                   help="fraction of the window in which the passive plant "
                        "meets each SLA. 0.50 puts every tenant exactly on "
                        "its margin, so lending capacity to one tenant breaks "
                        "another and reallocation nets ~0 by construction. "
                        "0.75 models SLAs provisioned with headroom.")
    p.add_argument("--from-epoch", type=int, default=0,
                   help="measure the passive plant from this epoch ...")
    p.add_argument("--to-epoch", type=int, default=None,
                   help="... up to this one. A window BEFORE a scheduled plant "
                        "change sets targets as SLAs fixed at provisioning time, "
                        "rather than recalibrated to a network that has since "
                        "degraded -- which is what makes all-reject fail after it.")
    a = p.parse_args()

    from intact_agentic.config import epoch_slots
    from intact_agentic.ran.simulator import RealisticRAN
    log = Log(None, a.quiet)
    cfg = load(a)
    pre, post = epoch_slots(cfg)
    tenants = sorted(cfg["ran"]["slices"])
    K = ("throughput_mbps", "delay_ms", "delay_p50_ms", "delay_p90_ms",
         "buffer_kb")
    acc = {t: {k: [] for k in K} for t in tenants}
    cell = {k: [] for k in ("prb_util_pct", "edge_fraction", "mean_sinr_db")}
    for sd in [int(s) for s in a.seeds.split(",")]:
        ran = RealisticRAN(cfg)
        ran.reset(sd)
        stop = a.to_epoch if a.to_epoch is not None else a.epochs
        for ep in range(stop):
            ran.epoch = ep
            k = ran.step(pre + post, record=False)
            if ep < a.from_epoch:
                continue
            for t in tenants:
                for kk in K:
                    acc[t][kk].append(k[t][kk])
            for kk in cell:
                cell[kk].append(k["_cell"][kk])
    q = lambda v, pc: float(np.percentile(v, pc))
    lo_pc = 100.0 * (1.0 - a.compliance)     # higher-is-better KPIs
    hi_pc = 100.0 * a.compliance             # lower-is-better KPIs
    floor = float(cfg["ran"].get("base_delay_ms", 2.0))
    med = {}
    for t_, d in acc.items():
        med[t_] = {}
        for kk, v in d.items():
            if kk.startswith("delay") or kk == "buffer_kb":
                val = q(v, hi_pc)
                if kk.startswith("delay"):
                    # An SLA that forbids ANY queueing is not a latency SLA:
                    # calibrated from a window pinned at the processing
                    # floor it gives a target no controller can ever meet
                    # once load rises.  Floor it at twice the fixed delay.
                    val = max(val, 2.0 * floor)
                med[t_][kk] = val
            else:
                med[t_][kk] = q(v, lo_pc)
    cm = {kk: q(v, 50) for kk, v in cell.items()}
    log(f"passive plant: PRB util p10/p50/p90 = "
        f"{q(cell['prb_util_pct'],10):.1f}/{cm['prb_util_pct']:.1f}/"
        f"{q(cell['prb_util_pct'],90):.1f}%")

    def delay_kpi(t):
        # the first delay statistic whose window distribution genuinely
        # varies: IQR above 0.5 ms, lower quartile off the processing floor,
        # upper quartile clear of the reporting clip
        clip = float(cfg["ran"].get("max_delay_ms", 1500.0))
        for kk in ("delay_p90_ms", "delay_ms", "delay_p50_ms"):
            v = acc[t][kk]
            if (q(v, 75) - q(v, 25) > 0.5 and q(v, 25) > 1.05 * floor
                    and q(v, 75) < 0.95 * clip):
                return kk
        return "delay_p90_ms"

    path = Path(a.scenario) if a.scenario.endswith(".yaml") else \
        ROOT / "configs" / "scenarios" / f"{a.scenario}.yaml"
    body_txt = re.sub(re.escape(MARK_A) + r".*?" + re.escape(MARK_B), "",
                      path.read_text(), flags=re.S)
    has_rs = re.search(r"^_replace_sections:", body_txt, flags=re.M)
    lines = [MARK_A + " -- written by scripts/calibrate_targets.py; do not "
             "edit by hand"]
    if not has_rs:
        lines.append("_replace_sections: [intents, xapps]")
    lines.append("intents:")
    iid = 0
    for t in tenants:
        iid += 1
        lines.append(f"  - {{iid: i{iid}, tenant: {t}, kpi: throughput_mbps, "
                     f"target: {med[t]['throughput_mbps']:.3f}, "
                     f"direction: higher_better, pi_class: 1.0, weight: 1.0, "
                     f"epsilon: 0.02}}")
        dk = delay_kpi(t)
        iid += 1
        note = "" if dk == "delay_p50_ms" else \
            f"   # {dk}: delay_p50 IQR degenerate here"
        lines.append(f"  - {{iid: i{iid}, tenant: {t}, kpi: {dk}, "
                     f"target: {med[t][dk]:.2f}, direction: lower_better, "
                     f"pi_class: 1.0, weight: 1.0, epsilon: 0.02}}{note}")
        log(f"  {t}: throughput {med[t]['throughput_mbps']:.3f} Mb/s, "
            f"{dk} {med[t][dk]:.2f} ms")
    # The host power cap is an intent only when some claim can actually move
    # transmit power.  When power is held at its provisioned value it is a
    # fixed operator constraint, and scoring it as an intent would add the
    # same constant 1.0 to every method and inflate every fulfilment figure.
    if any(c["param"] == "txpower" for c in cfg["claims"]):
        iid += 1
        tx = float(cfg["ran"]["initial_controls"].get("txpower", 40.0))
        lines.append(f"  - {{iid: i{iid}, tenant: HOST, kpi: txpower_dbm, "
                     f"target: {tx:.1f}, direction: lower_better, "
                     f"pi_class: 1.0, weight: 1.0, epsilon: 0.0}}")
    # xApps keep their IDENTITY and KIND from the scenario -- claims refer
    # to them by name -- and only their setpoint is replaced.
    tilt = float(cfg["ran"]["initial_controls"].get("tilt", 6.0))
    lines.append("xapps:")
    for x in cfg.get("xapps_template") or cfg["xapps"]:
        x = dict(x)
        prm, kind = x["param"], x["kind"]
        if x.get("fixed_setpoint"):
            # a setpoint that IS the experimental design (e.g. the two ends
            # of a deliberate tilt conflict) is not calibrated away
            body = ", ".join(f"{k}: {v}" for k, v in x.items())
            lines.append(f"  - {{{body}}}")
            continue
        # The tenant whose KPI this xApp closes a loop on is its DECLARED
        # tenant.  Inferring it from the parameter name fails for cell-scope
        # knobs: a tilt controller that serves T1 has no "_T1" in "tilt",
        # and was left at a template setpoint far from T1's operating point
        # -- permanently satisfied, so it pushed one way every epoch.
        ten = x.get("tenant") if x.get("tenant") in med else \
            (prm.split("_", 1)[1] if "_" in prm else None)
        if kind == "throughput" and ten in med:
            x["target"] = round(med[ten]["throughput_mbps"], 3)
            x["deadband"] = 0.10
        elif kind == "latency" and ten in med:
            x["target"] = round(med[ten]["delay_ms"], 2)
            x["deadband"] = max(round(0.1 * med[ten]["delay_ms"], 2), 0.5)
            # gain is PRB per ms of error; rescale with the delay magnitude
            x["gain"] = round(min(2.0, 4.0 / max(med[ten]["delay_ms"], 1.0)), 3)
        elif kind == "robustness" and ten in med:
            x["target"] = round(med[ten]["buffer_kb"], 1)
        elif kind == "steering":
            x["target"] = round(cm["prb_util_pct"], 1)
            x["deadband"] = 4.0
        elif kind == "coverage":
            x["edge_trigger"] = round(cm["edge_fraction"], 3)
            x["sinr_target"] = round(cm["mean_sinr_db"], 2)
        elif kind == "energy" and prm == "tilt":
            x["target"], x["target_alt"] = round(tilt + 1, 1), round(tilt - 1, 1)
        body = ", ".join(f"{k}: {v}" for k, v in x.items())
        lines.append(f"  - {{{body}}}")
    lines.append(MARK_B)
    block = "\n".join(lines)

    txt = path.read_text()
    if MARK_A in txt:
        txt = re.sub(re.escape(MARK_A) + r".*?" + re.escape(MARK_B), block,
                     txt, flags=re.S)
    else:
        txt = txt.rstrip() + "\n\n" + block + "\n"
    path.write_text(txt)
    log(f"\nwrote calibrated intents and xApp setpoints into {path}")


def _dom(cfg, param):
    for c in cfg["claims"]:
        if c["param"] == param:
            return list(c["domain"])
    return [6.0, 60.0]


if __name__ == "__main__":
    main()
