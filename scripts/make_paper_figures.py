#!/usr/bin/env python3
"""Publication figures from the held-out runs and the mechanism trace.  [AUTO]

  fig_heldout_if.png        fulfilment per method; every seed a point,
                            bootstrap 95% interval as a bar
  fig_heldout_safety.png    C1/C2 violations and causal safety crossings
  fig_heldout_cost.png      writes per epoch, near-RT decision latency, and
                            digital-twin compute (slow loop) -- separately
  fig_mechanism_slopes.png  true vs Agentic-estimated (+/- 1 sd) vs frozen
                            slopes through the migration
  fig_mechanism_decisions.png  reservations committed by INTACT-RA and
                            INTACT-RA-Agentic, the migrating tenant's
                            distance, and rolling fulfilment

Every figure is drawn from recorded data only.  Missing inputs are skipped
with a message rather than drawn from anything invented.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["oracle", "intact-ra-agentic", "b3", "intact-ra", "all-reject",
         "all-accept"]
LABEL = {"oracle": "Oracle\n(perfect copy)", "intact-ra-agentic":
         "INTACT-RA-\nAgentic", "b3": "B3", "intact-ra": "INTACT-RA",
         "all-reject": "All-reject", "all-accept": "B0\nall-admit"}
COL = {"oracle": "#555555", "intact-ra-agentic": "#1b7837",
       "b3": "#e08214", "intact-ra": "#2166ac", "all-reject": "#999999",
       "all-accept": "#b2182b"}


def style():
    plt.rcParams.update({"figure.dpi": 150, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25})


def boot(v, n=4000, seed=3):
    v = np.asarray(v, float)
    if len(v) < 2:
        return (float(v.mean()),) * 2 if len(v) else (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    b = [rng.choice(v, len(v)).mean() for _ in range(n)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def heldout(runs_path: Path, out: Path):
    runs = {}
    for line in runs_path.read_text().splitlines():
        r = json.loads(line)
        runs.setdefault(r["m"], {})[r["s"]] = r
    ms = [m for m in ORDER if runs.get(m)]
    # Compare methods ONLY on seeds every plotted method has completed.
    # Averaging each method over its own seed set compares different
    # plants: mid-run, a method that has finished an easy seed looks better
    # than one that has not reached it yet.
    common = set.intersection(*(set(runs[m]) for m in ms)) if ms else set()
    if not common:
        print("no seed completed by every method yet; figures skipped")
        return []
    dropped = {m: sorted(set(runs[m]) - common) for m in ms}
    if any(dropped.values()):
        print(f"using the {len(common)} seed(s) every method has completed "
              f"{sorted(common)}; excluded for now: "
              f"{ {m: d for m, d in dropped.items() if d} }")
    runs = {m: {s: runs[m][s] for s in sorted(common)} for m in ms}
    x = np.arange(len(ms))

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for k, m in enumerate(ms):
        v = [r["IF"] for r in runs[m].values()]
        lo, hi = boot(v)
        ax.bar(k, np.mean(v), color=COL[m], alpha=0.75, width=0.6)
        ax.errorbar(k, np.mean(v), yerr=[[np.mean(v) - lo], [hi - np.mean(v)]],
                    color="black", capsize=4, lw=1)
        ax.scatter(np.full(len(v), k) + np.linspace(-0.12, 0.12, len(v)), v,
                   color="black", s=12, zorder=3)
        ax.text(k, hi + 0.01, f"{np.mean(v):.3f}", ha="center", fontsize=8)
    ax.set_xticks(x, [LABEL[m] for m in ms])
    ax.set_ylabel("intent fulfilment (held-out)")
    ax.set_title(f"Held-out intent fulfilment on {len(common)} unseen seed(s) "
                 f"completed by every method\n(points = seeds, "
                 f"bars = bootstrap 95% CI)", fontsize=9)
    ax.set_ylim(0.45, 1.0)
    fig.tight_layout()
    fig.savefig(out / "fig_heldout_if.png")
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(8.0, 3.2))
    for ax, key, title in ((axs[0], "c1c2", "C1 + C2 violations per run"),
                           (axs[1], "cx", "causal safety crossings per run")):
        vals = [np.mean([r["c1"] + r["c2"] if key == "c1c2" else r[key]
                         for r in runs[m].values()]) for m in ms]
        ax.bar(x, vals, color=[COL[m] for m in ms], alpha=0.8)
        for k, v in enumerate(vals):
            ax.text(k, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x, [LABEL[m] for m in ms], fontsize=7)
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out / "fig_heldout_safety.png")
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(10.0, 3.2))
    for ax, key, title in ((axs[0], "wpe", "writes per epoch"),
                           (axs[1], "lat95", "p95 decision latency (ms)")):
        vals = [np.mean([r[key] for r in runs[m].values()]) for m in ms]
        ax.bar(x, vals, color=[COL[m] for m in ms], alpha=0.8)
        ax.set_xticks(x, [LABEL[m] for m in ms], fontsize=7)
        ax.set_title(title)
    axs[1].axhline(10.0, color="red", ls="--", lw=1)
    axs[1].text(0, 10.2, "near-RT budget 10 ms", color="red", fontsize=7)
    tw = [r.get("twin_s_per_call") for r in runs.get("intact-ra-agentic",
                                                     {}).values()
          if r.get("twin_s_per_call")]
    if tw:
        axs[2].bar([0], [np.mean(tw) * 1000], color=COL["intact-ra-agentic"])
        axs[2].set_xticks([0], ["digital twin\n(slow loop)"])
        axs[2].set_title("twin compute per call (ms)")
    else:
        axs[2].set_axis_off()
    fig.tight_layout()
    fig.savefig(out / "fig_heldout_cost.png")
    plt.close(fig)
    return ["fig_heldout_if.png", "fig_heldout_safety.png",
            "fig_heldout_cost.png"]


def mechanism(mech_path: Path, out: Path):
    m = json.loads(mech_path.read_text())
    made = []
    sl = m.get("slopes")
    if sl and sl["epoch"]:
        keys = list(sl["true"])
        fig, axs = plt.subplots(len(keys), 1, figsize=(7.2, 2.1 * len(keys)),
                                sharex=True)
        axs = np.atleast_1d(axs)
        ep = np.array(sl["epoch"])
        for ax, k in zip(axs, keys):
            est, sd = np.array(sl["est"][k]), np.array(sl["sd"][k])
            ax.plot(ep, sl["true"][k], color="black", lw=1.4, label="true (measured on plant)")
            ax.plot(ep, est, color=COL["intact-ra-agentic"], lw=1.4,
                    label="INTACT-RA-Agentic estimate")
            ax.fill_between(ep, est - sd, est + sd,
                            color=COL["intact-ra-agentic"], alpha=0.18)
            ax.plot(ep, sl["frozen"][k], color=COL["intact-ra"], lw=1.2,
                    ls="--", label="frozen table (INTACT-RA)")
            knob, iid = k.split("|")
            ax.set_ylabel(f"d g_{iid} / d {knob}", fontsize=8)
        axs[0].legend(fontsize=7, loc="upper right")
        axs[-1].set_xlabel("epoch (scored from burn-in end)")
        axs[0].set_title("Sensitivity trajectories through the migration")
        fig.tight_layout()
        fig.savefig(out / "fig_mechanism_slopes.png")
        plt.close(fig)
        made.append("fig_mechanism_slopes.png")

    if "intact-ra" in m and "intact-ra-agentic" in m:
        fig, axs = plt.subplots(3, 1, figsize=(7.2, 6.6), sharex=True)
        for name, ls in (("intact-ra", "--"), ("intact-ra-agentic", "-")):
            s = m[name]["series"]
            ep = np.array(s["epoch"])
            for t, c in (("T1", "#762a83"), ("T2", "#e08214"), ("T3", "#1b7837")):
                axs[0].plot(ep, s["res"][t], ls=ls, color=c, lw=1.1,
                            label=f"{t} ({'Agentic' if name != 'intact-ra' else 'INTACT-RA'})")
            # TRAILING mean: a centred window runs off the end of the data
            # and fabricates a final dip that is not in the run
            w = 20
            v = np.asarray(s["IF"], float)
            c = np.cumsum(np.insert(v, 0, 0.0))
            f = np.array([(c[k + 1] - c[max(0, k + 1 - w)]) / min(k + 1, w)
                          for k in range(len(v))])
            axs[2].plot(ep, f, ls=ls, color=COL[name], lw=1.3,
                        label=name.replace("intact-ra-agentic", "INTACT-RA-Agentic")
                        .replace("intact-ra", "INTACT-RA"))
        s = m["intact-ra-agentic"]["series"]
        axs[1].plot(s["epoch"], s["radius_T1"], color="#762a83")
        axs[0].set_ylabel("reserved PRBs")
        axs[0].legend(fontsize=6, ncol=3)
        axs[1].set_ylabel("T1 mean distance (m)")
        axs[2].set_ylabel("fulfilment (trailing 20-epoch mean)")
        axs[2].legend(fontsize=7)
        for ax in axs:
            ax.axvline(m.get("burn_in", 0), color="red", lw=0.8, ls=":")
        axs[-1].set_xlabel("epoch  (red line: end of burn-in, start of migration)")
        axs[0].set_title("Decisions: reservations committed by each method")
        fig.tight_layout()
        fig.savefig(out / "fig_mechanism_decisions.png")
        plt.close(fig)
        made.append("fig_mechanism_decisions.png")
    return made


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default=None)
    ap.add_argument("--mechanism", default=None)
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    style()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    made = []
    if a.runs and Path(a.runs).exists():
        made += heldout(Path(a.runs), out)
    else:
        print("no held-out runs file; held-out figures skipped")
    if a.mechanism and Path(a.mechanism).exists():
        made += mechanism(Path(a.mechanism), out)
    else:
        print("no mechanism trace; mechanism figures skipped")
    for f in made:
        print("wrote", out / f)


if __name__ == "__main__":
    main()
