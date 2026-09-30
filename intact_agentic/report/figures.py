"""Publication figures.

Two families, kept separate because they answer different questions:

  BENCHMARK figures   compare controllers.  wIF, safety, writes, latency,
                      per-intent breakdowns, ablation bars, drift-response
                      time series.
  RAN figures         describe the plant.  The synchronised causal chain
                      -- mobility, path loss, SINR, CQI/MCS, capacity,
                      offered traffic, allocation, delivered performance,
                      queues, QoS regret -- on one shared time axis, plus
                      the distributional views that a time series hides.

Everything is drawn from the CSVs that the run actually wrote.  No figure
invents a quantity the simulator does not model; where a panel would need
one, it is omitted and the omission is listed in docs/METRICS.md.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

# ---------------------------------------------------------------------------
# A single consistent look, so figures from different scripts compose into
# one paper without re-styling.
PALETTE = {
    "all-reject": "#8c8c8c",
    "all-accept": "#c44e52",
    "static-priority": "#dd8452",
    "greedy-value": "#937860",
    "intact-ra": "#4c72b0",
    "intact-ra-cell": "#8fa8d4",
    "intact-ra-agentic": "#55a868",
    "agentic-noOnline": "#a0c9ad",
    "agentic-noLearnedRegime": "#b5cfbe",
    "agentic-noDRL": "#8fbf9f",
    "agentic-noProbe": "#c6ddcd",
    "oracle": "#000000",
}
TENANT_COLOURS = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3"]


def apply_style(dpi: int = 160) -> None:
    plt.rcParams.update({
        "figure.dpi": dpi, "savefig.dpi": dpi,
        "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
        "legend.fontsize": 7.5, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.constrained_layout.use": True,
        "legend.frameon": False,
    })


def colour(method: str) -> str:
    return PALETTE.get(method, "#666666")


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ===========================================================================
# BENCHMARK FIGURES
# ===========================================================================
def fig_headline(summaries: Dict[str, Dict], path: Path,
                 ci: Optional[Dict[str, Dict]] = None,
                 title: str = "") -> Path:
    """The four-panel headline: wIF, safety, writes, latency.

    All four panels matter together.  A controller that wins on wIF while
    causing more safety crossings has not won, and one that wins on both
    while missing the near-real-time deadline cannot be deployed.
    """
    order = [m for m in ("all-reject", "all-accept", "static-priority",
                         "greedy-value", "intact-ra-cell", "intact-ra",
                         "intact-ra-agentic", "oracle") if m in summaries]
    order += [m for m in summaries if m not in order]
    fig, axes = plt.subplots(1, 4, figsize=(11.0, 3.0))
    labels = [summaries[m].get("label", m) for m in order]
    x = np.arange(len(order))
    cols = [colour(m) for m in order]

    def bars(ax, key, ylabel, fmt="{:.3f}", logy=False):
        v = [summaries[m].get(key, np.nan) for m in order]
        ax.bar(x, v, color=cols, width=0.68)
        if ci:
            lo = [summaries[m].get(key, np.nan)
                  - ci.get(m, {}).get(key, {}).get("lo", np.nan)
                  for m in order]
            hi = [ci.get(m, {}).get(key, {}).get("hi", np.nan)
                  - summaries[m].get(key, np.nan) for m in order]
            if np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)):
                ax.errorbar(x, v, yerr=[np.abs(lo), np.abs(hi)], fmt="none",
                            ecolor="#333333", elinewidth=0.9, capsize=2.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=38, ha="right")
        ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale("symlog", linthresh=1)
        for xi, vi in zip(x, v):
            if np.isfinite(vi):
                ax.annotate(fmt.format(vi), (xi, vi), ha="center",
                            va="bottom", fontsize=6.5,
                            xytext=(0, 1.5), textcoords="offset points")

    bars(axes[0], "wIF", "weighted intent fulfilment")
    axes[0].set_title("(a) wIF  (higher is better)")
    bars(axes[1], "safety_crossings", "safety crossings", "{:.0f}", logy=True)
    axes[1].set_title("(b) safety crossings  (lower)")
    bars(axes[2], "writes_per_epoch", "writes / epoch", "{:.2f}")
    axes[2].set_title("(c) actuation cost  (lower)")
    bars(axes[3], "latency_ms_p95", "decision latency p95 (ms)", "{:.2f}")
    axes[3].set_title("(d) near-RT latency p95")
    if title:
        fig.suptitle(title, fontsize=10)
    return _save(fig, path)


def fig_per_intent(summaries: Dict[str, Dict], path: Path,
                   intents: Sequence[str]) -> Path:
    """Per-intent fulfilment. Exposes a controller that wins the average
    by sacrificing one tenant."""
    order = [m for m in summaries]
    fig, ax = plt.subplots(figsize=(1.6 + 0.85 * len(intents), 3.2))
    w = 0.8 / max(len(order), 1)
    for k, m in enumerate(order):
        v = [summaries[m].get(f"fulfilment_{i}", np.nan) for i in intents]
        ax.bar(np.arange(len(intents)) + k * w, v, width=w,
               label=summaries[m].get("label", m), color=colour(m))
    ax.set_xticks(np.arange(len(intents)) + 0.4 - w / 2)
    ax.set_xticklabels(intents)
    ax.set_ylabel("fulfilment rate")
    ax.set_xlabel("intent")
    ax.axhline(1.0, color="#999999", lw=0.6, ls=":")
    ax.set_title("Per-intent fulfilment: does the average hide a sacrifice?")
    ax.legend(ncol=2, loc="lower right")
    return _save(fig, path)


def fig_drift_response(records: Dict[str, List], path: Path,
                       change_epoch: Optional[int] = None,
                       window: int = 25) -> Path:
    """Rolling wIF and rolling prediction error through the plant change.

    The point of the figure is the SHAPE either side of the change: a
    frozen table should track well before it and diverge after, and an
    adaptive one should recover.  A single end-of-run number cannot show
    that.
    """
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True)
    for m, recs in records.items():
        ep = np.array([r.epoch for r in recs])
        pi = {}
        for r in recs:
            pi.update(r.pi_class)
        iids = sorted({i for r in recs for i in r.g_after})
        wsum = sum(pi.get(i, 1.0) for i in iids) or 1.0
        inst = np.array([
            sum(pi.get(i, 1.0) * (1.0 if r.g_after.get(i, -1) >= 0 else 0.0)
                for i in iids) / wsum for r in recs])
        k = np.ones(window) / window
        if inst.size > window:
            axes[0].plot(ep[window - 1:], np.convolve(inst, k, "valid"),
                         color=colour(m), lw=1.4,
                         label=m)
        err = []
        for r in recs:
            e = [abs((r.g_after[i] - r.g_before[i]) - p)
                 for i, p in r.predicted.items()
                 if i in r.g_after and i in r.g_before and r.n_writes > 0]
            err.append(np.mean(e) if e else np.nan)
        err = np.array(err, dtype=float)
        ok = np.isfinite(err)
        if ok.sum() > 5:
            e2 = np.interp(np.arange(err.size), np.flatnonzero(ok), err[ok])
            if e2.size > window:
                axes[1].plot(ep[window - 1:], np.convolve(e2, k, "valid"),
                             color=colour(m), lw=1.4, label=m)
    if change_epoch is not None:
        for ax in axes:
            ax.axvline(change_epoch, color="#c44e52", ls="--", lw=1.0)
        axes[0].annotate("plant change", (change_epoch, 0.02),
                         xycoords=("data", "axes fraction"),
                         rotation=90, fontsize=7, color="#c44e52",
                         xytext=(3, 0), textcoords="offset points")
    axes[0].set_ylabel(f"wIF ({window}-epoch mean)")
    axes[0].set_title("Response to the plant change")
    axes[0].legend(ncol=3, loc="lower left")
    axes[1].set_ylabel("prediction |error|")
    axes[1].set_xlabel("epoch")
    axes[1].set_title("Sensitivity-model error: does the table notice?")
    return _save(fig, path)


def fig_ablation(summaries: Dict[str, Dict], path: Path,
                 full: str = "intact-ra-agentic") -> Path:
    """What each component contributes, as a drop from the full method."""
    base = summaries.get(full, {}).get("wIF", np.nan)
    abl = [m for m in summaries if m.startswith("agentic-")]
    if not abl:
        abl = [m for m in summaries if m != full]
    names = [full] + sorted(abl)
    vals = [summaries[m].get("wIF", np.nan) for m in names]
    drops = [v - base for v in vals]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    cols = ["#55a868"] + ["#c44e52" if d < 0 else "#4c72b0"
                          for d in drops[1:]]
    ax.bar(np.arange(len(names)), drops, color=cols, width=0.65)
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels([summaries[m].get("label", m) for m in names],
                       rotation=38, ha="right")
    ax.set_ylabel(f"wIF - wIF({full})")
    ax.set_title("Component contributions (negative = removing it hurts)")
    for i, d in enumerate(drops):
        if np.isfinite(d):
            ax.annotate(f"{d:+.3f}", (i, d), ha="center",
                        va="bottom" if d >= 0 else "top", fontsize=6.5,
                        xytext=(0, 2 if d >= 0 else -2),
                        textcoords="offset points")
    return _save(fig, path)


def fig_safety_vs_value(summaries: Dict[str, Dict], path: Path) -> Path:
    """The trade-off plane. The claim is the UPPER-LEFT corner."""
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    for m, s in summaries.items():
        x, y = s.get("safety_crossings", np.nan), s.get("wIF", np.nan)
        ax.scatter(max(x, 0.3), y, s=70, color=colour(m), zorder=3,
                   edgecolor="white", linewidth=0.8)
        ax.annotate(s.get("label", m), (max(x, 0.3), y), fontsize=7,
                    xytext=(5, 3), textcoords="offset points")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("safety crossings (log)")
    ax.set_ylabel("weighted intent fulfilment")
    ax.set_title("Value against harm\n(upper left is better)")
    return _save(fig, path)


def fig_latency(records: Dict[str, List], path: Path,
                budget_ms: float = 10.0) -> Path:
    """Decision-latency distribution against the near-RT budget."""
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    for m, recs in records.items():
        v = np.array([r.latency_ms for r in recs])
        v = np.sort(v[np.isfinite(v)])
        if v.size:
            ax.plot(v, np.linspace(0, 1, v.size), color=colour(m), lw=1.4,
                    label=f"{m} (p95 {np.percentile(v, 95):.2f} ms)")
    ax.axvline(budget_ms, color="#c44e52", ls="--", lw=1.0)
    ax.annotate(f"near-RT budget {budget_ms:g} ms", (budget_ms, 0.06),
                fontsize=7, color="#c44e52", rotation=90,
                xytext=(3, 0), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("arbitration latency (ms)")
    ax.set_ylabel("empirical CDF")
    ax.set_title("Does the decision fit inside the near-RT window?")
    ax.legend(loc="lower right")
    return _save(fig, path)


def fig_slope_drift(s_early: Dict, s_late: Dict, path: Path,
                    top: int = 12) -> Path:
    """True slopes early against late. Points off the diagonal are stale
    table entries; points in the off-sign quadrants are entries whose
    recommendation has REVERSED."""
    keys = [k for k in sorted(set(s_early) & set(s_late))
            if abs(s_early[k]) > 1e-5 or abs(s_late[k]) > 1e-5]
    keys = sorted(keys, key=lambda k: -abs(s_late[k] - s_early[k]))[:top]
    a = np.array([s_early[k] for k in keys])
    b = np.array([s_late[k] for k in keys])
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    lim = max(np.abs(np.concatenate([a, b])).max() * 1.15, 1e-4)
    ax.axhspan(-lim, 0, xmin=0.5, xmax=1.0, color="#c44e52", alpha=0.07)
    ax.axhspan(0, lim, xmin=0.0, xmax=0.5, color="#c44e52", alpha=0.07)
    ax.plot([-lim, lim], [-lim, lim], color="#999999", lw=0.8, ls=":")
    ax.axhline(0, color="#333333", lw=0.7)
    ax.axvline(0, color="#333333", lw=0.7)
    flip = (a * b) < 0
    ax.scatter(a[~flip], b[~flip], s=46, color="#4c72b0", zorder=3,
               edgecolor="white", linewidth=0.7, label="magnitude change")
    ax.scatter(a[flip], b[flip], s=70, color="#c44e52", marker="D", zorder=4,
               edgecolor="white", linewidth=0.7, label="SIGN FLIP")
    for k, xa, yb in zip(keys, a, b):
        ax.annotate(f"{k[0]}->{k[1]}", (xa, yb), fontsize=6,
                    xytext=(4, 2), textcoords="offset points")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("true slope, early (calibration window)")
    ax.set_ylabel("true slope, late (after the plant moved)")
    ax.set_title("Where the frozen table went wrong\n"
                 "(shaded quadrants: the recommendation reversed)")
    ax.legend(loc="upper left")
    return _save(fig, path)


# ===========================================================================
# RAN FIGURES
# ===========================================================================
def fig_causal_chain(cell: "pd.DataFrame", perf: "pd.DataFrame",
                     path: Path, tenants: Sequence[str]) -> Path:
    """The synchronised causal chain on one shared time axis.

    mobility -> path loss / SINR -> CQI, MCS -> capacity -> offered load
    -> PRB allocation -> delivered throughput -> delay -> QoS regret

    Shared x axis on purpose: the claim that a geometry change propagates
    all the way to an SLA breach is only legible if the reader can drop a
    vertical line through every panel at once.
    """
    fig = plt.figure(figsize=(8.2, 11.0))
    gs = gridspec.GridSpec(8, 1, figure=fig, hspace=0.12)
    axes = [fig.add_subplot(gs[i, 0]) for i in range(8)]
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)

    def per_tenant(ax, col, ylabel, title):
        for k, t in enumerate(tenants):
            d = perf[perf["tenant"] == t]
            if col in d and len(d):
                ax.plot(d["epoch"], d[col], lw=1.0,
                        color=TENANT_COLOURS[k % len(TENANT_COLOURS)],
                        label=t)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=8.5)

    per_tenant(axes[0], "mean_dist_m", "distance (m)",
               "(1) mobility: mean UE distance from the gNB")
    axes[0].legend(ncol=4, loc="upper left")
    per_tenant(axes[1], "sinr_db", "SINR (dB)",
               "(2) radio channel: wideband SINR after interference")
    per_tenant(axes[2], "mcs", "MCS index",
               "(3) link adaptation: selected MCS")
    per_tenant(axes[3], "spectral_efficiency", "bit/s/Hz",
               "(4) resulting spectral efficiency")
    per_tenant(axes[4], "offered_slice_mbps", "Mb/s",
               "(5) offered traffic (exogenous: never a function of control)")
    per_tenant(axes[5], "prb_alloc", "PRBs",
               "(6) scheduler: PRBs actually allocated")
    per_tenant(axes[6], "throughput_mbps", "Mb/s/UE",
               "(7) delivered per-UE throughput")
    per_tenant(axes[7], "delay_p50_ms", "ms",
               "(8) median packet delay")
    axes[7].set_xlabel("epoch")
    axes[7].tick_params(labelbottom=True)
    fig.suptitle("RAN causal chain: geometry to SLA, on one time axis",
                 fontsize=10)
    return _save(fig, path)


def fig_ran_distributions(ue: "pd.DataFrame", path: Path,
                          tenants: Sequence[str]) -> Path:
    """Distributions the time series hides: a mean SINR of 15 dB with a
    20 dB spread is a different cell from a tight 15 dB."""
    cols = [("sinr_db", "SINR (dB)"), ("cqi", "CQI"),
            ("bler", "first-transmission BLER"),
            ("d2d_m", "distance (m)")]
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.2))
    for ax, (c, lab) in zip(axes.ravel(), cols):
        if c not in ue:
            ax.set_visible(False)
            continue
        for k, t in enumerate(tenants):
            v = ue.loc[ue["tenant"] == t, c].to_numpy() if "tenant" in ue \
                else np.array([])
            v = v[np.isfinite(v)]
            if v.size > 10:
                ax.hist(v, bins=45, histtype="step", density=True, lw=1.2,
                        color=TENANT_COLOURS[k % len(TENANT_COLOURS)],
                        label=t)
        ax.set_xlabel(lab)
        ax.set_ylabel("density")
    axes[0, 0].legend(ncol=4)
    fig.suptitle("Per-UE distributions across the run", fontsize=10)
    return _save(fig, path)


def fig_cell_overview(cell: "pd.DataFrame", path: Path) -> Path:
    """Cell-level load, fairness and the cost of reconfiguration."""
    fig, axes = plt.subplots(2, 2, figsize=(7.8, 4.8))
    panels = [
        ("prb_util_pct", "PRB utilisation (%)", "Cell load"),
        ("mean_sinr_db", "mean SINR (dB)", "Cell radio quality"),
        ("jain_throughput", "Jain index", "Throughput fairness across UEs"),
        ("reconfig_prb", "PRBs lost", "Reconfiguration transient cost"),
    ]
    for ax, (c, ylab, title) in zip(axes.ravel(), panels):
        if c in cell:
            ax.plot(cell["epoch"] if "epoch" in cell else np.arange(len(cell)),
                    cell[c], lw=1.0, color="#4c72b0")
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left", fontsize=8.5)
        ax.set_xlabel("epoch")
    return _save(fig, path)


def fig_agent_internals(agent: "pd.DataFrame", sens: "pd.DataFrame",
                        path: Path) -> Path:
    """What the agent was doing: candidate count, entropy, probes,
    residuals and promotions."""
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.0), sharex=True)
    if "candidates" in agent:
        axes[0].plot(agent["epoch"], agent["candidates"], lw=1.0,
                     color="#4c72b0", label="candidates scored")
    if "entropy" in agent:
        ax2 = axes[0].twinx()
        ax2.plot(agent["epoch"], agent["entropy"], lw=1.0, color="#dd8452",
                 label="policy entropy")
        ax2.set_ylabel("policy entropy", color="#dd8452")
        ax2.grid(False)
    axes[0].set_ylabel("candidates")
    axes[0].set_title("(a) search effort and policy exploration", loc="left")

    if "probe" in agent:
        pe = agent.loc[agent["probe"] > 0, "epoch"].to_numpy()
        axes[1].vlines(pe, 0, 1, color="#55a868", lw=1.0)
        axes[1].set_ylim(0, 1.3)
        axes[1].set_yticks([])
        axes[1].set_title(f"(b) supervisor calibration probes "
                          f"({len(pe)} issued)", loc="left")

    if sens is not None and len(sens) and "residual" in sens:
        w = 30
        r = sens.groupby("epoch")["residual"].apply(
            lambda s: np.mean(np.abs(s)))
        if len(r) > w:
            axes[2].plot(r.index[w - 1:],
                         np.convolve(r.to_numpy(), np.ones(w) / w, "valid"),
                         lw=1.2, color="#c44e52")
        axes[2].set_ylabel("|residual|")
    if "sens_version" in agent:
        v = agent["sens_version"].to_numpy()
        ch = np.flatnonzero(np.diff(v) > 0) + 1
        for i in ch:
            axes[2].axvline(agent["epoch"].to_numpy()[i], color="#55a868",
                            ls="--", lw=0.8)
        axes[2].set_title(f"(c) model residual, with {len(ch)} promotion(s) "
                          f"marked", loc="left")
    axes[2].set_xlabel("epoch")
    return _save(fig, path)
