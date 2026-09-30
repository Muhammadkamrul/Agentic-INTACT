"""
intact_agentic/telemetry.py
===========================
Instrumentation.  Separated from the simulator on purpose: the simulator
calls exactly one hook (``record_slot``) and one aggregator
(``record_epoch``), so reporting can be switched off with
``report.telemetry.enabled: false`` and the numerical results do not move
by one bit.  A self-test in ``scripts/selftest.py`` asserts that.

WHAT IS RECORDED, AND AT WHAT RATE
----------------------------------
Per-UE state is the expensive one: a 600-epoch run with 16 slots per
epoch and 45 UEs is 432 000 rows.  It is therefore DECIMATED by
``report.telemetry.ue_every_slots`` (default 8) while the per-slice and
per-cell series are recorded every slot.  Decimation never changes the
simulation, only what is written down, and the decimation factor is
stamped into every file so no one can mistake a subsample for a census.

FILES PRODUCED  (matching the structure requested in the RAN reporting
brief, one directory per run)

    config/experiment_config.json   full resolved configuration
    config/metric_catalogue.csv     every metric: definition, unit,
                                    source, aggregation, formula
    csv/ue_state.csv                per-UE per-sampled-slot geometry
    csv/channel.csv                 per-UE per-sampled-slot propagation
    csv/phy.csv                     per-UE per-sampled-slot link adaptation
    csv/traffic.csv                 per-slice per-slot offered load
    csv/resource_allocation.csv     per-slice per-slot PRBs and shares
    csv/performance.csv             per-slice per-slot delivered KPIs
    csv/cell.csv                    per-slot whole-cell aggregates
    csv/qos_sla.csv                 per-epoch per-intent satisfaction
    csv/epoch_kpm.csv               per-epoch per-tenant KPM snapshot
    csv/decisions.csv               every arbitration decision
    csv/sensitivity_log.csv         predicted vs observed margin change
    csv/agent.csv                   DRL/agent internals when enabled

A metric that the simulator does not model is NOT written: it appears in
``metric_catalogue.csv`` with availability ``not_modelled`` and a reason.
Nothing is fabricated to fill a column.
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
class _CSVSink:
    """Append-only CSV writer that learns its header from the first row."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = None
        self._w = None
        self._fields: Optional[List[str]] = None
        self.rows = 0

    def write(self, row: Dict[str, Any]) -> None:
        if self._fh is None:
            self._fields = list(row.keys())
            self._fh = open(self.path, "w", newline="", encoding="utf-8")
            self._w = csv.DictWriter(self._fh, fieldnames=self._fields)
            self._w.writeheader()
        self._w.writerow({k: row.get(k, "") for k in self._fields})
        self.rows += 1

    def write_many(self, rows: Iterable[Dict[str, Any]]) -> None:
        for r in rows:
            self.write(r)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


# ---------------------------------------------------------------------------
class Telemetry:
    """Collects everything and flushes it to a run directory."""

    def __init__(self, rundir: os.PathLike, cfg: Dict, enabled: bool = True):
        self.root = Path(rundir)
        self.cfg = cfg
        rep = (cfg.get("report", {}) or {}).get("telemetry", {}) or {}
        self.enabled = bool(enabled and rep.get("enabled", True))
        self.ue_every = max(int(rep.get("ue_every_slots", 8)), 1)
        self.slice_every = max(int(rep.get("slice_every_slots", 1)), 1)
        self.max_ue_rows = int(rep.get("max_ue_rows", 400_000))
        self.run_id = str(cfg.get("_run_id", "run"))
        self.method = str(cfg.get("_method", "unknown"))
        self.scenario = str(cfg.get("_scenario_name", "base"))

        self._sinks: Dict[str, _CSVSink] = {}
        if self.enabled:
            for name in ("ue_state", "channel", "phy", "traffic",
                         "resource_allocation", "performance", "cell",
                         "qos_sla", "epoch_kpm", "decisions",
                         "sensitivity_log", "agent", "writes"):
                self._sinks[name] = _CSVSink(self.root / "csv" / f"{name}.csv")
        self.ue_rows = 0

    # ------------------------------------------------------------------
    def _base(self, **kw) -> Dict[str, Any]:
        d = {"run_id": self.run_id, "method": self.method,
             "scenario": self.scenario}
        d.update(kw)
        return d

    # ------------------------------------------------------------------
    def record_slot(self, *, t: int, epoch: int, tid: str, group, phy,
                    prb, delivered, queue, dropped, arrivals, delay_ms,
                    slot_s: float) -> None:
        """Per-UE and per-slice record for one slot of one slice."""
        if not self.enabled:
            return
        n = group.n
        # --- per-slice rows (every slot) ------------------------------
        if t % self.slice_every == 0:
            self._sinks["traffic"].write(self._base(
                slot=t, epoch=epoch, tenant=tid,
                offered_mbps=float(arrivals.sum() / slot_s / 1e6),
                arrival_bytes=float(arrivals.sum() / 8.0),
                backlog_kb=float(queue.sum() / 8e3),
                n_ue=n))
            self._sinks["resource_allocation"].write(self._base(
                slot=t, epoch=epoch, tenant=tid,
                prb_alloc=float(prb.sum()),
                prb_per_ue_mean=float(prb.mean()) if n else 0.0,
                retx_prb=float((prb * (phy["harq_tx"] - 1.0)).sum())))
            self._sinks["performance"].write(self._base(
                slot=t, epoch=epoch, tenant=tid,
                delivered_mbps=float(delivered.sum() / slot_s / 1e6),
                mean_ue_mbps=float(delivered.mean() / slot_s / 1e6)
                if n else 0.0,
                delay_ms=float(np.mean(delay_ms)),
                p95_delay_ms=float(np.percentile(delay_ms, 95)) if n else 0.0,
                queue_kb=float(queue.mean() / 8e3) if n else 0.0,
                dropped_kb=float(dropped.sum() / 8e3),
                bler=float(np.mean(phy["bler"])),
                resid_bler=float(np.mean(phy["resid_bler"])),
                spectral_efficiency=float(np.mean(phy["se_mcs"]))))

        # --- per-UE rows (decimated) ----------------------------------
        if t % self.ue_every or self.ue_rows >= self.max_ue_rows:
            return
        for u in range(n):
            self._sinks["ue_state"].write(self._base(
                slot=t, epoch=epoch, tenant=tid, ue_id=f"{tid}_u{u}",
                x_m=float(group.xy[u, 0]), y_m=float(group.xy[u, 1]),
                dist_m=float(phy["d2d_m"][u]),
                speed_mps=float(group.speed_mps()[u]),
                queue_kb=float(queue[u] / 8e3),
                offered_bits=float(arrivals[u]),
                delivered_mbps=float(delivered[u] / slot_s / 1e6),
                delay_ms=float(delay_ms[u])))
            self._sinks["channel"].write(self._base(
                slot=t, epoch=epoch, tenant=tid, ue_id=f"{tid}_u{u}",
                dist_m=float(phy["d2d_m"][u]),
                los=int(phy["los"][u]),
                pathloss_db=float(phy["pathloss_db"][u]),
                shadow_db=float(phy["shadow_db"][u]),
                fading_db=float(phy["fading_db"][u]),
                antenna_gain_db=float(phy["antenna_gain_db"][u]),
                rx_dbm=float(phy["rx_dbm"][u]),
                interf_dbm=float(phy["interf_dbm"][u]),
                noise_dbm=float(phy["noise_dbm"][u]),
                snr_db=float(phy["snr_db"][u]),
                sinr_db=float(phy["sinr_db"][u])))
            self._sinks["phy"].write(self._base(
                slot=t, epoch=epoch, tenant=tid, ue_id=f"{tid}_u{u}",
                sinr_db=float(phy["sinr_db"][u]),
                cqi=int(phy["cqi"][u]), mcs=int(phy["mcs"][u]),
                se_mcs=float(phy["se_mcs"][u]),
                se_shannon=float(phy["se_shannon"][u]),
                bler=float(phy["bler"][u]),
                resid_bler=float(phy["resid_bler"][u]),
                harq_tx=float(phy["harq_tx"][u]),
                prb=float(prb[u])))
            self.ue_rows += 1

    # ------------------------------------------------------------------
    def record_cell_slot(self, row: Dict[str, Any]) -> None:
        if self.enabled:
            self._sinks["cell"].write(self._base(**row))

    def record_epoch_kpm(self, epoch: int, kpm: Dict[str, Dict[str, float]]
                         ) -> None:
        if not self.enabled:
            return
        for key, vals in kpm.items():
            row = self._base(epoch=epoch, tenant=key)
            row.update({k: float(v) for k, v in vals.items()
                        if isinstance(v, (int, float))})
            self._sinks["epoch_kpm"].write(row)

    def record_qos(self, rows: Sequence[Dict[str, Any]]) -> None:
        if self.enabled:
            self._sinks["qos_sla"].write_many(self._base(**r) for r in rows)

    def record_decisions(self, rows: Sequence[Dict[str, Any]]) -> None:
        if self.enabled:
            self._sinks["decisions"].write_many(self._base(**r) for r in rows)

    def record_writes(self, rows: Sequence[Dict[str, Any]]) -> None:
        if self.enabled:
            self._sinks["writes"].write_many(self._base(**r) for r in rows)

    def record_sensitivity(self, rows: Sequence[Dict[str, Any]]) -> None:
        if self.enabled:
            self._sinks["sensitivity_log"].write_many(
                self._base(**r) for r in rows)

    def record_agent(self, row: Dict[str, Any]) -> None:
        if self.enabled:
            self._sinks["agent"].write(self._base(**row))

    # ------------------------------------------------------------------
    def close(self) -> None:
        for s in self._sinks.values():
            s.close()

    def row_counts(self) -> Dict[str, int]:
        return {k: v.rows for k, v in self._sinks.items()}


# ---------------------------------------------------------------------------
# Metric catalogue: definition, unit, source, aggregation, formula.
# Required by the reporting brief and by any reviewer who wants to know
# whether a number was measured or derived.
# ---------------------------------------------------------------------------
METRIC_CATALOGUE: List[Dict[str, str]] = [
    dict(metric="x_m,y_m", unit="m", availability="modelled",
         source="MobilityModel.groups[tid].xy", aggregation="instantaneous",
         formula="state variable integrated from velocity each slot",
         definition="UE position in the plane, origin at the serving gNB"),
    dict(metric="dist_m", unit="m", availability="derived",
         source="MobilityModel", aggregation="instantaneous",
         formula="sqrt(x^2 + y^2)",
         definition="2-D distance from the UE to the serving gNB"),
    dict(metric="speed_mps", unit="m/s", availability="modelled",
         source="MobilityModel", aggregation="instantaneous",
         formula="|v|", definition="UE scalar speed"),
    dict(metric="los", unit="flag", availability="modelled",
         source="ChannelModel", aggregation="instantaneous",
         formula="Bernoulli(P_LOS(d2D)) drawn at spawn, TR 38.901 UMa",
         definition="1 if the UE has a line-of-sight link"),
    dict(metric="pathloss_db", unit="dB", availability="modelled",
         source="ChannelModel.path_loss_db", aggregation="instantaneous",
         formula="TR 38.901 UMa LOS: 28+22log10(d3D)+20log10(fc); "
                 "NLOS: 13.54+39.08log10(d3D)+20log10(fc)-0.6(hUT-1.5)",
         definition="large-scale propagation loss"),
    dict(metric="shadow_db", unit="dB", availability="modelled",
         source="ChannelModel._update_shadow", aggregation="instantaneous",
         formula="AR(1) with rho=exp(-dx/37 m), sigma 4 dB LOS / 6 dB NLOS",
         definition="spatially correlated log-normal shadowing"),
    dict(metric="fading_db", unit="dB", availability="modelled",
         source="ChannelModel._update_fading", aggregation="instantaneous",
         formula="Rician (K=9 dB) if LOS else Rayleigh, AR(1) with "
                 "rho~J0(2 pi fd Ts), fd = v fc / c",
         definition="small-scale fading power"),
    dict(metric="antenna_gain_db", unit="dBi", availability="modelled",
         source="ChannelModel.antenna_gain_db", aggregation="instantaneous",
         formula="Gmax - min(12((theta-tilt)/HPBW)^2, SLA_V)",
         definition="3D antenna pattern, vertical cut, with downtilt"),
    dict(metric="interf_dbm", unit="dBm", availability="modelled",
         source="ChannelModel.evaluate", aggregation="instantaneous",
         formula="sum over neighbour sites of load * Prx_per_PRB",
         definition="inter-cell interference power per PRB"),
    dict(metric="noise_dbm", unit="dBm", availability="modelled",
         source="ChannelModel", aggregation="constant",
         formula="-174 + 10log10(W_prb) + NF",
         definition="thermal noise power in one PRB"),
    dict(metric="snr_db", unit="dB", availability="derived",
         source="ChannelModel", aggregation="instantaneous",
         formula="10log10(Prx / N)", definition="signal to noise ratio"),
    dict(metric="sinr_db", unit="dB", availability="derived",
         source="ChannelModel", aggregation="instantaneous",
         formula="10log10(Prx / (I + N)) with the subband-quality factor",
         definition="signal to interference plus noise ratio"),
    dict(metric="cqi", unit="index 0-15", availability="derived",
         source="channel.cqi_from_sinr_db", aggregation="instantaneous",
         formula="threshold table on wideband SINR (38.214 Tab 5.2.2.1-2)",
         definition="channel quality indicator reported by the UE"),
    dict(metric="mcs", unit="index 0-28", availability="derived",
         source="channel.mcs_from_cqi", aggregation="instantaneous",
         formula="min(2*CQI-2, mcs_<tenant> knob)",
         definition="modulation and coding scheme after link adaptation"),
    dict(metric="se_mcs", unit="bit/s/Hz", availability="derived",
         source="channel.se_from_mcs", aggregation="instantaneous",
         formula="Qm * R from 38.214 Table 5.1.3.1-1",
         definition="ACHIEVABLE PHY spectral efficiency at the chosen MCS"),
    dict(metric="se_shannon", unit="bit/s/Hz", availability="derived",
         source="ChannelModel", aggregation="instantaneous",
         formula="log2(1 + SINR)",
         definition="THEORETICAL channel capacity; an upper bound, never "
                    "achieved, reported separately on purpose"),
    dict(metric="bler", unit="fraction", availability="modelled",
         source="channel.bler_from_sinr", aggregation="instantaneous",
         formula="1/(1+exp((SINR - SINR_req(MCS))/1.1))",
         definition="first-transmission block error rate"),
    dict(metric="resid_bler", unit="fraction", availability="derived",
         source="ChannelModel", aggregation="instantaneous",
         formula="bler^(max_harq+1)",
         definition="residual error after HARQ retransmissions"),
    dict(metric="harq_tx", unit="count", availability="derived",
         source="ChannelModel", aggregation="instantaneous",
         formula="(1-bler^(N+1))/(1-bler)",
         definition="expected transmissions per delivered block"),
    dict(metric="offered_mbps", unit="Mb/s", availability="modelled",
         source="TrafficModel.arrivals_bits", aggregation="mean over slots",
         formula="arriving bits / slot duration",
         definition="EXOGENOUS offered traffic; never depends on control"),
    dict(metric="prb_alloc", unit="PRB", availability="modelled",
         source="RealisticRAN.step", aggregation="mean over slots",
         formula="min(PF share * min(quota, cap), backlog need) * "
                 "cell budget scale * (1 - reconfiguration loss)",
         definition="PRBs actually allocated to the slice"),
    dict(metric="delivered_mbps", unit="Mb/s", availability="modelled",
         source="RealisticRAN.step", aggregation="mean over slots",
         formula="min(prb * W * se_mcs * (1-resid_bler)/harq_tx, backlog)"
                 " / slot",
         definition="DELIVERED application throughput (goodput)"),
    dict(metric="phy_rate_mbps", unit="Mb/s", availability="derived",
         source="RealisticRAN.step", aggregation="mean over slots",
         formula="prb * W * se_mcs / slot",
         definition="achievable PHY rate before HARQ and backlog limits"),
    dict(metric="delay_ms", unit="ms", availability="derived",
         source="RealisticRAN.step", aggregation="mean over slots",
         formula="base_delay + 1000 * queue_bits / serve_rate (Little)",
         definition="mean packet delay"),
    dict(metric="jitter_ms", unit="ms", availability="derived",
         source="RealisticRAN.step", aggregation="std over 64-slot window",
         formula="std(delay_ms over the rolling window)",
         definition="delay variation"),
    dict(metric="delivery_pct", unit="%", availability="derived",
         source="RealisticRAN.step", aggregation="mean over slots",
         formula="100 * (1 - dropped_bits / offered_bits)",
         definition="fraction of offered traffic not dropped at the buffer"),
    dict(metric="prb_util_pct", unit="%", availability="derived",
         source="RealisticRAN._assemble_kpm", aggregation="mean over slots",
         formula="100 * (prb_used + prb_retx) / n_prb, capped at 100",
         definition="fraction of the cell's PRBs actually used"),
    dict(metric="offered_load_pct", unit="%", availability="derived",
         source="RealisticRAN._assemble_kpm", aggregation="mean over slots",
         formula="100 * PRB demand (incl. retx) / n_prb; MAY EXCEED 100",
         definition="PRB demand; congestion indicator, not a utilisation"),
    dict(metric="jain_throughput", unit="index", availability="derived",
         source="RealisticRAN._assemble_kpm", aggregation="per epoch",
         formula="(sum x)^2 / (n sum x^2)",
         definition="Jain fairness of per-UE delivered throughput"),
    dict(metric="regret_throughput/delay/bler", unit="fraction",
         availability="derived", source="RealisticRAN._assemble_kpm",
         aggregation="per epoch",
         formula="max((demand-achieved)/demand,0) and the delay/BLER duals",
         definition="xSlice-style per-slice QoS regret components"),
    dict(metric="margin g_i", unit="dimensionless", availability="derived",
         source="arbiter/margins.py", aggregation="per epoch",
         formula="d_i (KPI - target) / |target|, clipped",
         definition="normalised intent margin; the arbiter's only currency"),
    dict(metric="reconfig_prb", unit="PRB", availability="modelled",
         source="RealisticRAN.step", aggregation="sum over slots",
         formula="allocated PRBs * reconfig_loss_frac during the transient",
         definition="capacity lost to control-plane reconfiguration"),
    dict(metric="handover_count", unit="count", availability="not_modelled",
         source="-", aggregation="-", formula="-",
         definition="single-cell association is assumed; UEs never change "
                    "serving cell, so a handover counter would be "
                    "identically zero and is therefore not reported"),
    dict(metric="uplink_metrics", unit="-", availability="not_modelled",
         source="-", aggregation="-", formula="-",
         definition="only the downlink is scheduled; every uplink column "
                    "would be a fabrication"),
    dict(metric="per_subband_cqi", unit="-", availability="not_modelled",
         source="-", aggregation="-", formula="-",
         definition="the channel is wideband with a scalar subband-quality "
                    "factor; per-subband CQI is not resolved"),
]


def write_metric_catalogue(path: os.PathLike) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(METRIC_CATALOGUE[0].keys()))
        w.writeheader()
        for row in METRIC_CATALOGUE:
            w.writerow(row)
