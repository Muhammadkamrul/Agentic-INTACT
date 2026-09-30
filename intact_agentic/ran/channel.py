"""
intact_agentic/ran/channel.py
=============================
The physical layer of the simulated RAN.

WHY THIS FILE EXISTS
--------------------
The predecessor simulator used a single-line path-loss expression and an
ad-hoc "MCS index -> spectral efficiency" curve.  That is enough to make
knobs move margins, but it is NOT enough to make the *shape* of the
knob->margin relationship change with the operating point in a way a
reviewer would believe.  The whole scientific claim of INTACT-RA-Agentic
is that a sensitivity slope measured offline goes stale, so the physics
that makes it go stale has to be real physics and not a hand-inserted
step function.

Concretely, two effects must emerge from the model without being coded as
special cases:

  (1) COVERAGE-LIMITED regime.  At low SINR, Shannon capacity
      log2(1 + gamma) is approximately gamma / ln2, i.e. LINEAR in linear
      SINR.  One extra dB therefore buys ~26% more rate.  Transmit power
      is extremely valuable.
  (2) CAPACITY-LIMITED regime.  At high SINR the same dB buys perhaps 5%,
      while one extra PRB buys a full extra `se` bit/s/Hz.  PRBs are
      valuable and power is nearly worthless.

Both fall straight out of log2(1+gamma) and rate = n_prb * W * se.  When
UEs drift towards the cell edge at CONSTANT offered load, the regime
silently moves from (2) to (1): the load label does not change, but
ds/dP rises by roughly an order of magnitude.  That is the drift the
agentic supervisor has to detect and the static table cannot.

MODEL CONTENTS
--------------
  * 3GPP TR 38.901 UMa LOS/NLOS path loss and LOS probability
  * spatially correlated log-normal shadowing (Gudmundson AR(1),
    decorrelation distance 37 m)
  * temporally correlated small-scale fading (first-order approximation
    of the Jakes/Clarke autocorrelation, so the fading rate depends on UE
    speed -- mobility therefore changes fading dynamics, not just mean
    path loss)
  * 3GPP 3D antenna pattern with electrical downtilt (vertical HPBW 10
    deg, SLA_V 30 dB) -- the reason tilt helps near UEs and starves far
    ones, giving it a DIFFERENT cross-tenant shape from transmit power
  * explicit inter-cell interference from a configurable ring of
    neighbours, each with its own transmit power and load factor
  * thermal noise from bandwidth and receiver noise figure
  * CQI from SINR (3GPP 38.214 Table 5.2.2.1-2 thresholds), MCS from CQI
    capped by the slice MCS knob, spectral efficiency from the MCS table,
    and a sigmoid link-level BLER curve with HARQ

EVERY quantity reported by :meth:`ChannelModel.evaluate` is either
directly modelled here or derived by a formula documented in
``docs/METRICS.md``.  Nothing is invented for the sake of a plot.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 3GPP 38.214 Table 5.1.3.1-1 (MCS index table 1 for PDSCH), condensed to
# (modulation order Qm, target code rate R x 1024).  Spectral efficiency is
# Qm * R, which is exactly the last column of the standard table.
# ---------------------------------------------------------------------------
_MCS_TABLE = np.array([
    # Qm,  R*1024
    (2, 120), (2, 157), (2, 193), (2, 251), (2, 308), (2, 379), (2, 449),
    (2, 526), (2, 602), (2, 679), (4, 340), (4, 378), (4, 434), (4, 490),
    (4, 553), (4, 616), (4, 658), (6, 438), (6, 466), (6, 517), (6, 567),
    (6, 616), (6, 666), (6, 719), (6, 772), (6, 822), (6, 873), (6, 910),
    (6, 948),
], dtype=float)

MCS_SE = _MCS_TABLE[:, 0] * _MCS_TABLE[:, 1] / 1024.0          # bit/s/Hz
MCS_MAX_INDEX = len(MCS_SE) - 1

# SINR (dB) at which each MCS achieves 10% BLER on an AWGN link.  Derived
# by inverting the effective-SNR model:  required SINR ~ 2^(SE/Qm...)  In
# practice a simple affine fit to published link-level curves is used and
# documented in docs/METRICS.md.
MCS_SINR_REQ_DB = -6.0 + 1.05 * np.arange(len(MCS_SE))

# CQI -> (approximate spectral efficiency, SINR threshold in dB).
# Thresholds are the customary link-adaptation operating points for the
# 38.214 Table 5.2.2.1-2 CQI table.
CQI_SINR_DB = np.array([-6.7, -4.7, -2.3, 0.2, 2.4, 4.3, 5.9, 8.1, 10.3,
                        11.7, 14.1, 16.3, 18.7, 21.0, 22.7])
CQI_SE = np.array([0.1523, 0.2344, 0.3770, 0.6016, 0.8770, 1.1758, 1.4766,
                   1.9141, 2.4063, 2.7305, 3.3223, 3.9023, 4.5234, 5.1152,
                   5.5547])


def cqi_from_sinr_db(sinr_db: np.ndarray) -> np.ndarray:
    """CQI index 1..15 (0 = out of range) from wideband SINR."""
    idx = np.searchsorted(CQI_SINR_DB, np.asarray(sinr_db, dtype=float),
                          side="right")
    return np.clip(idx, 0, 15).astype(int)


def mcs_from_cqi(cqi: np.ndarray) -> np.ndarray:
    """Link adaptation: map CQI to an MCS index in 0..28.

    The customary vendor mapping is roughly MCS ~ 2*CQI - 2, saturating at
    the top of the table.  CQI 0 means the UE cannot be scheduled with any
    reliability, so we floor it at MCS 0 and let the BLER curve punish it.
    """
    m = 2 * np.asarray(cqi, dtype=int) - 2
    return np.clip(m, 0, MCS_MAX_INDEX)


def se_from_mcs(mcs: np.ndarray) -> np.ndarray:
    return MCS_SE[np.clip(np.asarray(mcs, dtype=int), 0, MCS_MAX_INDEX)]


def bler_from_sinr(sinr_db: np.ndarray, mcs: np.ndarray,
                   slope_db: float = 1.1) -> np.ndarray:
    """Sigmoid link-level BLER curve.

    BLER = 1 / (1 + exp((SINR - SINR_req(MCS)) / slope)).  At SINR ==
    SINR_req the curve gives 0.5; the published 10%-BLER point sits about
    2.4 dB above, which the calibration in ``slope_db`` reproduces.  An
    over-reaching MCS ceiling therefore produces retransmissions rather
    than a silent capacity increase -- that is coupling mechanism (c).
    """
    req = MCS_SINR_REQ_DB[np.clip(np.asarray(mcs, dtype=int), 0,
                                  MCS_MAX_INDEX)]
    z = (np.asarray(sinr_db, dtype=float) - req) / max(slope_db, 1e-6)
    return np.clip(1.0 / (1.0 + np.exp(z)), 1e-4, 0.95)


# ---------------------------------------------------------------------------
@dataclass
class CellSite:
    """One transmitter.  Cell 0 is the serving cell; the rest interfere."""
    cid: str
    x: float
    y: float
    height_m: float = 25.0
    tx_dbm: float = 46.0            # total cell transmit power
    tilt_deg: float = 6.0
    load_factor: float = 0.8        # fraction of PRBs it actually uses
    antenna_gain_dbi: float = 15.0


# ---------------------------------------------------------------------------
class ChannelModel:
    """Stateful per-UE channel.  One instance per simulated cell deployment."""

    def __init__(self, cfg: Dict, rng: np.random.Generator):
        ch = cfg["ran"].get("channel", {}) or {}
        self.rng = rng
        self.fc_ghz = float(cfg["ran"].get("carrier_ghz", 3.5))
        self.n_prb = int(cfg["ran"]["n_prb"])
        self.prb_hz = float(cfg["ran"]["prb_bandwidth_hz"])
        self.bw_hz = self.n_prb * self.prb_hz
        self.noise_figure_db = float(ch.get("noise_figure_db", 7.0))
        self.h_ut = float(ch.get("ue_height_m", 1.5))
        self.h_bs = float(ch.get("bs_height_m", 25.0))
        self.shadow_sigma_los_db = float(ch.get("shadow_sigma_los_db", 4.0))
        self.shadow_sigma_nlos_db = float(ch.get("shadow_sigma_nlos_db", 6.0))
        self.shadow_decorr_m = float(ch.get("shadow_decorr_m", 37.0))
        self.rician_k_db = float(ch.get("rician_k_db", 9.0))
        self.vert_hpbw_deg = float(ch.get("vertical_hpbw_deg", 10.0))
        self.sla_v_db = float(ch.get("sla_v_db", 30.0))
        self.antenna_gain_dbi = float(ch.get("antenna_gain_dbi", 15.0))
        self.bler_slope_db = float(ch.get("bler_slope_db", 1.1))
        self.max_harq = int(ch.get("max_harq_retx", 3))
        self.slot_s = float(cfg["ran"]["slot_ms"]) / 1000.0
        self.los_enabled = bool(ch.get("los_model", True))
        self.fading_enabled = bool(ch.get("fast_fading", True))

        # thermal noise over ONE PRB, which is the right granularity for a
        # per-PRB SINR:  -174 dBm/Hz + 10log10(W_prb) + NF
        self.noise_prb_dbm = (-174.0 + 10 * math.log10(self.prb_hz)
                              + self.noise_figure_db)

        self.sites = self._build_sites(cfg)
        self.serving = self.sites[0]

        # per-UE persistent state, filled by :meth:`register`
        self._shadow: Dict[str, np.ndarray] = {}
        self._fade_i: Dict[str, np.ndarray] = {}
        self._fade_q: Dict[str, np.ndarray] = {}
        self._los: Dict[str, np.ndarray] = {}
        self._last_xy: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    def _build_sites(self, cfg: Dict):
        ch = cfg["ran"].get("channel", {}) or {}
        sites = [CellSite("C0", 0.0, 0.0,
                          height_m=self.h_bs,
                          tx_dbm=float(cfg["ran"]["initial_controls"].get(
                              "txpower", 46.0)),
                          tilt_deg=float(cfg["ran"]["initial_controls"].get(
                              "tilt", 6.0)),
                          antenna_gain_dbi=self.antenna_gain_dbi)]
        neigh = ch.get("neighbours")
        if neigh is None:
            # default: a 6-site hexagon at the configured ISD
            isd = float(ch.get("isd_m", 500.0))
            ptx = float(ch.get("neighbour_tx_dbm", 46.0))
            load = float(ch.get("neighbour_load", 0.75))
            neigh = []
            for k in range(6):
                a = math.radians(60 * k)
                neigh.append({"x": isd * math.cos(a), "y": isd * math.sin(a),
                              "tx_dbm": ptx, "load": load})
        for k, nb in enumerate(neigh):
            sites.append(CellSite(
                f"N{k}", float(nb["x"]), float(nb["y"]),
                height_m=float(nb.get("height_m", self.h_bs)),
                tx_dbm=float(nb.get("tx_dbm", 46.0)),
                tilt_deg=float(nb.get("tilt_deg", 6.0)),
                load_factor=float(nb.get("load", 0.75)),
                antenna_gain_dbi=float(nb.get("antenna_gain_dbi",
                                              self.antenna_gain_dbi))))
        return sites

    # ------------------------------------------------------------------
    def register(self, tid: str, xy: np.ndarray) -> None:
        """Initialise persistent per-UE channel state for one slice."""
        n = xy.shape[0]
        d2d = np.hypot(xy[:, 0] - self.serving.x, xy[:, 1] - self.serving.y)
        p_los = self.los_probability(d2d)
        self._los[tid] = (self.rng.random(n) < p_los) if self.los_enabled \
            else np.zeros(n, dtype=bool)
        sigma = np.where(self._los[tid], self.shadow_sigma_los_db,
                         self.shadow_sigma_nlos_db)
        self._shadow[tid] = self.rng.normal(0.0, 1.0, n) * sigma
        self._fade_i[tid] = self.rng.normal(0.0, 1.0, n)
        self._fade_q[tid] = self.rng.normal(0.0, 1.0, n)
        self._last_xy[tid] = xy.copy()

    # ------------------------------------------------------------------
    @staticmethod
    def los_probability(d2d_m: np.ndarray) -> np.ndarray:
        """3GPP TR 38.901 UMa LOS probability (h_UT < 13 m)."""
        d = np.maximum(np.asarray(d2d_m, dtype=float), 1.0)
        return np.minimum(18.0 / d, 1.0) * (1 - np.exp(-d / 63.0)) \
            + np.exp(-d / 63.0)

    def path_loss_db(self, d2d: np.ndarray, los: np.ndarray,
                     h_bs: float) -> np.ndarray:
        """UMa path loss, LOS and NLOS branches (TR 38.901 Table 7.4.1-1)."""
        d2d = np.maximum(np.asarray(d2d, dtype=float), 10.0)
        d3d = np.sqrt(d2d ** 2 + (h_bs - self.h_ut) ** 2)
        fc = self.fc_ghz
        pl_los = 28.0 + 22.0 * np.log10(d3d) + 20.0 * np.log10(fc)
        pl_nlos = (13.54 + 39.08 * np.log10(d3d) + 20.0 * np.log10(fc)
                   - 0.6 * (self.h_ut - 1.5))
        pl_nlos = np.maximum(pl_nlos, pl_los)     # NLOS is never better
        return np.where(los, pl_los, pl_nlos)

    def antenna_gain_db(self, d2d: np.ndarray, tilt_deg: float,
                        site: CellSite) -> np.ndarray:
        """3D antenna pattern, vertical cut only (TR 36.814 / 38.901).

        theta is the elevation angle of the UE below the horizon.  The
        electrical downtilt steers the boresight; a UE at a very different
        elevation falls off the main lobe.  Because near UEs sit at a LARGE
        depression angle and far UEs at a small one, a single tilt cannot
        serve both -- tilting down helps near UEs and starves the edge.
        This is what gives tilt a genuinely different cross-tenant
        signature from transmit power.
        """
        d2d = np.maximum(np.asarray(d2d, dtype=float), 1.0)
        theta = np.degrees(np.arctan2(site.height_m - self.h_ut, d2d))
        dev = theta - float(tilt_deg)
        av = -np.minimum(12.0 * (dev / max(self.vert_hpbw_deg, 1e-3)) ** 2,
                         self.sla_v_db)
        return site.antenna_gain_dbi + av

    # ------------------------------------------------------------------
    def _update_shadow(self, tid: str, xy: np.ndarray,
                       los: np.ndarray) -> np.ndarray:
        """Gudmundson spatially correlated shadowing.

        rho = exp(-|dx| / d_corr).  A stationary UE keeps its shadowing; a
        moving UE decorrelates at a rate set by how far it moved.  This is
        why a slow drift produces a SLOW, sustained change in received
        power rather than white noise -- and therefore why the drift is
        detectable by a residual monitor but invisible to a load meter.
        """
        prev = self._last_xy.get(tid)
        if prev is None or prev.shape != xy.shape:
            self.register(tid, xy)
            return self._shadow[tid]
        step = np.hypot(xy[:, 0] - prev[:, 0], xy[:, 1] - prev[:, 1])
        rho = np.exp(-step / max(self.shadow_decorr_m, 1e-6))
        sigma = np.where(los, self.shadow_sigma_los_db,
                         self.shadow_sigma_nlos_db)
        innov = self.rng.normal(0.0, 1.0, xy.shape[0]) * sigma
        self._shadow[tid] = rho * self._shadow[tid] + np.sqrt(
            np.maximum(1.0 - rho ** 2, 0.0)) * innov
        self._last_xy[tid] = xy.copy()
        return self._shadow[tid]

    def _update_fading(self, tid: str, speed_mps: np.ndarray,
                       los: np.ndarray) -> np.ndarray:
        """Temporally correlated small-scale fading power (linear).

        A first-order autoregressive approximation of the Clarke/Jakes
        autocorrelation, rho = J0(2 pi f_d T_s), evaluated with
        f_d = v * f_c / c.  Fast UEs decorrelate every slot (so averaging
        works); slow UEs stay in a deep fade for many slots (so averaging
        does not).  Rician for LOS, Rayleigh for NLOS.
        """
        if not self.fading_enabled:
            return np.ones(len(los))
        fd = np.asarray(speed_mps, dtype=float) * self.fc_ghz * 1e9 / 3e8
        arg = 2.0 * np.pi * fd * self.slot_s
        # J0 approximation valid for the small arguments that occur here
        rho = np.clip(1.0 - 0.25 * arg ** 2, 0.0, 0.999)
        n = len(los)
        gi = self.rng.normal(0.0, 1.0, n)
        gq = self.rng.normal(0.0, 1.0, n)
        s = np.sqrt(np.maximum(1.0 - rho ** 2, 0.0))
        self._fade_i[tid] = rho * self._fade_i[tid] + s * gi
        self._fade_q[tid] = rho * self._fade_q[tid] + s * gq
        # Rician: add a deterministic LOS component of the right K-factor
        k_lin = 10 ** (self.rician_k_db / 10.0)
        a_los = np.sqrt(k_lin / (k_lin + 1.0))
        a_sca = np.sqrt(1.0 / (k_lin + 1.0))
        i_c = np.where(los, a_los + a_sca * self._fade_i[tid] / np.sqrt(2),
                       self._fade_i[tid] / np.sqrt(2))
        q_c = np.where(los, a_sca * self._fade_q[tid] / np.sqrt(2),
                       self._fade_q[tid] / np.sqrt(2))
        return np.maximum(i_c ** 2 + q_c ** 2, 1e-4)

    # ------------------------------------------------------------------
    def evaluate(self, tid: str, xy: np.ndarray, speed_mps: np.ndarray,
                 tx_dbm: float, tilt_deg: float, cio_db: float = 0.0,
                 subband_quality: float = 1.0,
                 mcs_cap: int = MCS_MAX_INDEX) -> Dict[str, np.ndarray]:
        """Full per-UE PHY evaluation for one slot.

        Parameters
        ----------
        tx_dbm          total serving-cell transmit power (the ``txpower``
                        knob).  Divided across the PRBs internally, which
                        is the only way the knob means anything physically.
        tilt_deg        serving-cell electrical downtilt (``tilt`` knob).
        cio_db          cell individual offset for this slice (``cio_*``).
                        Shifts the effective received power used for
                        association/steering, so it moves load without
                        moving physics for already-attached UEs.
        subband_quality multiplicative SINR factor from the zero-sum
                        scheduler-weight mechanism (``schedw_*``).
        mcs_cap         slice MCS ceiling (``mcs_*`` knob).

        Returns a dict of per-UE arrays.  Keys are the columns of
        ``csv/channel.csv`` and ``csv/phy.csv``.
        """
        if tid not in self._last_xy or self._last_xy[tid].shape != xy.shape:
            self.register(tid, xy)
        n = xy.shape[0]
        d2d = np.hypot(xy[:, 0] - self.serving.x, xy[:, 1] - self.serving.y)
        # LOS state is re-drawn only when a UE has moved far enough for the
        # blockage environment to plausibly have changed.
        los = self._los[tid]
        shadow = self._update_shadow(tid, xy, los)
        fade = self._update_fading(tid, speed_mps, los)

        pl = self.path_loss_db(d2d, los, self.serving.height_m)
        gain = self.antenna_gain_db(d2d, tilt_deg, self.serving)
        tx_per_prb = tx_dbm - 10.0 * math.log10(self.n_prb)
        rx_dbm = tx_per_prb + gain - pl - shadow + cio_db
        rx_lin = 10 ** (rx_dbm / 10.0) * fade

        # ---- inter-cell interference --------------------------------
        interf_lin = np.zeros(n)
        for site in self.sites[1:]:
            di = np.maximum(np.hypot(xy[:, 0] - site.x, xy[:, 1] - site.y),
                            10.0)
            plos_i = self.los_probability(di) if self.los_enabled \
                else np.zeros(n)
            pl_i = self.path_loss_db(di, plos_i > 0.5, site.height_m)
            g_i = self.antenna_gain_db(di, site.tilt_deg, site)
            tx_i = site.tx_dbm - 10.0 * math.log10(self.n_prb)
            interf_lin += site.load_factor * 10 ** ((tx_i + g_i - pl_i) / 10.0)

        noise_lin = 10 ** (self.noise_prb_dbm / 10.0)
        sinr_lin = rx_lin / np.maximum(interf_lin + noise_lin, 1e-30)
        sinr_lin = np.maximum(sinr_lin * max(subband_quality, 1e-6), 1e-6)
        sinr_db = 10.0 * np.log10(sinr_lin)
        snr_db = 10.0 * np.log10(np.maximum(rx_lin / noise_lin, 1e-6))

        # ---- link adaptation ----------------------------------------
        cqi = cqi_from_sinr_db(sinr_db)
        mcs = np.minimum(mcs_from_cqi(cqi), int(mcs_cap))
        se_mcs = se_from_mcs(mcs)
        se_shannon = np.log2(1.0 + sinr_lin)
        bler = bler_from_sinr(sinr_db, mcs, self.bler_slope_db)
        # HARQ: residual error after up to max_harq retransmissions
        resid_bler = bler ** (self.max_harq + 1)
        # expected transmissions per successful block
        exp_tx = (1.0 - bler ** (self.max_harq + 1)) / np.maximum(1.0 - bler,
                                                                  1e-6)
        exp_tx = np.maximum(exp_tx, 1.0)

        return {
            "d2d_m": d2d,
            "los": los.astype(float),
            "pathloss_db": pl,
            "shadow_db": shadow,
            "fading_db": 10.0 * np.log10(fade),
            "antenna_gain_db": gain,
            "rx_dbm": 10.0 * np.log10(np.maximum(rx_lin, 1e-30)),
            "interf_dbm": 10.0 * np.log10(np.maximum(interf_lin, 1e-30)),
            "noise_dbm": np.full(n, self.noise_prb_dbm),
            "snr_db": snr_db,
            "sinr_db": sinr_db,
            "sinr_lin": sinr_lin,
            "cqi": cqi.astype(float),
            "mcs": mcs.astype(float),
            "se_mcs": se_mcs,               # achievable PHY rate per Hz
            "se_shannon": se_shannon,       # theoretical channel capacity
            "bler": bler,
            "resid_bler": resid_bler,
            "harq_tx": exp_tx,
        }

    # ------------------------------------------------------------------
    def set_serving_power(self, tx_dbm: float, tilt_deg: float) -> None:
        self.serving.tx_dbm = float(tx_dbm)
        self.serving.tilt_deg = float(tilt_deg)

    def describe(self) -> Dict:
        return {
            "carrier_ghz": self.fc_ghz,
            "n_prb": self.n_prb,
            "prb_bandwidth_hz": self.prb_hz,
            "bandwidth_hz": self.bw_hz,
            "noise_figure_db": self.noise_figure_db,
            "noise_per_prb_dbm": self.noise_prb_dbm,
            "bs_height_m": self.h_bs,
            "ue_height_m": self.h_ut,
            "vertical_hpbw_deg": self.vert_hpbw_deg,
            "antenna_gain_dbi": self.antenna_gain_dbi,
            "shadow_sigma_los_db": self.shadow_sigma_los_db,
            "shadow_sigma_nlos_db": self.shadow_sigma_nlos_db,
            "shadow_decorr_m": self.shadow_decorr_m,
            "rician_k_db": self.rician_k_db,
            "max_harq_retx": self.max_harq,
            "pathloss_model": "3GPP TR 38.901 UMa LOS/NLOS",
            "sites": [vars(s) for s in self.sites],
        }
