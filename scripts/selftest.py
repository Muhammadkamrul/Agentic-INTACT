#!/usr/bin/env python3
"""STEP 0. Run this first, and after any change.

Checks the things that are easy to break silently and hard to notice in a
result: hand-derived gradients, reproducibility, checkpoint fidelity,
telemetry neutrality, the safety invariants, and the physics the whole
benchmark rests on.

Exits non-zero on any failure.
"""
from __future__ import annotations
import argparse, math, sys, tempfile
from pathlib import Path
import numpy as np
from _common import ROOT, Log

FAIL = []
PASSED = []


def check(name, cond, detail=""):
    (PASSED if cond else FAIL).append(name)
    print(f"  [{'pass' if cond else 'FAIL'}] {name}"
          + (f"   {detail}" if detail else ""), flush=True)
    return cond


# ---------------------------------------------------------------------------
def test_gradients():
    """Every backward() must match a central finite difference.

    The networks here are hand-differentiated NumPy. A sign error in one
    backward pass does not crash: it trains to a worse optimum and looks
    like a hyperparameter problem for weeks.
    """
    print("\n-- hand-derived gradients vs central finite differences")
    from intact_agentic.agent.nn import Linear, Sequential, ReLU, Tanh, mlp
    rng = np.random.default_rng(0)

    for name, net, din in (
            ("Linear", Linear(4, 3, rng, "t"), 4),
            ("Sequential(Linear,ReLU,Linear)",
             Sequential(Linear(5, 6, rng, "a"), ReLU(),
                        Linear(6, 2, rng, "b")), 5),
            ("mlp(6,8,8,3) tanh", mlp([6, 8, 8, 3], rng, "m", act=Tanh), 6)):
        x = rng.normal(size=(7, din))
        y = net.forward(x)
        gy = rng.normal(size=y.shape)
        net.zero_grad()
        net.backward(gy)
        ok, worst = True, 0.0
        for prm in net.parameters():
            flat = prm.value.reshape(-1)
            gflat = prm.grad.reshape(-1)
            idx = rng.choice(flat.size, min(6, flat.size), replace=False)
            for i in idx:
                old = flat[i]
                eps = 1e-6 * max(abs(old), 1.0)
                flat[i] = old + eps
                lp = float(np.sum(net.forward(x) * gy))
                flat[i] = old - eps
                lm = float(np.sum(net.forward(x) * gy))
                flat[i] = old
                num = (lp - lm) / (2 * eps)
                den = max(abs(num), abs(gflat[i]), 1e-6)
                err = abs(num - gflat[i]) / den
                worst = max(worst, err)
                if err > 2e-3:
                    ok = False
        check(f"gradient: {name}", ok, f"worst relative error {worst:.2e}")


def test_gcn_gradients():
    print("\n-- GCN encoder gradients")
    from intact_agentic.agent.gcn import BipartiteGraph, GCNEncoder
    rng = np.random.default_rng(1)
    ns, nk = 9, 3
    memb = (rng.random((ns, nk)) < 0.5).astype(float)
    memb[np.arange(ns), rng.integers(0, nk, ns)] = 1.0   # no orphan sessions
    g = BipartiteGraph(rng.normal(size=(ns, 10)), rng.normal(size=(nk, 10)),
                       memb, [f"T{i}" for i in range(nk)])
    enc = GCNEncoder(10, 10, 8, 6, rng, layers=3, name="g")
    pooled, _ = enc.forward(g)
    gp = rng.normal(size=pooled.shape)
    enc.zero_grad()
    try:
        enc.backward(gp)
    except Exception as e:
        check("GCN backward runs", False, str(e))
        return
    ok, worst = True, 0.0
    for prm in enc.parameters():
        flat, gflat = prm.value.reshape(-1), prm.grad.reshape(-1)
        for i in rng.choice(flat.size, min(4, flat.size), replace=False):
            old = flat[i]
            eps = 1e-6 * max(abs(old), 1.0)
            flat[i] = old + eps
            lp = float(np.sum(enc.forward(g)[0] * gp))
            flat[i] = old - eps
            lm = float(np.sum(enc.forward(g)[0] * gp))
            flat[i] = old
            num = (lp - lm) / (2 * eps)
            den = max(abs(num), abs(gflat[i]), 1e-6)
            worst = max(worst, abs(num - gflat[i]) / den)
            if abs(num - gflat[i]) / den > 5e-3:
                ok = False
    check("gradient: GCNEncoder", ok, f"worst relative error {worst:.2e}")


# ---------------------------------------------------------------------------
def test_determinism(cfg, reg, prior):
    print("\n-- reproducibility")
    from intact_agentic import methods as M
    from intact_agentic.experiment import Experiment
    out = []
    for _ in range(2):
        with tempfile.TemporaryDirectory() as td:
            ex = Experiment(cfg, M.get("intact-ra-agentic"), reg, td, seed=5,
                            prior=prior, telemetry=False, log=lambda *a: None)
            out.append(ex.run())
    check("same seed gives identical wIF", out[0]["wIF"] == out[1]["wIF"],
          f"{out[0]['wIF']:.10f} vs {out[1]['wIF']:.10f}")
    check("same seed gives identical write count",
          out[0]["writes_total"] == out[1]["writes_total"])


def test_telemetry_neutral(cfg, reg, prior):
    """Recording must not change the numbers. If it does, the telemetry is
    consuming random numbers and every logged run is a different
    experiment from the unlogged one."""
    print("\n-- telemetry neutrality")
    from intact_agentic import methods as M
    from intact_agentic.experiment import Experiment
    res = {}
    for tel in (False, True):
        with tempfile.TemporaryDirectory() as td:
            ex = Experiment(cfg, M.get("intact-ra"), reg, td, seed=6,
                            prior=prior, telemetry=tel, log=lambda *a: None)
            res[tel] = ex.run()
    check("telemetry on/off gives identical wIF",
          abs(res[True]["wIF"] - res[False]["wIF"]) < 1e-12,
          f"{res[False]['wIF']:.10f} vs {res[True]['wIF']:.10f}")


def test_checkpoint(cfg, reg, prior):
    print("\n-- checkpoint and resume")
    from intact_agentic import methods as M
    from intact_agentic.experiment import Experiment
    n = int(cfg["run"]["epochs"])
    with tempfile.TemporaryDirectory() as td:
        a = Experiment(cfg, M.get("intact-ra"), reg, Path(td) / "a", seed=8,
                       prior=prior, telemetry=False, log=lambda *x: None)
        full = a.run()
        b = Experiment(cfg, M.get("intact-ra"), reg, Path(td) / "b", seed=8,
                       prior=prior, telemetry=False, log=lambda *x: None)
        b.run(until=n // 2)
        b.save_checkpoint()
        c = Experiment(cfg, M.get("intact-ra"), reg, Path(td) / "b", seed=8,
                       prior=prior, telemetry=False, log=lambda *x: None)
        check("checkpoint loads", c.load_checkpoint())
        resumed = c.run()
    check("resumed run reaches the same epoch count",
          resumed["epochs_run"] == full["epochs_run"])
    check("resumed run gives the same wIF",
          abs(resumed["wIF"] - full["wIF"]) < 1e-9,
          f"{full['wIF']:.8f} vs {resumed['wIF']:.8f}")


def test_checkpoint_twin():
    """Resume INTACT-RA-Agentic WITH the digital twin and Kalman tracker.

    test_checkpoint runs on the default scenario, where neither the twin nor
    the Kalman tracker is used, so it could not see that a checkpoint once
    failed to restore the Kalman posteriors, the twin's random stream and
    the supervisor.  A resumed agent then silently continued with its memory
    wiped.  This runs the headline configuration straight through and again
    with a stop, checkpoint, reload and continue, and requires every epoch
    to match exactly.
    """
    print("\n-- checkpoint and resume WITH the digital twin (S16)")
    import tempfile
    from intact_agentic import methods as M
    from intact_agentic.experiment import Experiment
    from intact_agentic.config import load_config, build_registry
    from intact_agentic.arbiter.sensitivity import StaticSensitivity
    sc = "S16_high_ceiling"
    cache = ROOT / "artifacts" / f"prior_{sc}.json"
    if not cache.exists():
        check("S16 offline table available for the twin resume test", False,
              f"missing {cache}; run any S16 experiment once to build it")
        return

    def cfg():
        c = load_config(scenario=sc)
        c["run"].update({"epochs": 60, "burn_in_epochs": 20, "log_every": 0,
                         "checkpoint_every": 0, "counterfactual": False})
        c["agent"]["twin"].update({"every_burn_in": 2})
        return c
    c0 = cfg()
    reg = build_registry(c0)
    prior = StaticSensitivity.load(c0, cache)
    with tempfile.TemporaryDirectory() as td:
        a = Experiment(cfg(), M.get("intact-ra-agentic"), reg, td + "/a",
                       seed=7, prior=prior, telemetry=False,
                       log=lambda *z: None)
        sa = a.run()
        b = Experiment(cfg(), M.get("intact-ra-agentic"), reg, td + "/b",
                       seed=7, prior=prior, telemetry=False,
                       log=lambda *z: None)
        for _ in range(41):
            b.step_epoch()
        b.save_checkpoint()
        b2 = Experiment(cfg(), M.get("intact-ra-agentic"), reg, td + "/b",
                        seed=7, prior=prior, telemetry=False,
                        log=lambda *z: None)
        loaded = b2.load_checkpoint()
        sb = b2.run()
    check("twin run: checkpoint loads", loaded)
    check("twin run: the twin was actually exercised", a.twin is not None
          and a.twin.calls > 0, f"{a.twin.calls if a.twin else 0} twin calls")
    ra, rb = a.metrics.records, b2.metrics.records
    check("twin run: resumed decisions identical every epoch",
          [r.n_writes for r in ra] == [r.n_writes for r in rb])
    check("twin run: resumed margins identical every epoch",
          all(x.g_after == y.g_after for x, y in zip(ra, rb)))
    check("twin run: resumed IF identical", sa["IF"] == sb["IF"],
          f"{sa['IF']:.10f} vs {sb['IF']:.10f}")


def test_safety_invariants(cfg, reg, prior):
    print("\n-- arbiter invariants")
    from intact_agentic import methods as M
    from intact_agentic.experiment import Experiment
    with tempfile.TemporaryDirectory() as td:
        ex = Experiment(cfg, M.get("intact-ra"), reg, td, seed=9,
                        prior=prior, telemetry=False, log=lambda *a: None)
        s = ex.run()
    check("no C1 violation (two writes to one parameter in an epoch)",
          s["c1_violations"] == 0, f"{s['c1_violations']}")
    check("no C2 violation (tenant envelope exceeded)",
          s["c2_violations"] == 0, f"{s['c2_violations']}")
    lat_ex = float(np.percentile([r.latency_ms for r in ex.metrics.records],
                                 99))
    # Exhaustive enumeration is EXPECTED to overrun the near-real-time
    # budget as the claim set grows -- that is the problem the learned
    # top-K proposer exists to solve, not a defect.  What must hold is
    # that the proposed method fits.
    with tempfile.TemporaryDirectory() as td:
        ag = Experiment(cfg, M.get("intact-ra-agentic"), reg, td, seed=9,
                        prior=prior, telemetry=False, log=lambda *a: None)
        ag.run()
    # Two DIFFERENT properties, reported separately.  Before the PPO
    # proposer is trained it falls back to scoring the whole candidate list,
    # which on a large claim set overruns the budget; after warm-up it
    # returns a short list.  A single p99 over all epochs mixes the two, and
    # with a 40-epoch test and a 25-epoch warm-up it measures mostly cold
    # start while claiming to measure the operating regime.
    wu = int(cfg.get("agent", {}).get("policy", {}).get("warmup_epochs", 25))
    recs = ag.metrics.records
    warm = [r.latency_ms for r in recs if r.epoch >= wu]
    cold = [r.latency_ms for r in recs if r.epoch < wu]
    lat_ag = float(np.percentile(warm, 99)) if warm else float("nan")
    lat_cold = float(np.percentile(cold, 99)) if cold else float("nan")
    print(f"         (exhaustive enumeration p99 = {lat_ex:.2f} ms over "
          f"{ex.metrics.summary()['candidates_mean']:.0f} candidates -- "
          f"this is what the top-K proposer is for)")
    check("after warm-up, the proposed method's decision fits the near-RT "
          "budget", lat_ag < 10.0,
          f"agentic p99 {lat_ag:.2f} ms over epochs >= {wu} (need > 0 such "
          f"epochs: --epochs must exceed {wu}) vs exhaustive {lat_ex:.2f} ms")
    if not warm:
        check("the latency test ran past the proposer's warm-up", False,
              f"--epochs {len(recs)} <= warm-up {wu}")
    # cold start: measured and shown, never silently dropped
    tag = "WARN" if lat_cold >= 10.0 else "info"
    print(f"  [{tag}] cold-start decision latency (proposer warm-up, epochs "
          f"< {wu}): p99 {lat_cold:.2f} ms"
          + ("  -- exceeds the 10 ms near-RT budget. Known limitation: until "
             "the proposer is trained it scores the full candidate list. Set "
             "agent.policy.cold_start_max_candidates to bound it, or keep "
             "warm-up inside an unscored burn-in as S16 does."
             if lat_cold >= 10.0 else ""))
    for r in ex.metrics.records:
        if r.n_writes == 0 and r.candidates > 0:
            break
    check("a no-action portfolio is always available",
          any(r.n_writes == 0 for r in ex.metrics.records))


def test_unknown_is_not_zero(cfg, reg, prior):
    """The documented failure mode, asserted as a test."""
    print("\n-- missing table entries")
    from intact_agentic.arbiter.sensitivity import (OnlineSensitivity,
                                                    StaticSensitivity)
    st = StaticSensitivity(cfg, {})
    check("StaticSensitivity: missing entry reports slope 0 AND sigma 0 "
          "(this is the failure mode, not a bug)",
          st.get("L1C0", "txpower", "i1") == 0.0
          and st.sigma("L1C0", "txpower", "i1") == 0.0)
    on = OnlineSensitivity(cfg, StaticSensitivity(cfg, {}),
                           params=["txpower"], intents=["i1"])
    check("OnlineSensitivity: missing entry reports full unknown sigma",
          on.sigma("L1C0", "txpower", "i1") >= on.unknown_sigma * 0.999,
          f"sigma = {on.sigma('L1C0', 'txpower', 'i1'):.4f}")


def test_physics(cfg):
    print("\n-- RAN physics sanity")
    from intact_agentic.ran.simulator import RealisticRAN
    from intact_agentic.config import epoch_slots
    pre, post = epoch_slots(cfg)
    ran = RealisticRAN(cfg)
    ran.step(48, record=False)
    k = ran.step(pre + post, record=False)
    ts = sorted(ran.slices)
    ok = True
    for t in ts:
        r = k[t]
        if not (0.0 <= r["bler"] <= 1.0):
            ok = False
        if r["throughput_mbps"] > r["shannon_capacity_mbps"] * 1.05 + 1e-6:
            ok = False
        if r["delay_ms"] < cfg["ran"]["base_delay_ms"] - 1e-9:
            ok = False
    check("KPMs are physically admissible "
          "(0<=BLER<=1, throughput<=Shannon, delay>=floor)", ok)
    util = k["_cell"]["prb_util_pct"]
    check("the cell is contended but not saturated", 45.0 <= util <= 96.0,
          f"PRB utilisation {util:.1f}%")
    sinrs = [k[t]["sinr_db"] for t in ts]
    check("tenants are radio-distinguishable (>=4 dB SINR spread)",
          max(sinrs) - min(sinrs) >= 4.0,
          f"spread {max(sinrs)-min(sinrs):.1f} dB")

    # Reconfiguration must actually cost something, or "accept everything"
    # is free and the benchmark is degenerate.  The comparison has to be
    # made INSIDE the transient window with a control change that is
    # negligible in steady state, otherwise a better operating point masks
    # the transient and the test silently passes on the wrong evidence.
    slots = int((cfg["ran"].get("actuation", {}) or {}).get(
        "reconfig_slots", 0))
    if slots <= 0:
        check("reconfiguration transient is configured", False,
              "ran.actuation.reconfig_slots is 0: all-accept writes for free")
    else:
        def settle():
            r = RealisticRAN(cfg)
            r.reset(4242)
            r.step(64, record=False)
            return r
        ctl = settle().step(slots, record=False)["_cell"]["delivered_mbps"]
        rw = settle()
        # +0.01 dB is far below anything that changes the link budget, so
        # any difference measured is the reconfiguration transient alone
        rw.apply("txpower", rw.current_controls()["txpower"] + 0.01,
                 scope="cell")
        hit = rw.step(slots, record=False)["_cell"]["delivered_mbps"]
        check("a cell-scope write costs throughput during the transient",
              hit < ctl * 0.995,
              f"{ctl:.2f} -> {hit:.2f} Mb/s over {slots} slots "
              f"({100*(hit-ctl)/max(ctl,1e-9):+.1f}%)")

    # and the cost must scale with the number of writes, or two writes in
    # one epoch are as cheap as one and churn is unpenalised
    if slots > 0:
        r4 = RealisticRAN(cfg)
        r4.reset(4242)
        r4.step(64, record=False)
        c4 = r4.current_controls()
        for prm in ("quota_T1", "quota_T2", "quota_T3"):
            if prm in c4:
                r4.apply(prm, c4[prm] + 0.01, scope="slice")
        many = r4.step(slots, record=False)["_cell"]["delivered_mbps"]
        check("more writes cost more than one write", many < ctl,
              f"{ctl:.2f} -> {many:.2f} Mb/s")


def test_proposal_stream(cfg, reg):
    print("\n-- workload identity")
    from intact_agentic.ran.simulator import RealisticRAN
    from intact_agentic.xapps import build_xapps, collect_proposals
    from intact_agentic.config import epoch_slots
    pre, post = epoch_slots(cfg)
    ran = RealisticRAN(cfg)
    x = build_xapps(cfg)
    ran.step(32, record=False)
    k = ran.step(pre + post, record=False)
    c = ran.current_controls()
    p1 = collect_proposals(x, reg.claims, k, c, 3)
    p2 = collect_proposals(x, reg.claims, k, c, 3)
    check("polling the xApps twice in one epoch gives the same proposals",
          p1 == p2, f"{len(p1)} claims proposing")
    seen = set()
    for ep in range(30):
        k = ran.step(pre + post, record=False)
        seen |= set(collect_proposals(x, reg.claims, k,
                                      ran.current_controls(), ep))
    check("every claim proposes at least once in 30 epochs",
          len(seen) == len(reg.claims),
          f"{len(seen)}/{len(reg.claims)}: missing "
          f"{sorted(set(reg.claims) - seen)}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", default="S1_edge_drift")
    ap.add_argument("--epochs", type=int, default=80,
                    help="must exceed agent.policy.warmup_epochs (40) so the "
                         "latency check reaches the operating regime")
    ap.add_argument("--fast", action="store_true",
                    help="gradients and physics only")
    a = ap.parse_args()

    from intact_agentic.config import load_config, build_registry
    from intact_agentic.experiment import build_prior

    print("=" * 68)
    print("INTACT-RA-Agentic self-test")
    print("=" * 68)
    test_gradients()
    test_gcn_gradients()

    cfg = load_config(scenario=a.scenario)
    cfg["run"]["epochs"] = int(a.epochs)
    cfg["run"]["log_every"] = 0
    cfg["run"]["checkpoint_every"] = 0
    reg = build_registry(cfg)
    test_physics(cfg)
    test_proposal_stream(cfg, reg)

    if not a.fast:
        prior = build_prior(cfg, reg, log=lambda *x: None,
                            cache=ROOT / "artifacts" / "sensitivity_prior.json")
        test_unknown_is_not_zero(cfg, reg, prior)
        test_determinism(cfg, reg, prior)
        test_telemetry_neutral(cfg, reg, prior)
        test_checkpoint(cfg, reg, prior)
        test_checkpoint_twin()
        test_safety_invariants(cfg, reg, prior)

    print("\n" + "=" * 68)
    print(f"{len(PASSED)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILURES:")
        for f in FAIL:
            print(f"  - {f}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
