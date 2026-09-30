#!/usr/bin/env python3
"""Held-out report with PRE-DECLARED acceptance gates.  [AUTO]

Reads the per-run records of the held-out evaluation and states, for every
acceptance criterion, whether it PASSED, FAILED or is INCONCLUSIVE, with
the numbers behind the verdict.  The thresholds below were fixed before any
held-out result was seen; they are not tuned to the outcome, and nothing in
this script turns a failure into a pass.  It works on partial data, so it
can be run while the evaluation is still in progress (verdicts then say how
many seeds they rest on).

Usage
    python scripts/heldout_report.py --runs results/heldout/runs.jsonl \\
        --manifest results/heldout/manifest.json --out results/heldout
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# ---- thresholds, fixed before the held-out results were seen --------------
ORACLE_GAIN_MIN = 0.030      # oracle - frozen INTACT-RA
CONTROL_GAIN_MIN = 0.030     # best mediated method - all-reject
CONSISTENCY_MIN = 0.80       # share of seeds on which Agentic beats INTACT-RA
LATENCY_MAX_MS = 10.0        # near-RT decision budget (p95)
MIN_SEEDS_FOR_VERDICT = 3

LABEL = {"all-reject": "all-reject", "all-accept": "B0 all-admit",
         "b3": "B3", "intact-ra": "INTACT-RA",
         "intact-ra-cell": "INTACT-RA (cell regime)",
         "intact-ra-pertenant-sweep": "INTACT-RA (per-tenant sweep)",
         "intact-ra-agentic": "INTACT-RA-Agentic", "oracle": "oracle"}


def paired(runs, a, b, key="IF", n_boot=4000, seed=7):
    """Paired a-minus-b over the seeds both methods completed."""
    common = sorted(set(runs.get(a, {})) & set(runs.get(b, {})))
    if not common:
        return None
    d = np.array([runs[a][s][key] - runs[b][s][key] for s in common])
    rng = np.random.default_rng(seed)
    if len(d) > 1:
        boots = np.array([rng.choice(d, len(d), replace=True).mean()
                          for _ in range(n_boot)])
        lo, hi = np.percentile(boots, [2.5, 97.5])
    else:
        lo = hi = float(d[0])
    return {"n": len(d), "mean": float(d.mean()), "lo": float(lo),
            "hi": float(hi), "wins": int((d > 0).sum()), "d": d.tolist()}


def verdict(ok, n, need=MIN_SEEDS_FOR_VERDICT):
    if n < need:
        return "INCONCLUSIVE"
    return "PASS" if ok else "FAIL"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--out", default=".")
    ap.add_argument("--ablations", default=None,
                    help="runs.jsonl of ablations on the same held-out seeds")
    a = ap.parse_args()

    # a plain dict: looking up a method that has not run yet must not
    # silently create an empty entry that then counts as a method
    runs = {}
    for line in open(a.runs):
        r = json.loads(line)
        runs.setdefault(r["m"], {})[r["s"]] = r
    man = json.load(open(a.manifest)) if a.manifest and Path(
        a.manifest).exists() else {}

    # Every NON-paired figure -- the results table, and gates 3, 6a, 6b and
    # 7 -- is computed only over seeds EVERY method has completed.  Averaging
    # each method over its own finished seeds compares different plants:
    # mid-run, a method that has reached an easy seed looks better than one
    # that has not.  Paired gates use their own pairwise intersection.
    present = [m for m in runs if runs[m]]
    common = sorted(set.intersection(*(set(runs[m]) for m in present))) \
        if present else []
    excluded = {m: sorted(set(runs[m]) - set(common)) for m in present}
    excluded = {m: s for m, s in excluded.items() if s}

    def mean(m, k="IF"):
        v = [runs[m][s][k] for s in common if s in runs.get(m, {})
             and isinstance(runs[m][s].get(k), (int, float))]
        return (float(np.mean(v)), len(v)) if v else (float("nan"), 0)

    L = ["# Held-out evaluation report", ""]
    if man:
        L += [f"Scenario `{man.get('scenario')}`, configuration fingerprint "
              f"`{man.get('fingerprint')}`, frozen {man.get('frozen_at')}. "
              f"Held-out seeds {man.get('seeds')}; development seeds "
              f"{man.get('dev_seeds')} were never used here.", ""]
    L += ["## Results", "",
          f"Means over the {len(common)} seed(s) completed by EVERY method: "
          f"{common}." + (f" Excluded for now, because not every method has "
                          f"completed them: {excluded}." if excluded else ""),
          "",
          "| method | seeds | IF | shortfall | C1 | C2 | causal crossings "
          "| writes/ep | p95 latency (ms) |",
          "|---|---|---|---|---|---|---|---|---|"]
    order = ["oracle", "intact-ra-agentic", "intact-ra-cell", "b3",
             "intact-ra-pertenant-sweep", "intact-ra", "all-reject",
             "all-accept"]
    for m in order:
        if not runs.get(m):
            continue
        f, n = mean(m)
        L.append(f"| {LABEL[m]} | {n} | **{f:.4f}** | {mean(m,'short')[0]:.3f} "
                 f"| {mean(m,'c1')[0]:.1f} | {mean(m,'c2')[0]:.1f} "
                 f"| {mean(m,'cx')[0]:.1f} | {mean(m,'wpe')[0]:.2f} "
                 f"| {mean(m,'lat95')[0]:.2f} |")
    L.append("")

    gates = []

    def gate(name, status, text):
        gates.append((name, status, text))

    # 1. dynamic sensitivities matter
    p = paired(runs, "oracle", "intact-ra")
    if p:
        gate("1. Oracle beats frozen INTACT-RA (current sensitivities matter)",
             verdict(p["mean"] >= ORACLE_GAIN_MIN, p["n"]),
             f"{p['mean']:+.4f} over {p['n']} seed(s), 95% CI "
             f"[{p['lo']:+.4f}, {p['hi']:+.4f}]; threshold {ORACLE_GAIN_MIN}")
    else:
        gate("1. Oracle beats frozen INTACT-RA", "INCONCLUSIVE", "no paired seeds yet")

    # 2. headline
    p = paired(runs, "intact-ra-agentic", "intact-ra")
    if p:
        ok = p["lo"] > 0 and p["n"] >= MIN_SEEDS_FOR_VERDICT
        gate("2. INTACT-RA-Agentic beats INTACT-RA (headline)",
             verdict(ok, p["n"]),
             f"{p['mean']:+.4f} over {p['n']} seed(s), 95% CI "
             f"[{p['lo']:+.4f}, {p['hi']:+.4f}]; requires CI entirely above 0")

    # 3. improvement not merely from accepting more actions
    ag_w, ra_w = mean("intact-ra-agentic", "wpe")[0], mean("intact-ra", "wpe")[0]
    b0_if, b0_w = mean("all-accept")[0], mean("all-accept", "wpe")[0]
    if np.isfinite(ag_w) and np.isfinite(ra_w):
        ok = (ag_w <= ra_w + 1e-9) or (np.isfinite(b0_if) and
                                        b0_if < mean("intact-ra-agentic")[0])
        gate("3. Improvement is not merely from accepting more actions",
             verdict(ok, mean("intact-ra-agentic")[1]),
             f"Agentic writes {ag_w:.2f}/epoch vs INTACT-RA {ra_w:.2f}; the "
             f"method that writes most (B0, {b0_w:.2f}/epoch) scores "
             f"{b0_if:.4f}, the lowest of all")

    # 4. held-out consistency
    p = paired(runs, "intact-ra-agentic", "intact-ra")
    if p:
        share = p["wins"] / p["n"]
        gate("4. Held-out consistency (Agentic > INTACT-RA per seed)",
             verdict(share >= CONSISTENCY_MIN, p["n"]),
             f"wins on {p['wins']}/{p['n']} held-out seed(s); need >= "
             f"{CONSISTENCY_MIN:.0%}")

    # 5. real B3 included, and how Agentic compares
    p = paired(runs, "intact-ra-agentic", "b3")
    if p:
        gate("5. Real B3 included; Agentic vs B3 (reported, not a gate)",
             "REPORTED",
             f"{p['mean']:+.4f} over {p['n']} seed(s), 95% CI "
             f"[{p['lo']:+.4f}, {p['hi']:+.4f}], Agentic ahead on "
             f"{p['wins']}/{p['n']}")

    # 6. control necessary, indiscriminate intervention harmful
    ar, n_ar = mean("all-reject")
    best = max((mean(m)[0], m) for m in ("intact-ra-agentic", "b3",
                                         "intact-ra") if runs.get(m))
    if np.isfinite(ar):
        gate("6a. Useful control is necessary (best mediated - all-reject)",
             verdict(best[0] - ar >= CONTROL_GAIN_MIN, n_ar),
             f"{LABEL[best[1]]} {best[0]:.4f} vs all-reject {ar:.4f} "
             f"({best[0]-ar:+.4f}); threshold {CONTROL_GAIN_MIN}")
    if np.isfinite(b0_if):
        c1, c2 = mean("all-accept", "c1")[0], mean("all-accept", "c2")[0]
        gate("6b. Indiscriminate intervention is harmful (B0)",
             verdict(b0_if < ar and (c1 + c2) > 0, mean("all-accept")[1]),
             f"B0 {b0_if:.4f} vs all-reject {ar:.4f}; C1 {c1:.1f}, C2 "
             f"{c2:.1f} violations per run")

    # 7. safety alongside the optimisation metric
    mediated = [m for m in ("intact-ra-agentic", "intact-ra", "b3", "oracle")
                if runs.get(m)]
    viol = sum(mean(m, "c1")[0] + mean(m, "c2")[0] for m in mediated)
    ag_cx, ra_cx = mean("intact-ra-agentic", "cx")[0], mean("intact-ra", "cx")[0]
    gate("7. Safety: mediated methods respect C1/C2; causal crossings reported",
         verdict(viol == 0, min(mean(m)[1] for m in mediated)),
         f"total C1+C2 across mediated methods {viol:.1f}; causal crossings "
         f"per run: Agentic {ag_cx:.1f}, INTACT-RA {ra_cx:.1f}")

    # 8. latency, reported separately from twin compute
    lat = mean("intact-ra-agentic", "lat95")[0]
    tw = mean("intact-ra-agentic", "twin_s_per_call")[0]
    gate("8. Near-RT decision latency (twin compute reported separately)",
         verdict(lat <= LATENCY_MAX_MS, mean("intact-ra-agentic")[1]),
         f"Agentic p95 decision latency {lat:.2f} ms (budget "
         f"{LATENCY_MAX_MS} ms); digital-twin calibration "
         f"{tw*1000:.0f} ms per call, in the slow loop")

    L += ["## Acceptance gates", "",
          "| gate | verdict | evidence |", "|---|---|---|"]
    for name, st, text in gates:
        L.append(f"| {name} | **{st}** | {text} |")
    L.append("")

    core = [g for g in gates if g[0][0] in "1246"]
    if any(g[1] == "FAIL" for g in core):
        head = ("**HEADLINE NOT SUPPORTED.** At least one core gate failed "
                "on the held-out seeds; see the table above.")
    elif any(g[1] == "INCONCLUSIVE" for g in core):
        head = ("**HEADLINE INCONCLUSIVE.** Not enough held-out seeds yet "
                "for every core gate to reach a verdict.")
    else:
        head = ("**HEADLINE SUPPORTED** on the held-out seeds, within the "
                "scope conditions of the scenario.")
    if a.ablations and Path(a.ablations).exists():
        for line in open(a.ablations):
            r = json.loads(line)
            runs.setdefault(r["m"], {})[r["s"]] = r
        abl = sorted({json.loads(l)["m"] for l in open(a.ablations)})
        L += ["## Ablations (same held-out seeds, paired)", "",
              "Each row removes or changes exactly one thing. Differences "
              "are paired over the seeds both methods completed, with "
              "bootstrap 95% intervals.", "",
              "| variant | seeds | IF | vs INTACT-RA-Agentic | vs INTACT-RA |",
              "|---|---|---|---|---|"]
        for m in abl:
            v = [r["IF"] for r in runs[m].values()]
            pa = paired(runs, m, "intact-ra-agentic")
            pr = paired(runs, m, "intact-ra")
            fmt = lambda q: (f"{q['mean']:+.4f} [{q['lo']:+.4f}, {q['hi']:+.4f}], "
                             f"above on {q['wins']}/{q['n']}") if q else "n/a"
            L.append(f"| {m} | {len(v)} | {np.mean(v):.4f} | {fmt(pa)} | {fmt(pr)} |")
        L.append("")
    L += ["## Verdict", "", head, "",
          "Gate 9 (mechanism) is demonstrated by the slope-trajectory and "
          "decision figures, not by this table."]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "HELDOUT_REPORT.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
