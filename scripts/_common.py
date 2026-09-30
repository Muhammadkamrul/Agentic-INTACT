"""Shared plumbing for every script: paths, logging, CLI conventions."""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class Log:
    def __init__(self, path=None, quiet=False):
        self.quiet = quiet
        self.fh = None
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.fh = open(path, "a", encoding="utf-8")
        self.t0 = time.time()

    def __call__(self, *a):
        msg = " ".join(str(x) for x in a)
        if not self.quiet:
            print(msg, flush=True)
        if self.fh:
            self.fh.write(f"[{time.time()-self.t0:8.1f}s] {msg}\n")
            self.fh.flush()

    def close(self):
        if self.fh:
            self.fh.close()


def common_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--scenario", default="S1_edge_drift",
                   help="scenario name in configs/scenarios/ or a path")
    p.add_argument("--base", default=str(ROOT / "configs" / "base.yaml"))
    p.add_argument("--out", default=None, help="results directory")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="dotted config overrides, e.g. run.epochs=200")
    p.add_argument("--quiet", action="store_true")
    return p


def load(args):
    from intact_agentic.config import load_config, parse_cli_overrides
    return load_config(scenario=args.scenario, base=args.base,
                       overrides=parse_cli_overrides(args.set))


def results_dir(args, cfg, sub="") -> Path:
    from intact_agentic.config import scenario_name
    root = Path(args.out) if args.out else \
        ROOT / (cfg.get("run", {}) or {}).get("results_root", "results")
    d = root / scenario_name(cfg) / sub if sub else root / scenario_name(cfg)
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=str)


def write_text(path, text):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
