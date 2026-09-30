"""
intact_agentic/config.py
========================
Configuration loading, merging, dotted overrides, and construction of the
typed Registry from YAML.

THE ONE RULE THAT MATTERS
-------------------------
A scenario file is merged ON TOP OF ``configs/base.yaml``.  The merge is
recursive for dicts, but a scenario may declare

    _replace_sections: [ran.slices, ran.initial_controls]

to force whole-section REPLACEMENT.  Without this, base parameters leak
into a scenario (this was a real, expensive bug in the predecessor
codebase: a base ``slices`` block with tenants T1..T4 silently survived
into an 8-tenant scenario).

Dotted overrides (``--set v2.foo=3``) are applied last and are CHECKED
against the merged config: a key that does not already exist is rejected
unless it appears in ``CREATABLE_KEYS``.  A silently-created key produces
an "ablation" that changes nothing, which is worse than a crash.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

from .types import (Claim, Direction, Intent, Kind, Registry, Scope, Tenant)

# Keys the CLI is allowed to create even if base.yaml does not declare them.
CREATABLE_KEYS = {
    "agent.enabled", "agent.model_dir", "method", "run.n_epochs",
    "run.seed", "ran.seed", "report.enabled",
}


# ---------------------------------------------------------------------------
def _deep_merge(base: Dict, over: Dict, replace: Sequence[str] = (),
                prefix: str = "") -> Dict:
    out = dict(base)
    for k, v in over.items():
        if k.startswith("_"):
            continue
        path = f"{prefix}{k}"
        if path in replace:
            out[k] = copy.deepcopy(v)
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v, replace, prefix=f"{path}.")
        else:
            out[k] = copy.deepcopy(v)
    return out


def _coerce(text: str) -> Any:
    low = str(text).strip()
    if low.lower() in ("true", "yes"):
        return True
    if low.lower() in ("false", "no"):
        return False
    if low.lower() in ("none", "null"):
        return None
    try:
        return int(low)
    except ValueError:
        pass
    try:
        return float(low)
    except ValueError:
        pass
    if low.startswith("[") or low.startswith("{"):
        try:
            return json.loads(low)
        except Exception:
            return low
    return low


def set_dotted(cfg: Dict, dotted: str, value: Any, *, create: bool = False
               ) -> None:
    parts = dotted.split(".")
    cur = cfg
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            if not create:
                raise KeyError(f"override path {dotted!r} does not exist "
                               f"(stopped at {p!r})")
            cur[p] = {}
        cur = cur[p]
    if parts[-1] not in cur and not create:
        raise KeyError(f"override key {dotted!r} does not exist")
    cur[parts[-1]] = value


def get_dotted(cfg: Dict, dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for p in dotted.split("."):
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return default
    return cur


def apply_overrides(cfg: Dict, overrides: Dict[str, Any]) -> List[str]:
    """Apply dotted overrides in place.  Returns the list of applied keys."""
    applied = []
    for dotted, value in overrides.items():
        create = dotted in CREATABLE_KEYS
        set_dotted(cfg, dotted, value, create=create)
        applied.append(dotted)
    return applied


def parse_cli_overrides(pairs: Iterable[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for item in pairs or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = _coerce(v)
    return out


# ---------------------------------------------------------------------------
def load_config(scenario: Optional[str] = None,
                base: Optional[str] = None,
                overrides: Optional[Dict[str, Any]] = None) -> Dict:
    """Load base.yaml, merge a scenario on top, then apply overrides."""
    here = Path(__file__).resolve().parents[1]
    base_path = Path(base) if base else here / "configs" / "base.yaml"
    with open(base_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    cfg["_base_path"] = str(base_path)

    if scenario:
        sc_path = Path(scenario)
        if not sc_path.exists():
            alt = here / "configs" / "scenarios" / scenario
            if alt.exists():
                sc_path = alt
            elif (here / "configs" / "scenarios" /
                  f"{scenario}.yaml").exists():
                sc_path = here / "configs" / "scenarios" / f"{scenario}.yaml"
            else:
                raise FileNotFoundError(f"scenario not found: {scenario}")
        with open(sc_path, "r", encoding="utf-8") as fh:
            sc = yaml.safe_load(fh) or {}
        replace = tuple(sc.get("_replace_sections", ()) or ())
        cfg = _deep_merge(cfg, sc, replace)
        cfg["_scenario_path"] = str(sc_path)
        cfg["_scenario_name"] = sc_path.stem

    if overrides:
        apply_overrides(cfg, overrides)
    validate_config(cfg)
    return cfg


def config_fingerprint(cfg: Dict) -> str:
    """Stable hash of everything that can change a numerical result."""
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    blob = json.dumps(clean, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


# ---------------------------------------------------------------------------
def validate_config(cfg: Dict) -> None:
    errs: List[str] = []
    for section in ("ran", "tenants", "intents", "claims", "xapps",
                    "arbiter", "run"):
        if section not in cfg:
            errs.append(f"missing top-level section {section!r}")
    if errs:
        raise ValueError("configuration invalid:\n  " + "\n  ".join(errs))

    tids = {t["tid"] for t in cfg["tenants"]}
    slices = set((cfg["ran"].get("slices") or {}).keys())
    for t in cfg["tenants"]:
        if t.get("is_host"):
            continue
        if t["tid"] not in slices:
            errs.append(f"tenant {t['tid']} has no ran.slices entry")
    for it in cfg["intents"]:
        if it["tenant"] not in tids:
            errs.append(f"intent {it['iid']} owned by unknown tenant "
                        f"{it['tenant']}")
    xnames = {x["name"] for x in cfg["xapps"]}
    for c in cfg["claims"]:
        if c["xapp"] not in xnames:
            errs.append(f"claim {c['jid']} names unknown xApp {c['xapp']}")
        if c["tenant"] not in tids:
            errs.append(f"claim {c['jid']} owned by unknown tenant "
                        f"{c['tenant']}")
        lo, hi = c["domain"]
        if hi <= lo:
            errs.append(f"claim {c['jid']} has empty domain {c['domain']}")
        if c["step"] <= 0:
            errs.append(f"claim {c['jid']} has non-positive step")
    init = cfg["ran"].get("initial_controls", {})
    for c in cfg["claims"]:
        if c["param"] not in init:
            errs.append(f"claim {c['jid']} writes {c['param']}, which has no "
                        f"entry in ran.initial_controls")
    if errs:
        raise ValueError("configuration invalid:\n  " + "\n  ".join(errs))


# ---------------------------------------------------------------------------
def build_registry(cfg: Dict) -> Registry:
    tenants = {}
    for t in cfg["tenants"]:
        tenants[t["tid"]] = Tenant(
            tid=t["tid"], omega=float(t.get("omega", 1.0)),
            rho_min=float(t.get("rho_min", 0.8)),
            envelope={k: float(v) for k, v in (t.get("envelope") or {}).items()},
            is_host=bool(t.get("is_host", False)),
            arrives_at_epoch=int(t.get("arrives_at_epoch", 0)))

    intents = {}
    for it in cfg["intents"]:
        intents[it["iid"]] = Intent(
            iid=it["iid"], tenant=it["tenant"], kpi=it["kpi"],
            target=float(it["target"]),
            direction=Direction(it["direction"]),
            pi_class=float(it.get("pi_class", 1.0)),
            epsilon=float(it.get("epsilon", cfg["arbiter"].get("epsilon", 0.02))),
            clip=float(it.get("clip", cfg["arbiter"].get("margin_clip", 1.5))),
            weight=float(it.get("weight", 1.0)),
            arrives_at_epoch=int(it.get("arrives_at_epoch", 0)))

    default_step_frac = float(cfg["arbiter"].get("max_step_frac", 0.25))
    claims = {}
    for c in cfg["claims"]:
        claims[c["jid"]] = Claim(
            jid=c["jid"], xapp=c["xapp"], tenant=c["tenant"],
            param=c["param"], scope=Scope(c["scope"]), kind=Kind(c["kind"]),
            domain=(float(c["domain"][0]), float(c["domain"][1])),
            step=float(c["step"]),
            resource=c.get("resource"),
            d_bar=float(c.get("d_bar", c["domain"][1])),
            max_step_frac=float(c.get("max_step_frac", default_step_frac)),
            r_j=float(c.get("r_j", 0.5)),
            arrives_at_epoch=int(c.get("arrives_at_epoch", 0)))
    return Registry(tenants=tenants, intents=intents, claims=claims)


# ---------------------------------------------------------------------------
def epoch_slots(cfg: Dict) -> Tuple[int, int]:
    """(pre_slots, post_slots).  An epoch has two halves:

    PRE   observe the state the decision is made FROM  -> g_before
    POST  observe the consequence of the writes        -> g_after

    The residual that grades the sensitivity model is g_after - predicted,
    so both halves are load-bearing and must not be collapsed.
    """
    ran = cfg["ran"]
    spe = int(ran.get("slots_per_epoch", 8))
    return int(ran.get("pre_slots", spe)), int(ran.get("post_slots", spe))


def scenario_name(cfg: Dict) -> str:
    return str(cfg.get("_scenario_name", "base"))


def dump_config(cfg: Dict, path: os.PathLike) -> None:
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    clean["_provenance"] = {k: v for k, v in cfg.items() if k.startswith("_")}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(clean, fh, indent=2, sort_keys=True, default=str)
