"""Schema and consistency checks for the YAML configuration.

Every problem found is collected and reported together, so a user editing the
configuration sees all mistakes at once instead of a failure deep inside the
simulation. Checks cover structure (required fields, known names), value
ranges (probabilities in [0, 1], positive rates and costs) and cross-references
(parameters named by mitigations, scenarios and outcome tables exist; the
dependency graph is acyclic; forced escalation targets are bridge neighbours).
"""

from __future__ import annotations

from graphlib import CycleError, TopologicalSorter
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from .config import ModelConfig

BASES = {"assumption", "order_of_magnitude", "engineering_convention", "derived", "synthetic_calibrated"}
REQUIRED_PARAM_FIELDS = ("definition", "unit", "category", "distribution", "basis", "rationale")
EFFECT_OPS = {"reduce", "increase", "close_gap"}
OVERRIDE_OPS = {"set", "multiply"}
ARCHITECTURE_KEYS = {
    "electric_firewater_pumps", "diesel_firewater_pumps", "gas_detectors_per_zone", "gas_detection_voting_k",
    "flame_detectors_per_zone", "flame_detection_voting_k", "esd_valves_in_series", "generators_installed",
    "generators_required", "compressors_installed",
}  # fmt: skip
FORCED_TYPES = {"release", "compressor", "power_loss", "pipeline", "weather", "spurious_trip", "collision"}
SIZES = {"small", "medium", "large"}
SOURCES_NEEDING_DRIVER = {"release", "compressor", "power_loss", "pipeline", "spurious_trip", "weather", "collision"}


class ConfigError(ValueError):
    """Raised when the configuration fails validation; lists every problem found."""

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("invalid configuration:\n  - " + "\n  - ".join(problems))


def _is_probability(spec: dict) -> bool:
    unit = str(spec.get("unit", "")).lower()
    return unit.startswith("probability") or unit.startswith("fraction")


def _check_parameters(cfg, errs: list[str]):
    for name, p in cfg.registry.params.items():
        spec = p.spec
        for f in REQUIRED_PARAM_FIELDS:
            if not spec.get(f):
                errs.append(f"parameter {name!r}: missing field {f!r}")
        if spec.get("basis") and spec["basis"] not in BASES:
            errs.append(f"parameter {name!r}: basis {spec['basis']!r} not in {sorted(BASES)}")
        lo, hi = float(p.dist.ppf(1e-9)), float(p.dist.ppf(1 - 1e-9))
        if not (np.isfinite(lo) and np.isfinite(hi)):
            errs.append(f"parameter {name!r}: distribution has non-finite support")
            continue
        if lo < 0:
            errs.append(f"parameter {name!r}: distribution allows negative values (min {lo:.3g})")
        if _is_probability(spec) and hi > 1:
            errs.append(f"parameter {name!r}: probability/fraction can exceed 1 (max {hi:.3g})")
    reg = cfg.registry.params
    if "p_large_release" in reg and "p_medium_release" in reg:
        top = float(reg["p_large_release"].dist.ppf(1 - 1e-9)) + float(reg["p_medium_release"].dist.ppf(1 - 1e-9))
        if top > 1:
            errs.append("p_large_release + p_medium_release can exceed 1")


def _check_asset(cfg, errs: list[str]) -> set[str]:
    a = cfg.asset
    plats = a.get("platforms", {})
    codes = set(plats)
    for c, p in plats.items():
        for f in ("name", "replacement_value_usd", "production_bopd", "release_equipment_factor", "p_power_loss_given_fire",
                  "gas_release_fraction"):  # fmt: skip
            if f not in p:
                errs.append(f"platform {c}: missing {f!r}")
        for f in ("p_power_loss_given_fire", "gas_release_fraction"):
            if f in p and not 0 <= float(p[f]) <= 1:
                errs.append(f"platform {c}: {f} must be in [0, 1]")
        for f in ("replacement_value_usd", "production_bopd", "release_equipment_factor"):
            if f in p and float(p[f]) < 0:
                errs.append(f"platform {c}: {f} must be non-negative")
    if not any(float(p.get("production_bopd", 0)) > 0 for p in plats.values()):
        errs.append("asset: at least one platform must produce")
    for b in a.get("bridges", []):
        if len(b) != 2 or not set(b) <= codes or b[0] == b[1]:
            errs.append(f"bridge {b}: must join two different known platforms")
    ts = TopologicalSorter({c: set() for c in codes})
    for d in a.get("dependencies", []):
        if d.get("platform") not in codes or d.get("provider") not in codes:
            errs.append(f"dependency {d}: unknown platform")
            continue
        if not 0 <= float(d.get("impact", -1)) <= 1:
            errs.append(f"dependency {d['platform']}<-{d['provider']}: impact must be in [0, 1]")
        ts.add(d["platform"], d["provider"])
    try:
        tuple(ts.static_order())
    except CycleError as e:
        errs.append(f"dependency graph has a cycle: {e.args[1]}")
    prot = a.get("protection", {})
    for k in ARCHITECTURE_KEYS:
        if k not in prot:
            errs.append(f"protection: missing {k!r}")
    for n_key, k_key in (("gas_detectors_per_zone", "gas_detection_voting_k"), ("flame_detectors_per_zone", "flame_detection_voting_k"),
                         ("generators_installed", "generators_required")):  # fmt: skip
        if n_key in prot and k_key in prot and not 1 <= int(prot[k_key]) <= int(prot[n_key]):
            errs.append(f"protection: need 1 <= {k_key} <= {n_key}")
    if int(prot.get("electric_firewater_pumps", 0)) + int(prot.get("diesel_firewater_pumps", 0)) < 1:
        errs.append("protection: at least one firewater pump is required")
    return codes


def _neighbours(cfg) -> dict[str, set[str]]:
    nb: dict[str, set[str]] = {}
    for a, b in cfg.asset.get("bridges", []):
        nb.setdefault(a, set()).add(b)
        nb.setdefault(b, set()).add(a)
    return nb


def _check_aleatory(cfg, codes: set[str], errs: list[str]):
    from .event_tree.hydrocarbon_release import OUTCOMES

    al, names = cfg.aleatory, set(cfg.registry.names)
    outcomes = al.get("outcomes", {})
    for o in OUTCOMES:
        if o not in outcomes:
            errs.append(f"aleatory.outcomes: missing outcome {o!r}")
            continue
        for key in ("downtime_days", "complex_shutdown_days", "damage_fraction", "er_cost_usd"):
            v = outcomes[o].get(key)
            if v is None:
                errs.append(f"aleatory.outcomes.{o}: missing {key!r}")
            elif isinstance(v, str) and v not in names:
                errs.append(f"aleatory.outcomes.{o}.{key}: unknown parameter {v!r}")
            elif not isinstance(v, str) and float(v) < 0:
                errs.append(f"aleatory.outcomes.{o}.{key}: must be non-negative")
    for c in al.get("trip_platforms", []):
        if c not in codes:
            errs.append(f"aleatory.trip_platforms: unknown platform {c!r}")
    tgt = al.get("collision_targets", {})
    for c, w in tgt.items():
        if c not in codes or float(w) < 0:
            errs.append(f"aleatory.collision_targets: bad entry {c!r}: {w}")
    if tgt and sum(float(w) for w in tgt.values()) <= 0:
        errs.append("aleatory.collision_targets: weights must sum to a positive number")
    for k in ("downtime_sigma", "cost_sigma", "damage_beta_concentration", "max_event_downtime_days"):
        if float(al.get(k, 0)) <= 0:
            errs.append(f"aleatory.{k}: must be positive")


def _check_drivers(cfg, errs: list[str]):
    drv = cfg.drivers
    names = list(drv.get("names", []))
    c = np.asarray(drv.get("correlation", []), dtype=float)
    if c.shape != (len(names), len(names)):
        errs.append("drivers.correlation: must be a square matrix matching drivers.names")
    else:
        if not np.allclose(c, c.T) or not np.allclose(np.diag(c), 1.0):
            errs.append("drivers.correlation: must be symmetric with unit diagonal")
        elif np.min(np.linalg.eigvalsh(c)) <= 0:
            errs.append("drivers.correlation: must be positive definite")
    for n in names:
        if float(drv.get("sigma", {}).get(n, -1)) < 0:
            errs.append(f"drivers.sigma: missing or negative for {n!r}")
    fd = drv.get("frequency_driver", {})
    for s in SOURCES_NEEDING_DRIVER:
        if fd.get(s) not in names:
            errs.append(f"drivers.frequency_driver: source {s!r} needs one of {names}")
    if "logistics" not in names:
        errs.append("drivers.names: a 'logistics' driver is required (it scales repair durations)")


def _check_mitigations(cfg, errs: list[str]):
    names = set(cfg.registry.names)
    for k, m in cfg.mitigations.items():
        for f in ("name", "capex_usd", "life_years", "effectiveness"):
            if f not in m:
                errs.append(f"mitigation {k}: missing {f!r}")
        if float(m.get("capex_usd", 0)) < 0 or float(m.get("opex_usd_per_year", 0)) < 0:
            errs.append(f"mitigation {k}: costs must be non-negative")
        if float(m.get("life_years", 0)) <= 0:
            errs.append(f"mitigation {k}: life_years must be positive")
        for fx in m.get("effects", []) or []:
            if fx.get("param") not in names:
                errs.append(f"mitigation {k}: unknown parameter {fx.get('param')!r}")
            if fx.get("op") not in EFFECT_OPS:
                errs.append(f"mitigation {k}: op {fx.get('op')!r} not in {sorted(EFFECT_OPS)}")
        for a in m.get("architecture") or {}:
            if a not in ARCHITECTURE_KEYS:
                errs.append(f"mitigation {k}: unknown architecture key {a!r}")
    if cfg.budget_usd < 0:
        errs.append("mitigation budget must be non-negative")


def _check_scenarios(cfg, codes: set[str], errs: list[str]):
    from .event_tree.hydrocarbon_release import OUTCOMES
    from .scenarios.analysis import FREQUENCY_METHODS

    names = set(cfg.registry.names)
    nb = _neighbours(cfg)
    for k, s in cfg.scenarios.items():
        for f in ("title", "initiating_event"):
            if not s.get(f):
                errs.append(f"scenario {k}: missing {f!r}")
        if s.get("frequency", "state") not in FREQUENCY_METHODS:
            errs.append(f"scenario {k}: frequency {s.get('frequency')!r} not in {FREQUENCY_METHODS}")
        for fe in s.get("forced_events", []) or []:
            t = fe.get("type")
            if t not in FORCED_TYPES:
                errs.append(f"scenario {k}: forced event type {t!r} not in {sorted(FORCED_TYPES)}")
                continue
            if t == "release":
                p = fe.get("platform")
                if p not in codes:
                    errs.append(f"scenario {k}: release platform {p!r} unknown")
                if fe.get("size") is not None and fe["size"] not in SIZES:
                    errs.append(f"scenario {k}: size {fe['size']!r} not in {sorted(SIZES)}")
                if fe.get("phase") is not None and fe["phase"] not in ("gas", "liquid"):
                    errs.append(f"scenario {k}: phase must be 'gas' or 'liquid'")
                if fe.get("outcome") is not None and fe["outcome"] not in OUTCOMES:
                    errs.append(f"scenario {k}: outcome {fe['outcome']!r} not in {OUTCOMES}")
                for e in fe.get("escalate_to", []) or []:
                    if e not in nb.get(p, set()):
                        errs.append(f"scenario {k}: {e!r} is not a bridge neighbour of {p!r}; escalation would be ignored")
        for p, ov in (s.get("overrides") or {}).items():
            if p not in names:
                errs.append(f"scenario {k}: override of unknown parameter {p!r}")
            if ov.get("op") not in OVERRIDE_OPS:
                errs.append(f"scenario {k}: override op {ov.get('op')!r} not in {sorted(OVERRIDE_OPS)}")
        for a in s.get("architecture") or {}:
            if a not in ARCHITECTURE_KEYS:
                errs.append(f"scenario {k}: unknown architecture key {a!r}")
        if s.get("price_override") is not None and float(s["price_override"]) < 0:
            errs.append(f"scenario {k}: price_override must be non-negative")
        if float(s.get("logistics_multiplier", 1.0)) <= 0:
            errs.append(f"scenario {k}: logistics_multiplier must be positive")


def _check_appetite(cfg, errs: list[str]):
    from .appetite.framework import APPETITE_METRICS

    app = cfg.appetite
    if not 0 < float(app.get("near_threshold_fraction", 0.8)) < 1:
        errs.append("risk_appetite.near_threshold_fraction must be in (0, 1)")
    for k, lim in app.get("limits", {}).items():
        if lim.get("metric") not in APPETITE_METRICS:
            errs.append(f"appetite limit {k}: metric {lim.get('metric')!r} not in {APPETITE_METRICS}")
        if float(lim.get("limit", 0)) <= 0:
            errs.append(f"appetite limit {k}: limit must be positive")


def validate_config(cfg: ModelConfig) -> None:
    """Raise :class:`ConfigError` listing every problem, or return None if the configuration is valid."""
    errs: list[str] = []
    _check_parameters(cfg, errs)
    codes = _check_asset(cfg, errs)
    _check_aleatory(cfg, codes, errs)
    _check_drivers(cfg, errs)
    _check_mitigations(cfg, errs)
    _check_scenarios(cfg, codes, errs)
    _check_appetite(cfg, errs)
    if errs:
        raise ConfigError(errs)
