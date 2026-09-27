"""Conditional stress scenarios.

A scenario answers: *given* this initiating event or degraded state, what does
the year look like? Forced events are added to every simulated year, and the
rest of the year is simulated normally with the same random numbers as the
baseline, so the per-year difference isolates the scenario's own effect.

Scenario probabilities are the model's annual frequency of the initiating
event, averaged over epistemic uncertainty. A degraded *state* has no
frequency in this model and is reported as conditional.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..asset.network import AssetModel
from ..config import ModelConfig
from ..event_tree.hydrocarbon_release import OUTCOMES, RELEASE_TREE
from ..fault_tree.protection import fire_protection_basic_events, fire_protection_tree
from ..financial.metrics import risk_metrics
from ..simulation.derived import derive
from ..simulation.engine import ScenarioSpec, SimulationSettings, Simulator

FREQUENCY_METHODS = (
    "state",
    "compressor",
    "release_PPA_large_gas",
    "major_fire_PPA",
    "explosion_PPB_escalating_PPA_UCP",
    "weather_and_compressor",
    "power_ccf",
)
REFERENCE_PLATFORM = "PPA"  # platform whose fire-protection PFD is reported per scenario


def _theta_sample(cfg: ModelConfig, n: int = 4000, seed: int = 11):
    reg = cfg.registry
    return reg.sample(np.random.default_rng(seed).random((n, len(reg))))


def outcome_probabilities(th, d, asset: AssetModel, plat: str) -> dict:
    """Event-tree outcome probabilities for a random release on `plat`, mixing sizes and phases."""
    i = asset.index(plat)
    g = asset.gas_fraction[i]
    tot = dict.fromkeys(OUTCOMES, 0.0)
    for s in range(3):
        for gas, w_phase in ((True, g), (False, 1 - g)):
            params = {
                "p_detect": d.p_detect[:, s],
                "p_isolate": d.p_isolate,
                "p_ign": d.p_ign[:, s],
                "p_ign_iso": d.p_ign[:, s] * th["ign_isolation_factor"],
                "p_exp": d.p_exp[:, s] * (1.0 if gas else th["liquid_explosion_factor"]),
                "p_fp_ok": 1 - d.pfd_fp[:, i],
            }
            op = RELEASE_TREE.outcome_probabilities(params)
            for o in OUTCOMES:
                tot[o] = tot[o] + d.size_probs[:, s] * w_phase * op[o]
    return tot


def scenario_frequency(cfg: ModelConfig, method: str, window_days: float = 10.0) -> float | None:
    """Annual frequency of a scenario's initiating event (mean over epistemic uncertainty).

    ``weather_and_compressor`` approximates the joint event as a storm followed
    by a compressor failure within ``window_days`` (independent Poisson processes).
    """
    if method not in FREQUENCY_METHODS:
        raise ValueError(f"unknown frequency method {method!r}")
    if method == "state":
        return None
    asset = AssetModel.from_config(cfg.asset)
    th = _theta_sample(cfg)
    d = derive(th, asset)
    if method == "compressor":
        return float(d.compressor_rate.mean())
    if method == "release_PPA_large_gas":
        i = asset.index("PPA")
        return float((d.release_rate[:, i] * d.size_probs[:, 2] * asset.gas_fraction[i]).mean())
    if method == "major_fire_PPA":
        i = asset.index("PPA")
        return float((d.release_rate[:, i] * outcome_probabilities(th, d, asset, "PPA")["major_fire"]).mean())
    if method == "explosion_PPB_escalating_PPA_UCP":
        i = asset.index("PPB")
        p_exp = outcome_probabilities(th, d, asset, "PPB")["explosion"]
        p_esc = th["esc_explosion"] * (1 - th["p_er_delayed"] + th["p_er_delayed"] * th["er_delay_escalation_multiplier"])
        return float((d.release_rate[:, i] * p_exp * np.minimum(p_esc, 1) ** 2).mean())
    if method == "weather_and_compressor":
        return float((d.weather_rate * d.compressor_rate * window_days / 365.0).mean())
    # power_ccf: common-cause loss of all generators
    return float((8760.0 * th["gen_ccf_beta"] * th["gen_lambda_per_h"]).mean())


def fire_protection_pfd_power_lost(cfg: ModelConfig, architecture: dict) -> float:
    """Fire-protection PFD at median parameters when main power is certainly lost."""
    asset = AssetModel.from_config(cfg.asset)
    arch = {**asset.protection, **architecture}
    th = cfg.registry.at_quantile(1, 0.5)
    ne, nd = arch["electric_firewater_pumps"], arch["diesel_firewater_pumps"]
    be = fire_protection_basic_events(th, 1.0, ne, nd, arch["flame_detectors_per_zone"], arch["flame_detection_voting_k"])
    return float(fire_protection_tree(ne, nd).probability(be)[0])


def run_scenarios(
    cfg: ModelConfig, n_years: int = 20_000, seed: int = 20260927, dependence: str = "correlated", keys=None
) -> tuple[pd.DataFrame, dict]:
    """Simulate the baseline and every scenario with common random numbers.

    Returns (summary DataFrame, dict of SimulationResult including 'baseline').
    """
    st = SimulationSettings(n_years=n_years, seed=seed, dependence=dependence)
    base = Simulator(cfg).run(st)
    results = {"baseline": base}
    bm = risk_metrics(base.total)
    rows = [
        {
            "scenario": "baseline",
            "title": "Baseline (no forced event)",
            "initiating_event": "-",
            "annual_frequency": np.nan,
            "return_period_years": np.nan,
            "mean_event_loss": np.nan,
            **{f"annual_{k}": bm[k] for k in ("eal", "median", "p95", "p99", "es99")},
            "incremental_mean_loss": 0.0,
            "mean_downtime_days": float(base.downtime_days.mean()),
            "p95_downtime_days": float(np.quantile(base.downtime_days, 0.95)),
            "pfd_fire_protection_PPA": base.platform_value("pfd_fire_protection", REFERENCE_PLATFORM),
            "assumptions": "",
        }
    ]
    for key, spec in cfg.scenarios.items():
        if keys is not None and key not in keys:
            continue
        sc = ScenarioSpec.from_config(key, spec)
        res = Simulator(cfg, scenario=sc).run(st)
        results[key] = res
        m = risk_metrics(res.total)
        freq = scenario_frequency(cfg, spec.get("frequency", "state"), float(spec.get("frequency_window_days", 10.0)))
        if spec.get("report_pfd_with_power_lost"):
            pfd_fp = fire_protection_pfd_power_lost(cfg, sc.architecture)
        else:
            pfd_fp = res.platform_value("pfd_fire_protection", REFERENCE_PLATFORM)
        rows.append(
            {
                "scenario": key,
                "title": spec["title"],
                "initiating_event": spec["initiating_event"],
                "annual_frequency": freq if freq is not None else np.nan,
                "return_period_years": (1 / freq) if freq else np.nan,
                "mean_event_loss": float(res.loss_by_source[:, -1].mean()) if sc.forced_events else np.nan,
                **{f"annual_{k}": m[k] for k in ("eal", "median", "p95", "p99", "es99")},
                "incremental_mean_loss": float((res.total - base.total).mean()),
                "mean_downtime_days": float(res.downtime_days.mean()),
                "p95_downtime_days": float(np.quantile(res.downtime_days, 0.95)),
                "pfd_fire_protection_PPA": pfd_fp,
                "assumptions": "; ".join(spec.get("assumptions", [])),
            }
        )
    return pd.DataFrame(rows), results
