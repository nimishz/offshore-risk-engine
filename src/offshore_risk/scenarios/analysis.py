"""Conditional stress scenarios.

A scenario answers: *given* this initiating event or degraded state, what does
the year look like? Forced events are added to every simulated year, and the
rest of the year is simulated normally with the same random numbers as the
baseline, so the per-year difference isolates the scenario's own effect.

Scenario probabilities are the model's (epistemic-mean) annual frequency of the
initiating event; a degraded *state* has no frequency in this model and is
reported as conditional.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..asset.network import AssetModel
from ..config import ModelConfig
from ..event_tree.hydrocarbon_release import OUTCOMES, RELEASE_TREE
from ..financial.metrics import risk_metrics
from ..simulation.derived import derive
from ..simulation.engine import ScenarioSpec, SimulationSettings, Simulator


def _theta_sample(cfg: ModelConfig, n: int = 4000, seed: int = 11):
    reg = cfg.registry
    return reg.sample(np.random.default_rng(seed).random((n, len(reg))))


def _outcome_probs(cfg, th, d, asset, plat: str) -> dict:
    """Event-tree outcome probabilities for a random release on `plat`, mixing sizes and phases."""
    i = asset.index(plat)
    g = asset.gas_fraction[i]
    tot = {o: 0.0 for o in OUTCOMES}
    for s in range(3):
        for gas, w_phase in ((True, g), (False, 1 - g)):
            params = {
                "p_detect": d.p_detect[:, s], "p_isolate": d.p_isolate,
                "p_ign": d.p_ign[:, s], "p_ign_iso": d.p_ign[:, s] * th["ign_isolation_factor"],
                "p_exp": d.p_exp[:, s] * (1.0 if gas else th["liquid_explosion_factor"]),
                "p_fp_ok": 1 - d.pfd_fp[:, i],
            }
            op = RELEASE_TREE.outcome_probabilities(params)
            for o in OUTCOMES:
                tot[o] = tot[o] + d.size_probs[:, s] * w_phase * op[o]
    return tot


def scenario_frequency(cfg: ModelConfig, method: str) -> float | None:
    """Annual frequency of a scenario's initiating event (mean over epistemic uncertainty)."""
    asset = AssetModel.from_config(cfg.asset)
    th = _theta_sample(cfg)
    d = derive(th, asset)
    if method == "state":
        return None
    if method == "compressor":
        return float(d.compressor_rate.mean())
    if method == "release_PPA_large_gas":
        i = asset.index("PPA")
        return float((d.release_rate[:, i] * d.size_probs[:, 2] * asset.gas_fraction[i]).mean())
    if method == "major_fire_PPA":
        i = asset.index("PPA")
        return float((d.release_rate[:, i] * _outcome_probs(cfg, th, d, asset, "PPA")["major_fire"]).mean())
    if method == "explosion_PPB_escalating_PPA_UCP":
        i = asset.index("PPB")
        p_exp = _outcome_probs(cfg, th, d, asset, "PPB")["explosion"]
        p_esc = th["esc_explosion"] * (1 - th["p_er_delayed"] + th["p_er_delayed"] * th["er_delay_escalation_multiplier"])
        return float((d.release_rate[:, i] * p_exp * np.minimum(p_esc, 1) ** 2).mean())
    if method == "weather_and_compressor":
        window = 10.0 / 365.0
        return float((d.weather_rate * d.compressor_rate * window).mean())
    if method == "power_ccf":
        return float((8760.0 * th["gen_ccf_beta"] * th["gen_lambda_per_h"]).mean())
    raise ValueError(f"unknown frequency method {method!r}")


def run_scenarios(cfg: ModelConfig, n_years: int = 20_000, seed: int = 20260927, dependence: str = "correlated",
                  keys=None):
    """Simulate the baseline and every scenario with common random numbers.

    Returns (summary DataFrame, dict of SimulationResult incl. 'baseline').
    """
    st = SimulationSettings(n_years=n_years, seed=seed, dependence=dependence)
    results = {"baseline": Simulator(cfg).run(st)}
    base = results["baseline"]
    rows = []
    bm = risk_metrics(base.total)
    rows.append({"scenario": "baseline", "title": "Baseline (no forced event)", "initiating_event": "-",
                 "annual_frequency": np.nan, "return_period_years": np.nan,
                 "mean_event_loss": np.nan, **{f"annual_{k}": bm[k] for k in ("eal", "median", "p95", "p99", "es99")},
                 "incremental_mean_loss": 0.0, "mean_downtime_days": float(base.downtime_days.mean()),
                 "p95_downtime_days": float(np.quantile(base.downtime_days, 0.95)),
                 "pfd_fire_protection_PPA": float(base.derived_means["pfd_fire_protection"][1]), "assumptions": ""})
    asset = AssetModel.from_config(cfg.asset)
    for key, spec in cfg.scenarios.items():
        if keys is not None and key not in keys:
            continue
        sc = ScenarioSpec.from_config(key, spec)
        res = Simulator(cfg, scenario=sc).run(st)
        results[key] = res
        m = risk_metrics(res.total)
        freq = scenario_frequency(cfg, spec.get("frequency", "state"))
        ev = res.loss_by_source[:, -1]
        pfd_fp = res.derived_means["pfd_fire_protection"][asset.index("PPA")]
        if key.startswith("S6"):
            # fire protection while main power is lost (electric pumps unavailable)
            from ..fault_tree.protection import fire_protection_basic_events, fire_protection_tree

            th = cfg.registry.at_quantile(1, 0.5)
            arch = {**asset.protection, **sc.architecture}
            be = fire_protection_basic_events(th, 1.0, arch["electric_firewater_pumps"], arch["diesel_firewater_pumps"])
            pfd_fp = float(fire_protection_tree(arch["electric_firewater_pumps"], arch["diesel_firewater_pumps"]).probability(be)[0])
        rows.append({
            "scenario": key, "title": spec["title"], "initiating_event": spec["initiating_event"],
            "annual_frequency": freq if freq is not None else np.nan,
            "return_period_years": (1 / freq) if freq else np.nan,
            "mean_event_loss": float(ev.mean()) if sc.forced_events else np.nan,
            **{f"annual_{k}": m[k] for k in ("eal", "median", "p95", "p99", "es99")},
            "incremental_mean_loss": float((res.total - base.total).mean()),
            "mean_downtime_days": float(res.downtime_days.mean()),
            "p95_downtime_days": float(np.quantile(res.downtime_days, 0.95)),
            "pfd_fire_protection_PPA": float(pfd_fp),
            "assumptions": "; ".join(spec.get("assumptions", [])),
        })
    return pd.DataFrame(rows), results
