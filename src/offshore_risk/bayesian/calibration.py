"""Bayesian calibration of selected parameters against the synthetic records.

Updated parameters (and why these):

* ``fw_elec_pump_fts`` / ``fw_diesel_pump_fts`` - weekly start tests are the
  kind of data an operator actually has in volume (Beta-Binomial).
  The two electric pumps are pooled (assumed exchangeable: same model, room, crew).
* ``spurious_trip_rate`` - trips are frequent, so a dozen years of records are
  informative (Gamma-Poisson).

The total hydrocarbon release frequency is also updated for *display*, but the
posterior is not fed back: the model builds the release frequency from causes,
and a total count cannot say which cause rate was wrong.
"""
from __future__ import annotations

import pandas as pd

from ..config import ModelConfig
from ..data import synthetic
from .conjugate import BetaModel, GammaModel, gamma_from_lognormal

FED_BACK = ("fw_elec_pump_fts", "fw_diesel_pump_fts", "spurious_trip_rate")


def _beta_prior(cfg: ModelConfig, name: str) -> BetaModel:
    d = cfg.registry[name].spec["distribution"]
    return BetaModel(float(d["a"]), float(d["b"])) if "a" in d else BetaModel.from_mean(float(d["mean"]), float(d["n"]))


def _gamma_prior(cfg: ModelConfig, name: str) -> GammaModel:
    d = cfg.registry[name].spec["distribution"]
    return gamma_from_lognormal(float(d["median"]), float(d["ef"]))


def calibrate(cfg: ModelConfig, data: dict | None = None) -> dict:
    """Return {parameter: {prior, data, posterior}} for the updated parameters."""
    data = data or synthetic.load()
    tests, log, exp = data["proof_tests"], data["incident_log"], data["exposure"]
    out = {}
    for name, typ in (("fw_elec_pump_fts", "electric_firewater_pump"), ("fw_diesel_pump_fts", "diesel_firewater_pump")):
        t = tests[tests["equipment_type"] == typ]
        k, n = int((t["result"] != "pass").sum()), int(len(t))
        prior = _beta_prior(cfg, name)
        out[name] = {"model": "beta-binomial", "prior": prior, "failures": k, "exposure": n, "exposure_unit": "demands",
                     "posterior": prior.update(k, n)}
    T = float(exp["complex_years"].sum())
    k_trip = int((log["subcategory"] == "spurious_trip").sum())
    prior = _gamma_prior(cfg, "spurious_trip_rate")
    out["spurious_trip_rate"] = {"model": "gamma-poisson", "prior": prior, "failures": k_trip, "exposure": T,
                                 "exposure_unit": "complex-years", "posterior": prior.update(k_trip, T)}
    # display only: total release frequency (prior moment-matched to the model's prior predictive)
    from ..asset.network import AssetModel
    from ..simulation.derived import release_cause_rates
    import numpy as np

    asset = AssetModel.from_config(cfg.asset)
    th = cfg.registry.sample(np.random.default_rng(3).random((20000, len(cfg.registry))))
    tot = sum(release_cause_rates(th).values()) * asset.release_factor.sum()
    m, v = float(tot.mean()), float(tot.var())
    prior_rel = GammaModel(m * m / v, m / v)
    k_rel = int((log["subcategory"] == "release").sum())
    out["total_release_rate"] = {"model": "gamma-poisson (display only)", "prior": prior_rel, "failures": k_rel, "exposure": T,
                                 "exposure_unit": "complex-years", "posterior": prior_rel.update(k_rel, T)}
    return out


def summary_table(cal: dict) -> pd.DataFrame:
    rows = []
    for name, c in cal.items():
        pr, po = c["prior"], c["posterior"]
        rows.append({
            "parameter": name, "model": c["model"], "observed_failures": c["failures"], "exposure": c["exposure"],
            "exposure_unit": c["exposure_unit"], "mle": c["failures"] / c["exposure"],
            "prior_mean": pr.mean(), "prior_90ci": "[{:.3g}, {:.3g}]".format(*pr.interval()),
            "posterior_mean": po.mean(), "posterior_90ci": "[{:.3g}, {:.3g}]".format(*po.interval()),
            "ci_width_reduction": 1 - (po.interval()[1] - po.interval()[0]) / (pr.interval()[1] - pr.interval()[0]),
            "fed_back_to_model": name in FED_BACK,
        })
    return pd.DataFrame(rows)


def apply_posteriors(cfg: ModelConfig, cal: dict | None = None) -> ModelConfig:
    """Copy of the config with the fed-back priors replaced by posteriors."""
    cal = cal or calibrate(cfg)
    new = cfg.copy()
    for name in FED_BACK:
        new.registry = new.registry.with_distribution(name, cal[name]["posterior"].spec())
    return new
