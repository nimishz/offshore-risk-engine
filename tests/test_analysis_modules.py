"""Smoke and consistency tests for the analysis layers (small sample sizes)."""

import numpy as np
import pandas as pd
import pytest

from offshore_risk.appetite.framework import evaluate_appetite, exposure_metrics
from offshore_risk.data import synthetic
from offshore_risk.data.dictionary import build as build_dictionary
from offshore_risk.mitigation.evaluate import annualised_cost, evaluate_mitigations
from offshore_risk.scenarios.analysis import FREQUENCY_METHODS, run_scenarios, scenario_frequency
from offshore_risk.sensitivity.analysis import first_order_indices, rank_correlation, tornado
from offshore_risk.simulation import simulate
from offshore_risk.simulation.nested import run_nested


@pytest.fixture(scope="module")
def nested(cfg):
    return run_nested(cfg, n_outer=40, n_inner=200, seed=3)


def test_nested_shapes_and_ordering(nested):
    w = nested.worlds
    assert len(w) == 40
    assert (w["es99"] >= w["p99"]).all() and (w["p99"] >= w["p95"]).all()
    assert nested.lec.shape == (40, len(nested.lec_grid))
    assert (np.diff(nested.lec, axis=1) <= 1e-12).all()  # exceedance curves are non-increasing


def test_rank_correlation_and_s1(nested):
    rc = rank_correlation(nested, "eal")
    s1 = first_order_indices(nested, "eal", bins=5)
    assert rc["abs_rho"].between(0, 1).all()
    assert s1["S1"].between(0, 1).all()
    assert "bi_value_fraction" in set(rc.head(10)["parameter"])  # the dominant driver shows up even at small N


def test_tornado_direction(cfg):
    t = tornado(cfg, n_years=2000, params=["bi_value_fraction", "comp_weibull_scale_years"], metrics=("eal",)).set_index(
        "parameter"
    )
    assert t.loc["bi_value_fraction", "eal_high"] > t.loc["bi_value_fraction", "eal_low"]
    assert t.loc["comp_weibull_scale_years", "eal_high"] < t.loc["comp_weibull_scale_years", "eal_low"]  # longer life, less loss


def test_scenarios_consistent_with_baseline(cfg):
    df, res = run_scenarios(cfg, n_years=2000, keys=["S1_single_equipment_failure", "S2_firewater_degradation"])
    df = df.set_index("scenario")
    assert df.loc["S1_single_equipment_failure", "incremental_mean_loss"] == pytest.approx(
        df.loc["S1_single_equipment_failure", "mean_event_loss"], rel=1e-9
    )
    assert df.loc["S2_firewater_degradation", "pfd_fire_protection_PPA"] > df.loc["baseline", "pfd_fire_protection_PPA"]
    assert np.isnan(df.loc["S2_firewater_degradation", "annual_frequency"])


@pytest.mark.parametrize("method", [m for m in FREQUENCY_METHODS if m != "state"])
def test_scenario_frequencies_positive(cfg, method):
    f = scenario_frequency(cfg, method)
    assert 0 < f < 10


def test_mitigations_reduce_risk(cfg):
    df, _ = evaluate_mitigations(cfg, keys=["ptw_competence", "gas_detection_upgrade"], n_years=3000)
    assert (df["reduction_eal"] > 0).all()
    m = cfg.mitigations["ptw_competence"]
    assert annualised_cost(m, 0.0) == pytest.approx(m["capex_usd"] / m["life_years"] + m["opex_usd_per_year"])


def test_appetite_from_simulation(cfg):
    r = simulate(cfg, n_years=2000)
    app = evaluate_appetite(cfg.appetite, exposure_metrics(r))
    assert set(app["status"]) <= {"Within appetite", "Near threshold", "Outside appetite"}
    assert len(app) == len(cfg.appetite["limits"])


def test_synthetic_data_deterministic(cfg, tmp_path):
    a = synthetic.generate(cfg, tmp_path / "a")
    b = synthetic.generate(cfg, tmp_path / "b")
    pd.testing.assert_frame_equal(a["incident_log"], b["incident_log"])
    pd.testing.assert_frame_equal(a["proof_tests"], b["proof_tests"])


def test_data_dictionary_covers_every_parameter(cfg, tmp_path):
    df = build_dictionary(cfg, tmp_path / "dd.md", tmp_path / "dd.csv")
    assert set(df["variable"]) == set(cfg.registry.names)
    text = (tmp_path / "dd.md").read_text()
    assert all(f"`{n}`" in text for n in cfg.registry.names)
