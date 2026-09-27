import numpy as np
import pytest

from offshore_risk.appetite.framework import NEAR, OUTSIDE, WITHIN, classify, evaluate_appetite
from offshore_risk.financial.loss_model import capital_recovery_factor
from offshore_risk.financial.metrics import (
    bootstrap_ci,
    exceedance_curve,
    exceedance_probability,
    expected_shortfall,
    risk_metrics,
    tail_allocation,
    var,
)


def test_metrics_on_known_sample():
    x = np.arange(1, 101, dtype=float)  # 1..100
    m = risk_metrics(x)
    assert m["eal"] == pytest.approx(50.5)
    assert m["median"] == pytest.approx(50.5)
    assert m["max"] == 100
    assert expected_shortfall(x, 0.95) == pytest.approx(np.mean(np.arange(96, 101)))
    assert m["es99"] >= m["var99"] >= m["p95"] >= m["median"]


def test_es_geq_var_random():
    x = np.random.default_rng(0).lognormal(0, 1.5, 50_000)
    for a in (0.9, 0.95, 0.99):
        assert expected_shortfall(x, a) >= var(x, a)


def test_exceedance():
    x = np.array([0, 0, 1, 2, 10.0])
    assert exceedance_probability(x, [0.5, 5])[0.5] == pytest.approx(0.6)
    lec = exceedance_curve(x, grid=np.array([0.5, 1.0, 5.0]))
    assert lec["exceedance_probability"].tolist() == pytest.approx([0.6, 0.4, 0.2])
    assert lec["exceedance_probability"].is_monotonic_decreasing


def test_tail_allocation_sums_to_es():
    rng = np.random.default_rng(1)
    M = np.column_stack([rng.exponential(1, 20_000), rng.pareto(2.5, 20_000) * 3])
    df = tail_allocation(M, ["a", "b"], 0.99)
    assert df["es99_contribution"].sum() == pytest.approx(expected_shortfall(M.sum(axis=1), 0.99), rel=1e-9)
    assert df["share_of_eal"].sum() == pytest.approx(1.0)


def test_bootstrap_ci_contains_estimate():
    x = np.random.default_rng(2).exponential(1.0, 5000)
    lo, hi = bootstrap_ci(x, np.mean, n_boot=200)
    assert lo < x.mean() < hi


def test_crf():
    assert capital_recovery_factor(0.0, 10) == pytest.approx(0.1)
    assert capital_recovery_factor(0.08, 10) == pytest.approx(0.149029, rel=1e-5)


def test_appetite_classification(cfg):
    assert classify(50, 100, 0.8) == WITHIN
    assert classify(85, 100, 0.8) == NEAR
    assert classify(101, 100, 0.8) == OUTSIDE
    metrics = {
        "eal": 1e6,
        "p95": 1e9,
        "downtime_p95_days": 24.0,
        "max_event_p99": 0,
        "pfd_firewater": 0.0,
        "pfd_fire_protection": 0.0,
    }
    df = evaluate_appetite(cfg.appetite, metrics).set_index("limit_id")
    assert df.loc["expected_annual_loss", "status"] == WITHIN
    assert df.loc["p95_annual_loss", "status"] == OUTSIDE
    assert df.loc["production_downtime_p95", "status"] == NEAR
