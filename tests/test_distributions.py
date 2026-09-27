import numpy as np
import pytest

from offshore_risk.distributions import (
    beta_from_mean,
    latin_hypercube,
    lognormal_multiplier,
    make_distribution,
)

SPECS = [
    {"type": "lognormal", "median": 0.1, "ef": 3},
    {"type": "triangular", "low": 1, "mode": 2, "high": 5},
    {"type": "pert", "low": 1, "mode": 2, "high": 5},
    {"type": "uniform", "low": -1, "high": 3},
    {"type": "beta", "mean": 0.02, "n": 50},
    {"type": "beta", "a": 2, "b": 38},
    {"type": "gamma", "shape": 3, "rate": 2},
    {"type": "normal", "mean": 0, "sd": 2},
]


@pytest.mark.parametrize("spec", SPECS)
def test_ppf_cdf_roundtrip(spec):
    d = make_distribution(spec)
    u = np.linspace(0.01, 0.99, 25)
    assert np.allclose(d.cdf(d.ppf(u)), u, atol=1e-8)


@pytest.mark.parametrize("spec", SPECS)
def test_sample_mean_matches_analytic(spec):
    d = make_distribution(spec)
    x = d.sample(np.random.default_rng(1), 200_000)
    sd = np.std(x)
    assert abs(x.mean() - d.mean()) < 5 * sd / np.sqrt(len(x))


def test_lognormal_error_factor_definition():
    d = make_distribution({"type": "lognormal", "median": 2.0, "ef": 3.0})
    assert d.median() == pytest.approx(2.0)
    assert d.quantile(0.95) / d.median() == pytest.approx(3.0, rel=1e-6)
    assert d.median() / d.quantile(0.05) == pytest.approx(3.0, rel=1e-6)


def test_fixed_distribution():
    d = make_distribution({"type": "fixed", "value": 4.2})
    assert np.all(d.ppf([0.0, 0.3, 1.0]) == 4.2)
    assert d.is_fixed and d.mean() == 4.2


def test_invalid_specs():
    with pytest.raises(ValueError):
        make_distribution({"type": "triangular", "low": 3, "mode": 2, "high": 5})
    with pytest.raises(ValueError):
        make_distribution({"type": "nope"})
    with pytest.raises(ValueError):
        make_distribution({"type": "beta", "a": 0, "b": 1})


def test_lognormal_multiplier_has_unit_mean():
    z = np.random.default_rng(0).standard_normal(1_000_000)
    assert lognormal_multiplier(z, 0.5).mean() == pytest.approx(1.0, abs=0.005)


def test_beta_from_mean():
    u = np.random.default_rng(2).random(200_000)
    x = beta_from_mean(np.full_like(u, 0.06), 8.0, u)
    assert x.mean() == pytest.approx(0.06, rel=0.02)
    assert (x >= 0).all() and (x <= 1).all()


def test_latin_hypercube_stratification():
    u = latin_hypercube(50, 4, np.random.default_rng(3))
    for j in range(4):
        assert sorted(np.floor(u[:, j] * 50).astype(int)) == list(range(50))
