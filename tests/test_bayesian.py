import numpy as np
import pytest
from scipy import stats

from offshore_risk.bayesian import BetaModel, GammaModel, gamma_from_lognormal, sequential_update
from offshore_risk.bayesian.conjugate import grid_posterior
from offshore_risk.distributions import make_distribution


def test_beta_binomial_update():
    post = BetaModel.from_mean(0.01, 100).update(5, 200)
    assert (post.a, post.b) == pytest.approx((6.0, 294.0))
    assert post.mean() == pytest.approx(6 / 300)


def test_beta_posterior_matches_grid():
    prior = BetaModel(1.0, 99.0)
    k, n = 7, 300
    grid = np.linspace(1e-6, 0.2, 20001)
    num = grid_posterior(prior.pdf, lambda p: stats.binom.pmf(k, n, p), grid)
    assert np.allclose(num, prior.update(k, n).pdf(grid), atol=1e-3 * num.max())


def test_gamma_poisson_update_and_grid():
    prior = GammaModel(2.0, 1.5)
    post = prior.update(9, 12.0)
    assert (post.shape, post.rate) == pytest.approx((11.0, 13.5))
    grid = np.linspace(1e-6, 3, 20001)
    num = grid_posterior(prior.pdf, lambda lam: stats.poisson.pmf(9, lam * 12.0), grid)
    assert np.allclose(num, post.pdf(grid), atol=1e-3 * num.max())


def test_gamma_moment_matching():
    g = gamma_from_lognormal(3.0, 2.0)
    ln = make_distribution({"type": "lognormal", "median": 3.0, "ef": 2.0})
    assert g.mean() == pytest.approx(ln.mean(), rel=1e-9)
    assert g.dist.var() == pytest.approx(ln._frozen.var(), rel=1e-9)


def test_more_data_narrows_uncertainty():
    df = sequential_update(BetaModel(1, 49), [1] * 10, [50] * 10)
    assert df["width"].iloc[-1] < df["width"].iloc[0]
    assert df["width"].is_monotonic_decreasing


def test_posterior_mean_between_prior_and_mle():
    prior = BetaModel.from_mean(0.01, 100)
    post = prior.update(30, 1000)
    assert 0.01 < post.mean() < 0.03


def test_predictive_negative_binomial():
    g = GammaModel(4.0, 2.0)
    pred = g.predictive(1.0)
    assert pred.mean() == pytest.approx(g.mean())
    assert pred.var() > pred.mean()  # overdispersed relative to Poisson


def test_invalid_updates():
    with pytest.raises(ValueError):
        BetaModel(1, 1).update(5, 3)
    with pytest.raises(ValueError):
        GammaModel(1, 1).update(-1, 3)


def test_calibration_runs(cfg, tmp_path):
    from offshore_risk.bayesian.calibration import apply_posteriors, calibrate, summary_table
    from offshore_risk.data import synthetic

    data = synthetic.generate(cfg, tmp_path)
    cal = calibrate(cfg, {k: data[k] for k in ("incident_log", "proof_tests", "equipment_register", "exposure")})
    tab = summary_table(cal)
    assert (tab["ci_width_reduction"] > 0).all()
    new = apply_posteriors(cfg, cal)
    assert new.registry["fw_elec_pump_fts"].dist.kind == "beta"
    assert new.registry["spurious_trip_rate"].dist.kind == "gamma"
    # original config untouched
    assert cfg.registry["spurious_trip_rate"].dist.kind == "lognormal"
