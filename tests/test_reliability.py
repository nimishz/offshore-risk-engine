from math import comb

import numpy as np
import pytest
from scipy import integrate

from offshore_risk.reliability import models as rm


def test_exponential_basics():
    assert rm.exp_reliability(0.0, 100) == 1.0
    assert rm.exp_reliability(0.01, 100) == pytest.approx(np.exp(-1))
    assert rm.exp_unreliability(1e-9, 1) == pytest.approx(1e-9, rel=1e-6)


def test_weibull_reduces_to_exponential_when_shape_is_one():
    t = np.linspace(0, 10, 11)
    assert np.allclose(rm.weibull_reliability(t, 1.0, 4.0), rm.exp_reliability(0.25, t))
    assert np.allclose(rm.weibull_hazard(t[1:], 1.0, 4.0), 0.25)


def test_weibull_mean_by_integration():
    shape, scale = 1.7, 3.0
    num, _ = integrate.quad(lambda t: rm.weibull_reliability(t, shape, scale), 0, np.inf)
    assert rm.weibull_mean(shape, scale) == pytest.approx(num, rel=1e-6)


def test_weibull_hazard_increasing_for_wear_out():
    h = rm.weibull_hazard(np.array([0.5, 1, 2, 4]), 2.0, 3.0)
    assert np.all(np.diff(h) > 0)


def test_conditional_reliability():
    shape, scale = 2.0, 5.0
    r = rm.weibull_conditional_reliability(2.0, 1.0, shape, scale)
    assert r == pytest.approx(rm.weibull_reliability(3.0, shape, scale) / rm.weibull_reliability(2.0, shape, scale))


def test_nhpp_expected_failures_is_cumulative_hazard_increment():
    assert rm.nhpp_expected_failures(0, 3.5, 1.6, 3.5) == pytest.approx(1.0)
    # constant hazard: expected failures = lambda * dt regardless of age
    assert rm.nhpp_expected_failures(7, 8, 1.0, 2.0) == pytest.approx(0.5)


def test_series_parallel_koon():
    r = [0.9, 0.8, 0.95]
    assert rm.series(*r) == pytest.approx(0.9 * 0.8 * 0.95)
    assert rm.parallel(*r) == pytest.approx(1 - 0.1 * 0.2 * 0.05)
    assert rm.k_out_of_n(3, r) == pytest.approx(rm.series(*r))
    assert rm.k_out_of_n(1, r) == pytest.approx(rm.parallel(*r))
    # identical 2oo3 closed form
    p = 0.9
    assert rm.k_out_of_n(2, [p] * 3) == pytest.approx(3 * p**2 * (1 - p) + p**3)


def test_koon_vectorised():
    out = rm.k_out_of_n(2, [np.array([0.9, 0.5]), 0.9, 0.9])
    assert out.shape == (2,)


def test_standby_closed_form_and_limits():
    lam, t = 1e-3, 1000.0
    assert rm.standby_reliability(lam, t) == pytest.approx(np.exp(-1) * 2)
    # a standby that never starts is no better than a single unit
    assert rm.standby_reliability(lam, t, q_start=1.0) == pytest.approx(np.exp(-1))
    # cold standby beats active parallel of two identical units
    assert rm.standby_reliability(lam, t) > rm.parallel(np.exp(-1), np.exp(-1))
    # warm standby: between single unit and cold standby; tends to cold as lam_d -> 0
    warm = rm.standby_reliability(lam, t, lam_dormant=5e-4)
    assert np.exp(-1) < warm < 2 * np.exp(-1)
    assert rm.standby_reliability(lam, t, lam_dormant=1e-12) == pytest.approx(2 * np.exp(-1), rel=1e-6)


def test_standby_by_monte_carlo():
    rng = np.random.default_rng(0)
    lam, t, q = 2e-3, 600.0, 0.1
    a = rng.exponential(1 / lam, 400_000)
    s_ok = rng.random(400_000) > q
    b = rng.exponential(1 / lam, 400_000)
    life = a + np.where(s_ok, b, 0.0)
    assert (life > t).mean() == pytest.approx(rm.standby_reliability(lam, t, q_start=q), abs=3e-3)


def test_pfd_formulas():
    lam, T = 2e-6, 8760.0
    assert rm.pfd_1oo1(lam, T) == pytest.approx(lam * T / 2)
    assert rm.pfd_koon(1, 1, lam, T) == pytest.approx(lam * T / 2)
    assert rm.pfd_koon(1, 2, lam, T) == pytest.approx((lam * T) ** 2 / 3)
    assert rm.pfd_koon(2, 3, lam, T) == pytest.approx((lam * T) ** 2)
    beta = 0.1
    expected = ((1 - beta) * lam * T) ** 2 / 3 + beta * lam * T / 2
    assert rm.pfd_koon(1, 2, lam, T, beta) == pytest.approx(expected)


def test_pfd_1oo2_by_simulation():
    """Time-average unavailability of 1oo2 with periodic perfect proof test, no CCF."""
    rng = np.random.default_rng(1)
    lam, T, n = 5e-5, 1000.0, 200_000
    t1, t2 = rng.exponential(1 / lam, n), rng.exponential(1 / lam, n)
    down = np.clip(T - np.maximum(t1, t2), 0, None)  # time both are failed within the interval
    exact, _ = integrate.quad(lambda t: (1 - np.exp(-lam * t)) ** 2, 0, T)
    exact /= T
    assert down.mean() / T == pytest.approx(exact, rel=0.03)
    # the simplified formula is a first-order approximation that errs on the conservative side
    approx = rm.pfd_koon(1, 2, lam, T)
    assert exact < approx < 1.1 * exact


def test_ccf_dominates_redundancy():
    lam, T = 3e-6, 8760.0
    no_ccf = rm.pfd_koon(1, 2, lam, T, 0.0)
    with_ccf = rm.pfd_koon(1, 2, lam, T, 0.1)
    assert with_ccf > 3 * no_ccf


def test_repairable_frequency_formula_and_mc():
    lam, mttr = 1e-3, 20.0
    f = rm.repairable_koon_failure_frequency(2, 3, lam, mttr)
    assert f == pytest.approx(comb(3, 2) * 2 * lam**2 * mttr)
    # 2oo4 needs three concurrent failures
    f4 = rm.repairable_koon_failure_frequency(2, 4, lam, mttr)
    assert f4 == pytest.approx(comb(4, 3) * 3 * lam**3 * mttr**2)
    # common cause adds beta*lam and is not reduced by adding units
    assert rm.repairable_koon_failure_frequency(2, 4, lam, mttr, 0.05) > 0.05 * lam


def test_repairable_2oo3_by_event_simulation():
    """Discrete-event check of the 2oo3 frequency approximation (exponential repair)."""
    rng = np.random.default_rng(5)
    lam, mu = 1e-2, 1 / 5.0  # lam*mttr = 0.05
    t_end, t, failed, sys_fail = 2e6, 0.0, 0, 0
    while t < t_end:
        up = 3 - failed
        r_fail, r_rep = up * lam, failed * mu
        t += rng.exponential(1 / (r_fail + r_rep))
        if rng.random() < r_fail / (r_fail + r_rep):
            failed += 1
            if failed == 2:
                sys_fail += 1
        else:
            failed -= 1
    f_sim = sys_fail / t_end
    assert f_sim == pytest.approx(rm.repairable_koon_failure_frequency(2, 3, lam, 5.0), rel=0.15)


def test_beta_ignored_for_single_channel():
    assert rm.pfd_koon(1, 1, 2e-6, 8760, beta=0.1) == pytest.approx(rm.pfd_1oo1(2e-6, 8760))
