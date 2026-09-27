"""Reliability models used for failure frequencies and protection-layer PFDs.

All functions accept numpy arrays so they can be evaluated for every
epistemic sample at once.

Conventions
-----------
* ``lam`` failure rate (per hour unless stated), ``t`` time in the same unit.
* PFD = average probability of failure on demand of a periodically
  proof-tested, low-demand safety function (simplified IEC 61508-6 forms;
  repair time and diagnostic coverage neglected, proof tests assumed perfect).
"""

from __future__ import annotations

from math import comb

import numpy as np
from scipy.special import gamma as gamma_fn

# ---------------------------------------------------------------- constant hazard


def exp_reliability(lam, t):
    """R(t) = exp(-lam t) for a constant hazard rate."""
    return np.exp(-np.asarray(lam) * np.asarray(t))


def exp_unreliability(lam, t):
    return -np.expm1(-np.asarray(lam) * np.asarray(t))


def annual_probability(rate_per_year):
    """P(at least one event in a year) for a Poisson rate."""
    return -np.expm1(-np.asarray(rate_per_year))


# ---------------------------------------------------------------------- Weibull


def weibull_reliability(t, shape, scale):
    """R(t) = exp(-(t/scale)^shape)."""
    return np.exp(-((np.asarray(t) / scale) ** shape))


def weibull_hazard(t, shape, scale):
    """h(t) = (shape/scale) (t/scale)^(shape-1); increasing if shape > 1."""
    t = np.asarray(t, dtype=float)
    return (shape / scale) * (t / scale) ** (shape - 1.0)


def weibull_mean(shape, scale):
    return scale * gamma_fn(1.0 + 1.0 / np.asarray(shape))


def weibull_conditional_reliability(age, x, shape, scale):
    """P(survive a further x | survived to age) = R(age + x) / R(age)."""
    return np.exp(-(((age + x) / scale) ** shape - (age / scale) ** shape))


def nhpp_expected_failures(t0, t1, shape, scale):
    """Expected failures in [t0, t1] of a power-law NHPP (minimal repair).

    Cumulative intensity W(t) = (t/scale)^shape, i.e. the Weibull cumulative
    hazard. This is the right model for a repairable item that is restored to
    'as bad as old' after each failure and renewed only at overhaul.
    """
    return (np.asarray(t1) / scale) ** shape - (np.asarray(t0) / scale) ** shape


# ------------------------------------------------------------ system structures


def series(*reliabilities):
    """All components must work."""
    out = 1.0
    for r in reliabilities:
        out = out * np.asarray(r)
    return out


def parallel(*reliabilities):
    """At least one component must work (independent)."""
    q = 1.0
    for r in reliabilities:
        q = q * (1.0 - np.asarray(r))
    return 1.0 - q


def k_out_of_n(k: int, reliabilities):
    """P(at least k of n independent, possibly non-identical components work).

    Dynamic programming over the number of working components; exact.
    """
    rs = [np.asarray(r, dtype=float) for r in reliabilities]
    n = len(rs)
    if not 0 <= k <= n:
        raise ValueError("k must be between 0 and n")
    shape = np.broadcast(*rs).shape if rs else ()
    dist = [np.ones(shape)] + [np.zeros(shape) for _ in range(n)]  # P(j working)
    for r in rs:
        for j in range(n, 0, -1):
            dist[j] = dist[j] * (1 - r) + dist[j - 1] * r
        dist[0] = dist[0] * (1 - r)
    return sum(dist[k:])


def standby_reliability(lam, t, q_start=0.0, lam_dormant=0.0):
    """One active unit plus one cold/warm standby unit, identical rate ``lam``.

    The standby fails to start with probability ``q_start`` and may fail while
    dormant at ``lam_dormant`` (0 = cold standby). Perfect switch-over otherwise.
    For lam_dormant = 0:  R(t) = e^{-lam t} (1 + (1-q_start) lam t).
    """
    lam = np.asarray(lam, dtype=float)
    t = np.asarray(t, dtype=float)
    if np.all(np.asarray(lam_dormant) == 0):
        return np.exp(-lam * t) * (1.0 + (1.0 - q_start) * lam * t)
    ld = np.asarray(lam_dormant, dtype=float)
    # Active fails at time s (density lam e^{-lam s}); standby must have survived
    # dormancy to s, start, then survive (t - s).
    return np.exp(-lam * t) + (1.0 - q_start) * lam / ld * np.exp(-lam * t) * (1.0 - np.exp(-ld * t))


# ------------------------------------------------------------ safety functions


def pfd_1oo1(lam_du, test_interval):
    """PFDavg of a single proof-tested channel: lam_DU * T / 2."""
    return np.asarray(lam_du) * np.asarray(test_interval) / 2.0


def pfd_koon(k: int, n: int, lam_du, test_interval, beta=0.0):
    """PFDavg of a k-out-of-n voted group with beta-factor common cause.

    The group fails when m = n - k + 1 channels have failed. Independent part:
    C(n, m) ((1-beta) lam T)^m / (m + 1). Common-cause part: beta lam T / 2
    (all channels fail together). For n = 1 this reduces to lam T / 2.
    """
    if not 1 <= k <= n:
        raise ValueError("require 1 <= k <= n")
    lam_du = np.asarray(lam_du, dtype=float)
    beta = np.asarray(beta, dtype=float) if n > 1 else np.asarray(0.0)  # no common cause for a single channel
    m = n - k + 1
    lt = (1.0 - beta) * lam_du * test_interval
    independent = comb(n, m) * lt**m / (m + 1)
    ccf = beta * lam_du * test_interval / 2.0 if n > 1 else 0.0
    return independent + ccf


def beta_factor_split(q, beta):
    """Split a per-component failure probability into (independent, common-cause) parts."""
    q = np.asarray(q)
    return (1.0 - beta) * q, beta * q


def repairable_koon_failure_frequency(k: int, n: int, lam, mttr, beta=0.0):
    """Frequency (per unit time) of losing a k-out-of-n repairable system.

    Independent contribution: the system fails when the m-th concurrent failure
    occurs while m-1 units are under repair:  C(n, m) m lam^m mttr^(m-1),
    valid for lam * mttr << 1 and independent repair crews.
    Common-cause contribution: beta * lam (all units lost together).
    The independent rate uses (1 - beta) lam.
    """
    lam = np.asarray(lam, dtype=float)
    mttr = np.asarray(mttr, dtype=float)
    m = n - k + 1
    li = (1.0 - np.asarray(beta)) * lam
    independent = comb(n, m) * m * li**m * mttr ** (m - 1)
    return independent + np.asarray(beta) * lam
