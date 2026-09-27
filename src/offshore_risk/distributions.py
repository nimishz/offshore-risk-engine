"""Parametric distributions, all sampled by inverse transform.

Every uncertain quantity in the model is produced as ``dist.ppf(u)`` from a
uniform ``u``. Using one mechanism everywhere gives, for free:

* Latin-hypercube sampling (stratified ``u``),
* Gaussian copulas (``u = Phi(z)`` with correlated ``z``),
* one-at-a-time sensitivity (``u = 0.1`` / ``0.9``),
* common random numbers between runs that differ only in parameters.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

Z95 = stats.norm.ppf(0.95)  # 1.6449: error factor EF = P95/P50 => sigma = ln(EF)/Z95
_EPS = 1e-12


@dataclass
class Distribution:
    """A frozen one-dimensional distribution defined by a config spec."""

    kind: str
    spec: dict = field(default_factory=dict)
    _frozen: Any = None
    _value: float | None = None

    # ------------------------------------------------------------------ core
    def ppf(self, u):
        u = np.clip(np.asarray(u, dtype=float), _EPS, 1.0 - _EPS)
        if self.kind == "fixed":
            return np.full_like(u, self._value, dtype=float)
        return self._frozen.ppf(u)

    def cdf(self, x):
        if self.kind == "fixed":
            return (np.asarray(x, dtype=float) >= self._value).astype(float)
        return self._frozen.cdf(x)

    def pdf(self, x):
        if self.kind == "fixed":
            raise ValueError("fixed distribution has no density")
        return self._frozen.pdf(x)

    def sample(self, rng: np.random.Generator, size) -> np.ndarray:
        return self.ppf(rng.random(size))

    # ------------------------------------------------------------ summaries
    def mean(self) -> float:
        return float(self._value) if self.kind == "fixed" else float(self._frozen.mean())

    def median(self) -> float:
        return float(self.ppf(0.5))

    def quantile(self, q: float) -> float:
        return float(self.ppf(q))

    @property
    def is_fixed(self) -> bool:
        return self.kind == "fixed"

    def describe(self) -> str:
        s = self.spec
        if self.kind == "fixed":
            return f"fixed {s['value']:g}"
        if self.kind == "lognormal":
            if "ef" in s:
                return f"lognormal(median={s['median']:g}, EF={s['ef']:g})"
            return f"lognormal(median={s['median']:g}, sigma={s['sigma']:g})"
        if self.kind in ("triangular", "pert"):
            return f"{self.kind}({s['low']:g}, {s['mode']:g}, {s['high']:g})"
        if self.kind == "uniform":
            return f"uniform({s['low']:g}, {s['high']:g})"
        if self.kind == "beta":
            a, b = _beta_ab(s)
            return f"beta(a={a:g}, b={b:g}; mean={a / (a + b):.3g})"
        if self.kind == "gamma":
            return f"gamma(shape={s['shape']:g}, rate={s['rate']:g})"
        if self.kind == "normal":
            return f"normal(mean={s['mean']:g}, sd={s['sd']:g})"
        return self.kind


def _beta_ab(spec: Mapping) -> tuple[float, float]:
    if "a" in spec:
        return float(spec["a"]), float(spec["b"])
    m, n = float(spec["mean"]), float(spec["n"])
    return m * n, (1.0 - m) * n


def make_distribution(spec: Mapping) -> Distribution:
    """Build a :class:`Distribution` from a YAML-style dictionary."""
    spec = dict(spec)
    kind = spec.get("type")
    if kind == "fixed":
        return Distribution("fixed", spec, None, float(spec["value"]))
    if kind == "lognormal":
        median = float(spec["median"])
        if "ef" in spec:
            sigma = np.log(float(spec["ef"])) / Z95
        else:
            sigma = float(spec["sigma"])
        return Distribution(kind, spec, stats.lognorm(s=sigma, scale=median))
    if kind == "triangular":
        lo, mo, hi = float(spec["low"]), float(spec["mode"]), float(spec["high"])
        _check_order(lo, mo, hi)
        return Distribution(kind, spec, stats.triang(c=(mo - lo) / (hi - lo), loc=lo, scale=hi - lo))
    if kind == "pert":
        lo, mo, hi = float(spec["low"]), float(spec["mode"]), float(spec["high"])
        _check_order(lo, mo, hi)
        a = 1.0 + 4.0 * (mo - lo) / (hi - lo)
        b = 1.0 + 4.0 * (hi - mo) / (hi - lo)
        return Distribution(kind, spec, stats.beta(a, b, loc=lo, scale=hi - lo))
    if kind == "uniform":
        lo, hi = float(spec["low"]), float(spec["high"])
        return Distribution(kind, spec, stats.uniform(loc=lo, scale=hi - lo))
    if kind == "beta":
        a, b = _beta_ab(spec)
        if a <= 0 or b <= 0:
            raise ValueError(f"beta parameters must be positive, got a={a}, b={b}")
        return Distribution(kind, spec, stats.beta(a, b))
    if kind == "gamma":
        return Distribution(kind, spec, stats.gamma(a=float(spec["shape"]), scale=1.0 / float(spec["rate"])))
    if kind == "normal":
        return Distribution(kind, spec, stats.norm(loc=float(spec["mean"]), scale=float(spec["sd"])))
    raise ValueError(f"unknown distribution type: {kind!r}")


def _check_order(lo, mo, hi):
    if not (lo <= mo <= hi and lo < hi):
        raise ValueError(f"require low <= mode <= high and low < high, got {lo}, {mo}, {hi}")


def lognormal_multiplier(z, sigma):
    """Lognormal factor with mean exactly 1: exp(sigma*z - sigma^2/2)."""
    return np.exp(sigma * np.asarray(z) - 0.5 * sigma**2)


def lognormal_from_median(median, sigma, u):
    """Per-event lognormal draw around a (possibly array-valued) median."""
    u = np.clip(u, _EPS, 1.0 - _EPS)
    return np.asarray(median) * np.exp(sigma * stats.norm.ppf(u))


def beta_from_mean(mean, concentration, u):
    """Per-event Beta draw with the given (array) mean and a+b = concentration."""
    mean = np.clip(np.asarray(mean, dtype=float), 1e-9, 1 - 1e-9)
    a = mean * concentration
    b = (1.0 - mean) * concentration
    return stats.beta.ppf(np.clip(u, _EPS, 1 - _EPS), a, b)


def latin_hypercube(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """n x d stratified uniforms: each column has one point in each of n strata."""
    u = (rng.random((n, d)) + np.arange(n)[:, None]) / n
    for j in range(d):
        u[:, j] = u[rng.permutation(n), j]
    return u
