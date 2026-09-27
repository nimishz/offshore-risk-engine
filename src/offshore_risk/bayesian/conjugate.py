"""Conjugate Bayesian updating for failure data.

Two data types appear in reliability work and each has a natural conjugate model:

* **Failures on demand** (k failures in n tests): Beta prior on the per-demand
  probability p, Binomial likelihood -> Beta posterior.
* **Failures in time** (k failures in T exposure-years): Gamma prior on the
  rate lambda, Poisson likelihood -> Gamma posterior.

Generic priors are usually quoted as a median with an error factor. They are
converted to conjugate priors by moment matching (``gamma_from_lognormal``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from ..distributions import Z95


@dataclass(frozen=True)
class BetaModel:
    a: float
    b: float

    @classmethod
    def from_mean(cls, mean: float, n: float) -> BetaModel:
        return cls(mean * n, (1 - mean) * n)

    def update(self, failures: int, demands: int) -> BetaModel:
        if failures > demands or failures < 0:
            raise ValueError("need 0 <= failures <= demands")
        return BetaModel(self.a + failures, self.b + demands - failures)

    @property
    def dist(self):
        return stats.beta(self.a, self.b)

    def mean(self) -> float:
        return self.a / (self.a + self.b)

    def interval(self, level: float = 0.9) -> tuple[float, float]:
        lo, hi = self.dist.ppf([(1 - level) / 2, (1 + level) / 2])
        return float(lo), float(hi)

    def pdf(self, x):
        return self.dist.pdf(x)

    @staticmethod
    def likelihood_density(x, failures: int, demands: int):
        """Binomial likelihood in p, normalised to integrate to 1 (i.e. Beta(k+1, n-k+1))."""
        return stats.beta(failures + 1, demands - failures + 1).pdf(x)

    def spec(self) -> dict:
        return {"type": "beta", "a": float(self.a), "b": float(self.b)}


@dataclass(frozen=True)
class GammaModel:
    shape: float
    rate: float  # in exposure units (e.g. years)

    def update(self, events: int, exposure: float) -> GammaModel:
        if events < 0 or exposure < 0:
            raise ValueError("events and exposure must be non-negative")
        return GammaModel(self.shape + events, self.rate + exposure)

    @property
    def dist(self):
        return stats.gamma(a=self.shape, scale=1.0 / self.rate)

    def mean(self) -> float:
        return self.shape / self.rate

    def interval(self, level: float = 0.9) -> tuple[float, float]:
        lo, hi = self.dist.ppf([(1 - level) / 2, (1 + level) / 2])
        return float(lo), float(hi)

    def pdf(self, x):
        return self.dist.pdf(x)

    @staticmethod
    def likelihood_density(x, events: int, exposure: float):
        """Poisson likelihood in lambda, normalised (i.e. Gamma(k+1, T))."""
        return stats.gamma(a=events + 1, scale=1.0 / exposure).pdf(x)

    def predictive(self, exposure: float = 1.0):
        """Posterior predictive count over `exposure`: negative binomial."""
        p = self.rate / (self.rate + exposure)
        return stats.nbinom(self.shape, p)

    def spec(self) -> dict:
        return {"type": "gamma", "shape": float(self.shape), "rate": float(self.rate)}


def gamma_from_lognormal(median: float, ef: float) -> GammaModel:
    """Moment-match a Gamma prior to a lognormal (median, error factor).

    Lognormal: mean m = median exp(s^2/2), variance (exp(s^2)-1) m^2.
    Gamma with the same mean and variance: shape = 1/(exp(s^2)-1), rate = shape/m.
    """
    s = np.log(ef) / Z95
    m = median * np.exp(s**2 / 2)
    shape = 1.0 / np.expm1(s**2)
    return GammaModel(float(shape), float(shape / m))


def sequential_update(prior, counts, exposures, labels=None, level: float = 0.9) -> pd.DataFrame:
    """Posterior after each period of data (cumulative). Works for Beta or Gamma models."""
    rows = [{"period": "prior", "mean": prior.mean(), **dict(zip(("lo", "hi"), prior.interval(level)))}]
    post = prior
    labels = labels if labels is not None else list(range(1, len(counts) + 1))
    for lab, k, e in zip(labels, counts, exposures):
        post = post.update(int(k), e)
        lo, hi = post.interval(level)
        rows.append({"period": lab, "mean": post.mean(), "lo": lo, "hi": hi})
    df = pd.DataFrame(rows)
    df["width"] = df["hi"] - df["lo"]
    return df


def grid_posterior(prior_pdf, likelihood, grid):
    """Numerical posterior on a grid (used to validate the conjugate results)."""
    w = prior_pdf(grid) * likelihood(grid)
    w = w / np.trapezoid(w, grid)
    return w
