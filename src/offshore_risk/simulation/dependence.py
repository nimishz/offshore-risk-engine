"""Year-level drivers and the two dependence models.

Model A ('independent'): drivers are independent.
Model B ('correlated'):  drivers are joined by a Gaussian copula.

Both use identical lognormal marginals with mean 1, so each risk's expected
frequency is the same under A and B; only co-movement differs.
"""

from __future__ import annotations

import numpy as np

from ..distributions import lognormal_multiplier

MODELS = ("independent", "correlated")


def driver_correlation(drivers_cfg: dict) -> np.ndarray:
    c = np.asarray(drivers_cfg["correlation"], dtype=float)
    if not np.allclose(c, c.T):
        raise ValueError("driver correlation matrix must be symmetric")
    if np.min(np.linalg.eigvalsh(c)) <= 0:
        raise ValueError("driver correlation matrix must be positive definite")
    return c


def sample_drivers(n: int, rng: np.random.Generator, drivers_cfg: dict, model: str) -> dict[str, np.ndarray]:
    if model not in MODELS:
        raise ValueError(f"dependence model must be one of {MODELS}")
    names = drivers_cfg["names"]
    z = rng.standard_normal((n, len(names)))
    if model == "correlated":
        z = z @ np.linalg.cholesky(driver_correlation(drivers_cfg)).T
    return {nm: lognormal_multiplier(z[:, j], float(drivers_cfg["sigma"][nm])) for j, nm in enumerate(names)}
