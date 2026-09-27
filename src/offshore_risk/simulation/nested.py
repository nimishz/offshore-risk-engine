"""Two-loop simulation separating epistemic from aleatory uncertainty.

Outer loop: N_out 'worlds', each a draw of the epistemic parameters theta
(Latin hypercube). Inner loop: N_in simulated years per world with theta held
fixed, so the spread inside a world is aleatory only. The spread of a metric
(EAL, P95, ...) *across* worlds is our uncertainty about that metric.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import ModelConfig
from ..distributions import latin_hypercube
from ..financial.metrics import expected_shortfall
from .engine import ScenarioSpec, SimulationSettings, Simulator


@dataclass
class NestedResult:
    worlds: pd.DataFrame  # one row per world: theta values + metrics
    lec_grid: np.ndarray
    lec: np.ndarray  # (n_out, len(grid)) exceedance probabilities per world
    param_names: list[str]
    n_inner: int

    def metric_summary(self, metrics=("eal", "p95", "p99", "es99")) -> pd.DataFrame:
        rows = []
        for m in metrics:
            v = self.worlds[m]
            rows.append(
                {
                    "metric": m,
                    "p05": v.quantile(0.05),
                    "median": v.median(),
                    "mean": v.mean(),
                    "p95": v.quantile(0.95),
                    "ratio_p95_p05": v.quantile(0.95) / max(v.quantile(0.05), 1e-9),
                }
            )
        return pd.DataFrame(rows)


def run_nested(
    cfg: ModelConfig,
    n_outer: int = 300,
    n_inner: int = 1000,
    seed: int = 7,
    dependence: str = "correlated",
    mitigations: Iterable[str] = (),
    scenario: ScenarioSpec | None = None,
    lec_grid=None,
) -> NestedResult:
    reg = cfg.registry
    rng = np.random.default_rng([seed, 999])
    theta_w = reg.sample(latin_hypercube(n_outer, len(reg), rng))
    theta_rep = {k: np.repeat(v, n_inner) for k, v in theta_w.items()}
    sim = Simulator(cfg, mitigations, scenario)
    res = sim.run(
        SimulationSettings(n_years=n_outer * n_inner, seed=seed, dependence=dependence, chunk_size=50_000), theta=theta_rep
    )
    L = res.total.reshape(n_outer, n_inner)
    dt = res.downtime_days.reshape(n_outer, n_inner)
    df = pd.DataFrame({k: v for k, v in theta_w.items()})
    df["eal"] = L.mean(axis=1)
    df["median_loss"] = np.median(L, axis=1)
    df["p95"] = np.quantile(L, 0.95, axis=1)
    df["p99"] = np.quantile(L, 0.99, axis=1)
    df["es99"] = [expected_shortfall(row, 0.99) for row in L]
    df["downtime_p95_days"] = np.quantile(dt, 0.95, axis=1)
    if lec_grid is None:
        lec_grid = np.geomspace(1e6, max(L.max(), 1e7), 120)
    Ls = np.sort(L, axis=1)
    lec = 1.0 - np.stack([np.searchsorted(row, lec_grid, side="right") for row in Ls]) / n_inner
    return NestedResult(worlds=df, lec_grid=lec_grid, lec=lec, param_names=reg.names, n_inner=n_inner)
