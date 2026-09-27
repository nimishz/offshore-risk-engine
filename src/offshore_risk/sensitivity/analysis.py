"""Which assumptions matter most?

Three complementary views:

1. **Tornado (one-at-a-time):** each uncertain epistemic parameter is moved to
   its P10 and P90 with all others at their medians; the metric is re-simulated
   with common random numbers, so the swing is the parameter's effect and not
   Monte Carlo noise. Local: ignores interactions.
2. **Rank correlation:** Spearman correlation between each parameter and the
   per-world metric of a nested run (all parameters vary together). Global,
   monotone relationships only.
3. **First-order variance-based index (given-data):** S_i = Var(E[Y|X_i]) / Var(Y)
   estimated by binning the nested sample on quantiles of X_i. A screening
   estimate; a value near the null level (B-1)/N is indistinguishable from zero.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ..config import ModelConfig
from ..financial.metrics import expected_shortfall
from ..simulation.engine import SimulationSettings, Simulator
from ..simulation.nested import NestedResult

_METRICS = {
    "eal": lambda L: float(L.mean()),
    "p95": lambda L: float(np.quantile(L, 0.95)),
    "p99": lambda L: float(np.quantile(L, 0.99)),
    "es99": lambda L: expected_shortfall(L, 0.99),
}


def tornado(cfg: ModelConfig, n_years: int = 20_000, seed: int = 20260927, params=None,
            low: float = 0.10, high: float = 0.90, metrics=("eal", "p99"), dependence: str = "correlated") -> pd.DataFrame:
    reg = cfg.registry
    params = list(params) if params is not None else reg.uncertain_names()
    sim = Simulator(cfg)
    st = SimulationSettings(n_years=n_years, seed=seed, dependence=dependence)
    base_L = sim.run(st, theta=reg.at_quantile(n_years, 0.5)).total
    base = {m: _METRICS[m](base_L) for m in metrics}
    rows = []
    for p in params:
        row = {"parameter": p, "unit": reg[p].unit, "definition": reg[p].spec.get("definition", ""),
               "value_low": reg[p].dist.quantile(low), "value_median": reg[p].dist.median(), "value_high": reg[p].dist.quantile(high)}
        for tag, q in (("low", low), ("high", high)):
            L = sim.run(st, theta=reg.at_quantile(n_years, 0.5, {p: q})).total
            for m in metrics:
                row[f"{m}_{tag}"] = _METRICS[m](L)
        for m in metrics:
            row[f"{m}_base"] = base[m]
            row[f"{m}_swing"] = abs(row[f"{m}_high"] - row[f"{m}_low"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(f"{metrics[0]}_swing", ascending=False).reset_index(drop=True)


def rank_correlation(nested: NestedResult, metric: str = "eal") -> pd.DataFrame:
    df = nested.worlds
    rows = []
    for p in nested.param_names:
        x = df[p].to_numpy()
        if np.ptp(x) == 0:
            continue
        rho, pval = stats.spearmanr(x, df[metric])
        rows.append({"parameter": p, "spearman_rho": float(rho), "abs_rho": abs(float(rho)), "p_value": float(pval)})
    return pd.DataFrame(rows).sort_values("abs_rho", ascending=False).reset_index(drop=True)


def first_order_indices(nested: NestedResult, metric: str = "eal", bins: int = 10) -> pd.DataFrame:
    df = nested.worlds
    y = df[metric].to_numpy()
    vy = y.var()
    n = len(y)
    null = (bins - 1) / n
    rows = []
    for p in nested.param_names:
        x = df[p].to_numpy()
        if np.ptp(x) == 0:
            continue
        edges = np.quantile(x, np.linspace(0, 1, bins + 1))
        idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, bins - 1)
        means = np.array([y[idx == b].mean() for b in range(bins) if np.any(idx == b)])
        counts = np.array([np.sum(idx == b) for b in range(bins) if np.any(idx == b)])
        s = float(np.sum(counts * (means - y.mean()) ** 2) / n / vy)
        rows.append({"parameter": p, "S1": s, "above_null": s > 2 * null})
    out = pd.DataFrame(rows).sort_values("S1", ascending=False).reset_index(drop=True)
    out.attrs["null_level"] = null
    return out


def importance_summary(tornado_df: pd.DataFrame, rank_df: pd.DataFrame, s1_df: pd.DataFrame, top: int = 12) -> pd.DataFrame:
    """Combine the three views; rank by average rank position."""
    t = tornado_df[["parameter", "eal_swing"]].copy()
    t["rank_tornado"] = t["eal_swing"].rank(ascending=False)
    r = rank_df[["parameter", "spearman_rho", "abs_rho"]].copy()
    r["rank_spearman"] = r["abs_rho"].rank(ascending=False)
    s = s1_df[["parameter", "S1"]].copy()
    s["rank_S1"] = s["S1"].rank(ascending=False)
    m = t.merge(r, on="parameter", how="outer").merge(s, on="parameter", how="outer")
    m["mean_rank"] = m[["rank_tornado", "rank_spearman", "rank_S1"]].mean(axis=1)
    return m.sort_values("mean_rank").head(top).reset_index(drop=True)
