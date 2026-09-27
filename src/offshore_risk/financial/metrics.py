"""Risk metrics for a simulated annual loss sample.

Conventions: losses are positive numbers; VaR_a is the a-quantile of the
annual loss (a loss level, not a deviation from the mean); ES_a (a.k.a. TVaR,
CVaR) is the mean loss in years at or above VaR_a.
"""
from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import pandas as pd

PERCENTILES = (0.50, 0.75, 0.90, 0.95, 0.99)

METRIC_NOTES = {
    "eal": "Expected annual loss: long-run average; the right basis for budgeting and cost-benefit of controls.",
    "median": "A typical year. Far below the EAL for skewed losses: most years are cheap, a few are very expensive.",
    "p75": "Loss exceeded one year in four.",
    "p90": "Loss exceeded one year in ten.",
    "p95": "Loss exceeded one year in twenty; a common planning / appetite anchor.",
    "p99": "Loss exceeded one year in a hundred.",
    "var95": "Value at Risk at 95 %: same number as P95, stated as a capital/contingency level.",
    "var99": "Value at Risk at 99 %.",
    "es95": "Expected shortfall at 95 %: average of the worst 5 % of years; captures how bad the tail is, not just where it starts.",
    "es99": "Expected shortfall at 99 %: average of the worst 1 % of years; sensitive to major-accident severity.",
    "max": "Largest simulated year: depends mostly on sample size; an indicator, not a risk measure.",
    "sd": "Standard deviation of annual loss (volatility).",
    "se_eal": "Monte Carlo standard error of the EAL estimate (s / sqrt(N)).",
}


def var(losses, alpha: float) -> float:
    return float(np.quantile(np.asarray(losses), alpha))


def expected_shortfall(losses, alpha: float) -> float:
    """Mean of losses at or above VaR_alpha (empirical tail mean)."""
    x = np.sort(np.asarray(losses, dtype=float))
    k = int(np.floor(alpha * len(x)))
    k = min(k, len(x) - 1)
    return float(x[k:].mean())


def exceedance_probability(losses, thresholds: Iterable[float]) -> dict[float, float]:
    x = np.asarray(losses)
    return {float(t): float((x > t).mean()) for t in thresholds}


def risk_metrics(losses) -> dict[str, float]:
    x = np.asarray(losses, dtype=float)
    out = {"eal": float(x.mean()), "sd": float(x.std(ddof=1)), "se_eal": float(x.std(ddof=1) / np.sqrt(len(x)))}
    for p in PERCENTILES:
        key = "median" if p == 0.5 else f"p{int(round(p * 100))}"
        out[key] = var(x, p)
    out["var95"], out["var99"] = out["p95"], out["p99"]
    out["es95"] = expected_shortfall(x, 0.95)
    out["es99"] = expected_shortfall(x, 0.99)
    out["max"] = float(x.max())
    out["n"] = int(len(x))
    return out


def bootstrap_ci(losses, stat, n_boot: int = 400, level: float = 0.9, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap interval for a statistic of the loss sample."""
    x = np.asarray(losses)
    rng = np.random.default_rng(seed)
    vals = np.array([stat(x[rng.integers(0, len(x), len(x))]) for _ in range(n_boot)])
    return float(np.quantile(vals, (1 - level) / 2)), float(np.quantile(vals, (1 + level) / 2))


def exceedance_curve(losses, grid: np.ndarray | None = None, n_points: int = 200) -> pd.DataFrame:
    """Loss exceedance curve P(L > x) on a (log-spaced) grid."""
    x = np.asarray(losses, dtype=float)
    if grid is None:
        pos = x[x > 0]
        lo = max(np.quantile(pos, 0.001), 1.0) if len(pos) else 1.0
        grid = np.geomspace(lo, max(x.max(), lo * 10), n_points)
    xs = np.sort(x)
    p = 1.0 - np.searchsorted(xs, grid, side="right") / len(xs)
    return pd.DataFrame({"loss": grid, "exceedance_probability": p, "return_period_years": np.where(p > 0, 1 / np.maximum(p, 1e-300), np.inf)})


def tail_allocation(loss_by_source: np.ndarray, sources: list[str], alpha: float = 0.99) -> pd.DataFrame:
    """Euler (co-TVaR) allocation: E[L_s | L >= VaR_alpha(L)]; sums to ES_alpha of the total.

    Compared with each source's share of the mean, it shows which sources drive
    the tail rather than the average year.
    """
    M = np.asarray(loss_by_source, dtype=float)
    total = M.sum(axis=1)
    order = np.argsort(total)
    k = min(int(np.floor(alpha * len(total))), len(total) - 1)
    tail = order[k:]
    mean_share = M.mean(axis=0) / total.mean()
    es_contrib = M[tail].mean(axis=0)
    df = pd.DataFrame(
        {
            "source": sources,
            "mean_loss": M.mean(axis=0),
            "share_of_eal": mean_share,
            f"es{int(alpha * 100)}_contribution": es_contrib,
            f"share_of_es{int(alpha * 100)}": es_contrib / es_contrib.sum() if es_contrib.sum() > 0 else 0.0,
        }
    )
    return df[df["mean_loss"] > 0].sort_values("mean_loss", ascending=False).reset_index(drop=True)


def metrics_table(results: Mapping[str, np.ndarray], scale: float = 1e6) -> pd.DataFrame:
    """Side-by-side metrics for several named loss samples (in units of `scale`)."""
    rows = {name: risk_metrics(x) for name, x in results.items()}
    df = pd.DataFrame(rows).T
    money = [c for c in df.columns if c not in ("n",)]
    df[money] = df[money] / scale
    return df
