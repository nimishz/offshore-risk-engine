"""Hypothetical risk appetite framework.

Limits in config/risk_appetite.yaml are illustrative management thresholds for
the fictional operator. They are not industry standards or regulatory limits.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..financial.metrics import risk_metrics

WITHIN, NEAR, OUTSIDE = "Within appetite", "Near threshold", "Outside appetite"


def exposure_metrics(result, production_platforms=(1, 2)) -> dict[str, float]:
    """Metrics referenced by the appetite limits, from a SimulationResult.

    Safety-system metrics take the worst production platform (index 1 = PPA, 2 = PPB).
    """
    m = risk_metrics(result.total)
    dm = result.derived_means
    return {
        "eal": m["eal"],
        "p95": m["p95"],
        "p99": m["p99"],
        "downtime_p95_days": float(np.quantile(result.downtime_days, 0.95)),
        "max_event_p99": float(np.quantile(result.max_event_loss, 0.99)),
        "pfd_firewater": float(max(dm["pfd_firewater"][i] for i in production_platforms)),
        "pfd_fire_protection": float(max(dm["pfd_fire_protection"][i] for i in production_platforms)),
    }


def classify(value: float, limit: float, near_fraction: float) -> str:
    if value > limit:
        return OUTSIDE
    if value >= near_fraction * limit:
        return NEAR
    return WITHIN


def evaluate_appetite(appetite_cfg: dict, metrics: dict) -> pd.DataFrame:
    near = float(appetite_cfg.get("near_threshold_fraction", 0.8))
    rows = []
    for key, lim in appetite_cfg["limits"].items():
        v = float(metrics[lim["metric"]])
        rows.append({
            "limit_id": key,
            "description": lim["description"],
            "value": v,
            "limit": float(lim["limit"]),
            "utilisation": v / float(lim["limit"]),
            "unit": lim["unit"],
            "status": classify(v, float(lim["limit"]), near),
        })
    return pd.DataFrame(rows)
