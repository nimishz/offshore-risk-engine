"""Hypothetical risk appetite framework.

Limits in config/risk_appetite.yaml are illustrative management thresholds for
the fictional operator. They are not industry standards or regulatory limits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..financial.metrics import risk_metrics

WITHIN, NEAR, OUTSIDE = "Within appetite", "Near threshold", "Outside appetite"


def exposure_metrics(result) -> dict[str, float]:
    """Metrics referenced by the appetite limits, from a SimulationResult.

    Safety-system metrics take the worst production platform.
    """
    m = risk_metrics(result.total)
    worst = {k: max(result.platform_value(k, p) for p in result.producers) for k in ("pfd_firewater", "pfd_fire_protection")}
    return {
        "eal": m["eal"],
        "p95": m["p95"],
        "p99": m["p99"],
        "downtime_p95_days": float(np.quantile(result.downtime_days, 0.95)),
        "max_event_p99": float(np.quantile(result.max_event_loss, 0.99)),
        **worst,
    }


def classify(value: float, limit: float, near_fraction: float) -> str:
    if value > limit:
        return OUTSIDE
    if value >= near_fraction * limit:
        return NEAR
    return WITHIN


APPETITE_METRICS = ("eal", "p95", "p99", "downtime_p95_days", "max_event_p99", "pfd_firewater", "pfd_fire_protection")


def evaluate_appetite(appetite_cfg: dict, metrics: dict) -> pd.DataFrame:
    near = float(appetite_cfg.get("near_threshold_fraction", 0.8))
    rows = []
    for key, lim in appetite_cfg["limits"].items():
        v = float(metrics[lim["metric"]])
        rows.append(
            {
                "limit_id": key,
                "description": lim["description"],
                "value": v,
                "limit": float(lim["limit"]),
                "utilisation": v / float(lim["limit"]),
                "unit": lim["unit"],
                "status": classify(v, float(lim["limit"]), near),
            }
        )
    return pd.DataFrame(rows)
