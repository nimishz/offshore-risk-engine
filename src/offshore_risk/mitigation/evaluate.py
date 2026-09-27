"""Measure risk reduction of controls by re-simulation with common random numbers."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from ..config import ModelConfig
from ..financial.loss_model import capital_recovery_factor
from ..financial.metrics import expected_shortfall
from ..simulation.engine import SimulationResult, SimulationSettings, Simulator


def annualised_cost(m: dict, discount_rate: float) -> float:
    return float(m["capex_usd"]) * capital_recovery_factor(discount_rate, float(m["life_years"])) + float(m.get("opex_usd_per_year", 0.0))


def _stats(L: np.ndarray) -> dict:
    return {"eal": float(L.mean()), "p95": float(np.quantile(L, 0.95)), "p99": float(np.quantile(L, 0.99)),
            "es99": expected_shortfall(L, 0.99)}


def compare(base: SimulationResult, res: SimulationResult) -> dict:
    """Risk reduction (baseline - residual) with the paired standard error of the EAL reduction."""
    b, r = _stats(base.total), _stats(res.total)
    diff = base.total - res.total
    out = {f"baseline_{k}": v for k, v in b.items()}
    out.update({f"residual_{k}": v for k, v in r.items()})
    out.update({f"reduction_{k}": b[k] - r[k] for k in b})
    out["reduction_eal_se"] = float(diff.std(ddof=1) / np.sqrt(len(diff)))
    out["reduction_downtime_days"] = float(base.downtime_days.mean() - res.downtime_days.mean())
    out["residual_pfd_fire_protection_PPA"] = float(res.derived_means["pfd_fire_protection"][1])
    out["residual_power_loss_rate"] = float(res.derived_means["rate_power_loss"])
    return out


def evaluate_mitigations(cfg: ModelConfig, keys: Iterable[str] | None = None, n_years: int = 20_000,
                         seed: int = 20260927, dependence: str = "correlated",
                         baseline: SimulationResult | None = None) -> tuple[pd.DataFrame, dict]:
    keys = list(keys) if keys is not None else list(cfg.mitigations)
    st = SimulationSettings(n_years=n_years, seed=seed, dependence=dependence)
    base = baseline if baseline is not None else Simulator(cfg).run(st)
    dr = float(cfg.financial.get("discount_rate", 0.08))
    rows, results = [], {"baseline": base}
    for k in keys:
        m = cfg.mitigations[k]
        res = Simulator(cfg, [k]).run(st)
        results[k] = res
        c = compare(base, res)
        ann = annualised_cost(m, dr)
        rows.append({
            "mitigation": k, "name": m["name"], "capex_usd": float(m["capex_usd"]),
            "opex_usd_per_year": float(m.get("opex_usd_per_year", 0.0)), "life_years": float(m["life_years"]),
            "annualised_cost_usd": ann, **c,
            "cost_per_usd_eal_reduction": ann / c["reduction_eal"] if c["reduction_eal"] > 0 else np.inf,
            "benefit_cost_ratio": c["reduction_eal"] / ann if ann > 0 else np.inf,
        })
    df = pd.DataFrame(rows).sort_values("reduction_eal", ascending=False).reset_index(drop=True)
    return df, results
