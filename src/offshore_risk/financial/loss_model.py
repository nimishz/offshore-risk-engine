"""Translation of operational consequences into money.

Direct loss:   physical damage, equipment repair, emergency response, logistics, restart.
Indirect loss: business interruption (deferred/lost production).
Environmental: a physical proxy (bbl of liquid released) is always recorded; a
               monetary value is only produced under an explicit *scenario
               assumption* (``env_cost_per_bbl_scenario_usd``) and is excluded
               from totals unless the caller opts in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

COMPONENTS = [
    "property_damage",
    "equipment_repair",
    "emergency_response",
    "logistics",
    "restart",
    "business_interruption",
    "environmental_scenario",
]
DIRECT = ["property_damage", "equipment_repair", "emergency_response", "logistics", "restart"]
INDIRECT = ["business_interruption"]
SCENARIO_ONLY = ["environmental_scenario"]
C = {name: i for i, name in enumerate(COMPONENTS)}


@dataclass
class FinancialModel:
    logistics_day_rate: float
    restart_cost: float
    evacuation_cost: float
    weather_demanning_cost: float
    env_cost_per_bbl: float
    discount_rate: float

    @classmethod
    def from_config(cls, fin: dict) -> FinancialModel:
        return cls(
            logistics_day_rate=float(fin["logistics_day_rate_usd"]),
            restart_cost=float(fin["restart_cost_usd"]),
            evacuation_cost=float(fin["evacuation_cost_usd"]),
            weather_demanning_cost=float(fin["weather_demanning_cost_usd"]),
            env_cost_per_bbl=float(fin.get("env_cost_per_bbl_scenario_usd", 0.0)),
            discount_rate=float(fin.get("discount_rate", 0.08)),
        )

    @staticmethod
    def business_interruption(lost_bbl, price, value_fraction):
        """Lost production x commodity price x economic value fraction of a deferred barrel."""
        return np.asarray(lost_bbl) * np.asarray(price) * np.asarray(value_fraction)

    def empty(self, n: int) -> np.ndarray:
        return np.zeros((n, len(COMPONENTS)))


def capital_recovery_factor(rate: float, years: float) -> float:
    """Annualise a capital cost: CRF = r (1+r)^n / ((1+r)^n - 1)."""
    if rate == 0:
        return 1.0 / years
    g = (1 + rate) ** years
    return rate * g / (g - 1)
