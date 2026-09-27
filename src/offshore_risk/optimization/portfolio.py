"""Budget-constrained selection of risk controls.

Formulation (binary x_i = implement control i):

    maximise   sum_i a_i x_i + sum_{i<j} b_ij y_ij
    subject to sum_i c_i x_i <= B
               y_ij <= x_i,  y_ij <= x_j,  y_ij >= x_i + x_j - 1   (y_ij = x_i x_j)
               x, y binary

a_i  = simulated stand-alone reduction of the chosen risk measure,
b_ij = simulated pairwise interaction  R(i and j) - a_i - a_j (usually < 0 when
       two controls protect against the same thing),
c_i  = capital cost; B = budget.

Objectives: 'eal' (expected annual loss reduction), 'es99' (tail reduction),
'net_benefit' (EAL reduction minus annualised cost of the controls).

Interactions of order > 2 are ignored by the MILP; ``simulate_portfolio``
re-simulates the chosen set so the prediction error can be reported, and
``exhaustive_search`` checks the optimum over all feasible portfolios.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from ..config import ModelConfig
from ..financial.metrics import expected_shortfall
from ..mitigation.evaluate import annualised_cost
from ..simulation.engine import SimulationSettings, Simulator

MEASURES = {
    "eal": lambda L: float(np.mean(L)),
    "es99": lambda L: expected_shortfall(L, 0.99),
    "p95": lambda L: float(np.quantile(L, 0.95)),
}


def solve_milp(costs: Sequence[float], a: Sequence[float], b: np.ndarray | None, budget: float):
    """Solve the quadratic knapsack above, linearised. Returns (x as 0/1 array, objective)."""
    c = np.asarray(costs, dtype=float)
    a = np.asarray(a, dtype=float)
    n = len(c)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)] if b is not None else []
    nv = n + len(pairs)
    obj = np.zeros(nv)
    obj[:n] = -a
    for k, (i, j) in enumerate(pairs):
        obj[n + k] = -b[i, j]
    rows, lo, hi = [], [], []
    r = np.zeros(nv)
    r[:n] = c
    rows.append(r)
    lo.append(-np.inf)
    hi.append(budget)
    for k, (i, j) in enumerate(pairs):
        for t in (i, j):  # y - x_t <= 0
            r = np.zeros(nv)
            r[n + k] = 1
            r[t] = -1
            rows.append(r)
            lo.append(-np.inf)
            hi.append(0)
        r = np.zeros(nv)
        r[i] = 1
        r[j] = 1
        r[n + k] = -1  # x_i + x_j - y <= 1
        rows.append(r)
        lo.append(-np.inf)
        hi.append(1)
    res = milp(obj, constraints=LinearConstraint(np.array(rows), lo, hi), integrality=np.ones(nv), bounds=Bounds(0, 1))
    if not res.success:
        raise RuntimeError(f"MILP failed: {res.message}")
    x = np.round(res.x[:n]).astype(int)
    return x, -float(res.fun)


def feasible_subsets(costs: Sequence[float], budget: float):
    n = len(costs)
    for r in range(n + 1):
        for combo in combinations(range(n), r):
            if sum(costs[i] for i in combo) <= budget + 1e-9:
                yield combo


def exhaustive_search(costs: Sequence[float], value: Callable[[tuple], float], budget: float):
    best, best_v = (), -np.inf
    table = []
    for combo in feasible_subsets(costs, budget):
        v = value(combo)
        table.append((combo, v))
        if v > best_v:
            best, best_v = combo, v
    return best, best_v, table


class PortfolioOptimizer:
    """Runs the simulations needed for a_i and b_ij, then solves the MILP."""

    def __init__(
        self,
        cfg: ModelConfig,
        keys: Sequence[str] | None = None,
        n_years: int = 20_000,
        seed: int = 20260927,
        dependence: str = "correlated",
    ):
        self.cfg = cfg
        self.keys = list(keys) if keys is not None else list(cfg.mitigations)
        self.st = SimulationSettings(n_years=n_years, seed=seed, dependence=dependence)
        self.costs = np.array([float(cfg.mitigations[k]["capex_usd"]) for k in self.keys])
        dr = float(cfg.financial.get("discount_rate", 0.08))
        self.annual_costs = np.array([annualised_cost(cfg.mitigations[k], dr) for k in self.keys])
        self._cache: dict[tuple, np.ndarray] = {}

    def losses(self, combo: tuple) -> np.ndarray:
        key = tuple(sorted(combo))
        if key not in self._cache:
            self._cache[key] = Simulator(self.cfg, [self.keys[i] for i in key]).run(self.st).total
        return self._cache[key]

    def reduction(self, combo: tuple, measure: str = "eal") -> float:
        f = MEASURES[measure]
        return f(self.losses(())) - f(self.losses(combo))

    def value(self, combo: tuple, objective: str) -> float:
        if objective == "net_benefit":
            return self.reduction(combo, "eal") - float(self.annual_costs[list(combo)].sum())
        return self.reduction(combo, objective)

    def coefficients(self, objective: str = "eal"):
        n = len(self.keys)
        a = np.array([self.value((i,), objective) for i in range(n)])
        b = np.zeros((n, n))
        for i, j in combinations(range(n), 2):
            b[i, j] = b[j, i] = self.value((i, j), objective) - a[i] - a[j]
        return a, b

    def optimise(self, budget: float | None = None, objective: str = "eal", interactions: bool = True) -> dict:
        budget = self.cfg.budget_usd if budget is None else budget
        a, b = self.coefficients(objective)
        x, pred = solve_milp(self.costs, a, b if interactions else None, budget)
        chosen = tuple(int(i) for i in np.flatnonzero(x))
        return {
            "objective": objective,
            "budget": budget,
            "selected": [self.keys[i] for i in chosen],
            "selected_idx": chosen,
            "capex": float(self.costs[list(chosen)].sum()),
            "predicted_value": pred,
            "simulated_value": self.value(chosen, objective),
            "simulated_eal_reduction": self.reduction(chosen, "eal"),
            "simulated_es99_reduction": self.reduction(chosen, "es99"),
            "annualised_cost": float(self.annual_costs[list(chosen)].sum()),
            "a": a,
            "b": b,
        }

    def exhaustive(self, budget: float | None = None, objective: str = "eal"):
        budget = self.cfg.budget_usd if budget is None else budget
        best, best_v, table = exhaustive_search(list(self.costs), lambda c: self.value(c, objective), budget)
        df = (
            pd.DataFrame(
                [
                    {
                        "portfolio": " + ".join(self.keys[i] for i in c) or "(none)",
                        "capex": float(self.costs[list(c)].sum()),
                        "value": v,
                    }
                    for c, v in table
                ]
            )
            .sort_values("value", ascending=False)
            .reset_index(drop=True)
        )
        return [self.keys[i] for i in best], best_v, df

    def coefficient_table(self, objective: str = "eal") -> pd.DataFrame:
        a, b = self.coefficients(objective)
        return pd.DataFrame(b, index=self.keys, columns=self.keys).assign(standalone=a)
