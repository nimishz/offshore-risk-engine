from itertools import combinations

import numpy as np
import pytest

from offshore_risk.mitigation.controls import apply_effect
from offshore_risk.optimization.portfolio import exhaustive_search, feasible_subsets, solve_milp


def brute(costs, a, b, budget):
    def val(c):
        return sum(a[i] for i in c) + sum(b[i, j] for i, j in combinations(sorted(c), 2))
    return exhaustive_search(costs, val, budget)


@pytest.mark.parametrize("seed", range(8))
def test_milp_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    n = 7
    costs = rng.uniform(0.5, 3.0, n)
    a = rng.uniform(0.1, 2.0, n)
    b = np.triu(rng.uniform(-0.5, 0.1, (n, n)), 1)
    b = b + b.T
    budget = 5.0
    x, obj = solve_milp(costs, a, b, budget)
    best, best_v, _ = brute(list(costs), a, b, budget)
    assert obj == pytest.approx(best_v, rel=1e-7)
    assert costs[x == 1].sum() <= budget + 1e-9


def test_budget_zero_selects_nothing():
    x, obj = solve_milp([1.0, 2.0], [5.0, 5.0], None, 0.0)
    assert x.sum() == 0 and obj == 0


def test_additive_case_is_knapsack():
    x, obj = solve_milp([3, 2, 2], [4, 3, 3], None, 4)
    assert list(x) == [0, 1, 1] and obj == pytest.approx(6)


def test_feasible_subsets_respect_budget():
    costs = [1, 2, 3]
    subs = list(feasible_subsets(costs, 3))
    assert () in subs and (0, 1) in subs and (2,) in subs and (1, 2) not in subs


def test_effect_ops():
    x = np.array([0.2, 0.5])
    assert np.allclose(apply_effect(x, "reduce", 0.5), [0.1, 0.25])
    assert np.allclose(apply_effect(x, "increase", 0.5), [0.3, 0.75])
    assert np.allclose(apply_effect(x, "close_gap", 0.5), [0.6, 0.75])
    with pytest.raises(ValueError):
        apply_effect(x, "bogus", 1)


def test_optimizer_small_end_to_end(cfg):
    from offshore_risk.optimization.portfolio import PortfolioOptimizer

    opt = PortfolioOptimizer(cfg, keys=["ptw_competence", "inspection_programme", "emergency_response"], n_years=3000, seed=2)
    res = opt.optimise(budget=1.0e6, objective="eal")
    assert res["capex"] <= 1.0e6
    best, best_v, _ = opt.exhaustive(budget=1.0e6, objective="eal")
    assert res["simulated_value"] == pytest.approx(best_v, rel=0.05) or sorted(best) == sorted(res["selected"])
