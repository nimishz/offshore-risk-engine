import numpy as np
import pytest

from offshore_risk.asset import AssetModel, lost_production_bbl


@pytest.fixture(scope="module")
def asset(cfg):
    return AssetModel.from_config(cfg.asset)


def test_capacity_bounds_and_full(asset):
    cap = asset.capacity_table()
    assert cap[0] == pytest.approx(asset.total_production)
    assert cap[-1] == 0.0
    assert (cap >= 0).all() and (cap <= cap[0] + 1e-9).all()


def test_capacity_monotone_in_down_set(asset):
    cap = asset.capacity_table()
    for mask in range(len(cap)):
        for i in range(asset.n):
            assert cap[mask | (1 << i)] <= cap[mask] + 1e-9


def test_dependency_propagation(asset):
    # UCP (power) down stops everything that produces
    assert asset.capacity({"UCP"}) == 0.0
    # PPA down: loses PPA and half of PPB (export route)
    assert asset.capacity({"PPA"}) == pytest.approx(18000 * 0.5)
    # WHP down: each processing platform keeps 40 % of feed; PPB also sees PPA at 40 %
    a = asset.availability({"WHP"})
    assert a["PPA"] == pytest.approx(0.4)
    assert a["PPB"] == pytest.approx(0.4 * (1 - 0.5 * 0.6))


def test_topological_order_respects_dependencies(asset):
    order = asset.topological_order()
    for d in asset.dependencies:
        assert order.index(d["provider"]) < order.index(d["platform"])


def test_neighbours(asset):
    assert set(asset.neighbours("UCP")) == {"PPB", "LQ", "SWI"}
    nb = asset.neighbour_matrix(3)
    assert nb.shape == (6, 3)


def test_lost_production_integration(asset):
    cap = asset.capacity_table()
    i = asset.index
    D = np.zeros((2, 6))
    D[0, i("PPA")] = 10.0                      # PPA down 10 days
    D[1, i("PPA")] = 10.0
    D[1, i("UCP")] = 4.0                       # UCP down 4 days as well
    lost = lost_production_bbl(D, cap)
    q = cap[0]
    assert lost[0] == pytest.approx(10 * (q - asset.capacity({"PPA"})))
    assert lost[1] == pytest.approx(4 * q + 6 * (q - asset.capacity({"PPA"})))
    assert lost_production_bbl(np.zeros((1, 6)), cap)[0] == 0.0
