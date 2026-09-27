import numpy as np
import pytest

from offshore_risk.financial.loss_model import COMPONENTS
from offshore_risk.simulation import SOURCES, ScenarioSpec, SimulationSettings, Simulator, simulate
from offshore_risk.simulation.derived import derive
from offshore_risk.asset import AssetModel

N = 20_000


@pytest.fixture(scope="module")
def base(cfg):
    return simulate(cfg, n_years=N, seed=1)


def test_reproducible(cfg, base):
    again = simulate(cfg, n_years=N, seed=1)
    assert np.array_equal(base.total, again.total)
    other = simulate(cfg, n_years=N, seed=2)
    assert not np.array_equal(base.total, other.total)


def test_shapes_and_accounting(base):
    assert base.loss_by_source.shape == (N, len(SOURCES))
    assert base.loss_by_component.shape == (N, len(COMPONENTS))
    # env scenario component excluded by default, so the two decompositions agree
    comp_total = base.loss_by_component[:, :-1].sum(axis=1)
    assert np.allclose(comp_total, base.total)
    assert (base.total >= 0).all()
    assert (base.max_event_loss <= base.total + 1e-6).all()


def test_event_counts_match_rates(cfg):
    """Mean number of releases per year equals the mean rate (mixed Poisson with mean-1 driver)."""
    res = simulate(cfg, n_years=100_000, seed=3)
    rate = res.derived_means["rate_release"]
    counts = res.counts["releases"]
    assert counts.mean() == pytest.approx(rate, abs=4 * counts.std() / np.sqrt(len(counts)))


def test_frequency_only_mitigation_never_increases_loss_in_any_year(cfg, base):
    """Thinning with common random numbers: a pure frequency reduction removes events, never adds."""
    res = Simulator(cfg, ["ptw_competence"]).run(SimulationSettings(n_years=N, seed=1))
    assert (res.total <= base.total + 1e-6).all()
    assert res.total.mean() < base.total.mean()


def test_zero_rates_edge_case(cfg):
    sc = ScenarioSpec(overrides={p: {"op": "set", "value": 0.0} for p in (
        "rel_valve_rate", "rel_flange_seal_rate", "rel_corrosion_rate", "ptw_isolation_failure_prob", "overpressure_demand_rate",
        "comp_weibull_shape", "gen_lambda_per_h", "pipeline_rate_per_km_yr", "spurious_trip_rate", "weather_rate", "collision_rate")})
    # comp_weibull_shape = 0 would give t^0 = 1 - 1 = 0 failures, i.e. no compressor events
    res = Simulator(cfg, scenario=sc).run(SimulationSettings(n_years=2000, seed=4))
    assert res.total.sum() == 0.0


def test_forced_event_adds_exactly_its_loss(cfg, base):
    sc = ScenarioSpec(name="t", forced_events=[{"type": "compressor"}])
    res = Simulator(cfg, scenario=sc).run(SimulationSettings(n_years=N, seed=1))
    assert np.allclose(res.total - base.total, res.loss_by_source[:, SOURCES.index("scenario_event")])
    assert (res.loss_by_source[:, SOURCES.index("scenario_event")] > 0).all()


def test_dependence_models_same_frequencies(cfg):
    a = simulate(cfg, n_years=100_000, seed=5, dependence="independent")
    b = simulate(cfg, n_years=100_000, seed=5, dependence="correlated")
    for k in ("releases",):
        assert a.counts[k].mean() == pytest.approx(b.counts[k].mean(), rel=0.03)
    # co-movement raises the upper-middle of the distribution
    assert np.quantile(b.total, 0.95) > np.quantile(a.total, 0.95)


def test_median_mode_is_deterministic_in_theta(cfg):
    r = simulate(cfg, n_years=1000, seed=1, epistemic="median")
    assert r.derived_means["pfd_esd"] == pytest.approx(float(derive(cfg.registry.at_quantile(1), AssetModel.from_config(cfg.asset)).pfd_esd[0]))


def test_chunking_does_not_change_sample_size(cfg):
    r = Simulator(cfg).run(SimulationSettings(n_years=12_345, seed=1, chunk_size=5000))
    assert r.n_years == 12_345


def test_event_recording(cfg):
    r = Simulator(cfg).run(SimulationSettings(n_years=50, seed=1, record_events=True))
    assert r.events is not None and len(r.events) > 0
    by_year = r.events.groupby("year")["total_usd"].sum()
    assert np.allclose(by_year.reindex(range(50), fill_value=0).to_numpy(), r.total)


def test_unknown_mitigation(cfg):
    with pytest.raises(KeyError):
        Simulator(cfg, ["does_not_exist"])
