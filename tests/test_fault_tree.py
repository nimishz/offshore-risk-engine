import numpy as np
import pytest

from offshore_risk.fault_tree import AND, KOFN, OR, BasicEvent, FaultTree
from offshore_risk.fault_tree.protection import fire_protection_basic_events, fire_protection_tree, firewater_subtree

A, B, Cc, D, P = (BasicEvent(x) for x in "ABCDP")


def shared_power_tree():
    """Two pumps, each unavailable on own failure OR shared power loss; both needed to fail."""
    return FaultTree(AND("BOTH", OR("PA", A, P), OR("PB", B, P)))


def test_simple_gates():
    probs = {"A": 0.1, "B": 0.2, "C": 0.3}
    assert FaultTree(AND("T", A, B)).probability(probs) == pytest.approx(0.02)
    assert FaultTree(OR("T", A, B)).probability(probs) == pytest.approx(1 - 0.9 * 0.8)
    assert FaultTree(KOFN("T", 2, A, B, Cc)).probability(probs) == pytest.approx(
        0.1 * 0.2 * 0.7 + 0.1 * 0.8 * 0.3 + 0.9 * 0.2 * 0.3 + 0.1 * 0.2 * 0.3
    )


def test_repeated_event_detected_and_naive_is_wrong():
    ft = shared_power_tree()
    assert ft.repeated_events == ["P"]
    p = {"A": 0.02, "B": 0.02, "P": 0.05}
    exact = ft.probability(p, "exact")
    enum = ft.probability(p, "enumeration")
    naive = ft.probability(p, "independent")
    assert exact == pytest.approx(enum, rel=1e-12)
    assert exact == pytest.approx(0.05 + 0.95 * 0.02 * 0.02)
    # naive multiplication squares the shared cause: order-of-magnitude underestimate
    assert naive < exact / 5


@pytest.mark.parametrize("seed", range(5))
def test_exact_matches_enumeration_random_trees(seed):
    rng = np.random.default_rng(seed)
    ev = [BasicEvent(f"E{i}") for i in range(7)]
    t = OR(
        "T", AND("G1", ev[0], OR("G2", ev[1], ev[2], ev[6])), KOFN("G3", 2, ev[2], ev[3], ev[4]), AND("G4", ev[5], ev[6], ev[0])
    )
    ft = FaultTree(t)
    p = {e.name: rng.uniform(0.01, 0.5) for e in ev}
    assert ft.probability(p, "exact") == pytest.approx(ft.probability(p, "enumeration"), rel=1e-12)


def test_minimal_cut_sets_and_bounds():
    ft = shared_power_tree()
    mcs = ft.minimal_cut_sets()
    assert set(map(frozenset, mcs)) == {frozenset({"P"}), frozenset({"A", "B"})}
    p = {"A": 0.02, "B": 0.02, "P": 0.05}
    exact = ft.probability(p)
    assert ft.probability(p, "rare_event") >= exact
    assert ft.probability(p, "mcub") >= exact - 1e-15
    assert ft.probability(p, "rare_event") == pytest.approx(exact, rel=0.01)


def test_kofn_cut_sets():
    ft = FaultTree(KOFN("T", 2, A, B, Cc))
    assert set(map(frozenset, ft.minimal_cut_sets())) == {frozenset("AB"), frozenset("AC"), frozenset("BC")}


def test_non_minimal_sets_removed():
    ft = FaultTree(OR("T", A, AND("G", A, B)))
    assert ft.minimal_cut_sets() == [frozenset({"A"})]


def test_vectorised_probabilities():
    ft = shared_power_tree()
    p = {"A": np.array([0.01, 0.1]), "B": 0.02, "P": np.array([0.0, 0.05])}
    out = ft.probability(p)
    assert out.shape == (2,)
    assert out[0] == pytest.approx(0.01 * 0.02)


def test_importance_measures():
    ft = shared_power_tree()
    imp = ft.importance({"A": 0.02, "B": 0.02, "P": 0.05}).set_index("basic_event")
    assert imp.loc["P", "fussell_vesely"] > imp.loc["A", "fussell_vesely"]
    assert imp.loc["P", "birnbaum"] == pytest.approx(1 - 0.02 * 0.02)


def test_missing_probability_raises():
    with pytest.raises(KeyError):
        shared_power_tree().probability({"A": 0.1})


def test_fire_protection_tree_structure(cfg):
    ft = fire_protection_tree(2, 1)
    assert set(ft.repeated_events) == {"POWER_LOSS", "CCF_ELEC_PUMPS"}
    th = cfg.registry.at_quantile(1, 0.5)
    be = fire_protection_basic_events(th, 0.1, 2, 1)
    exact = ft.probability(be, "exact")
    enum = ft.probability(be, "enumeration")
    assert float(exact[0]) == pytest.approx(float(enum[0]), rel=1e-10)
    fw = firewater_subtree(2, 1)
    assert float(fw.probability(be, "independent")[0]) < float(fw.probability(be, "exact")[0])


@pytest.mark.parametrize("ne,nd", [(1, 1), (2, 1), (3, 1), (3, 2), (0, 2), (2, 0)])
def test_architectures_have_unique_events_and_exact_evaluation(cfg, ne, nd):
    """Any pump count gives distinct basic events; exact evaluation still matches enumeration."""
    ft = fire_protection_tree(ne, nd)
    th = cfg.registry.at_quantile(1, 0.5)
    be = fire_protection_basic_events(th, 0.1, ne, nd)
    assert set(be) >= set(ft.basic_events)
    n_pump_events = sum(1 for e in ft.basic_events if e.startswith(("EPUMP_", "DPUMP_")))
    assert n_pump_events == 3 * (ne + nd)
    if len(ft.basic_events) <= 18:
        assert float(ft.probability(be)[0]) == pytest.approx(float(ft.probability(be, "enumeration")[0]), rel=1e-10)


def test_no_pumps_rejected():
    with pytest.raises(ValueError):
        fire_protection_tree(0, 0)


def test_more_pumps_never_worse(cfg):
    th = cfg.registry.at_quantile(1, 0.5)

    def pfd(ne, nd):
        return float(firewater_subtree(ne, nd).probability(fire_protection_basic_events(th, 0.1, ne, nd))[0])

    assert pfd(2, 1) <= pfd(1, 1)
    assert pfd(3, 1) <= pfd(2, 1)
    assert pfd(2, 2) <= pfd(2, 1)


def test_basic_events_clipped(cfg):
    th = cfg.registry.at_quantile(1, 0.5)
    th["fw_header_fail"] = th["fw_header_fail"] * 1e6
    be = fire_protection_basic_events(th, 0.1, 2, 1)
    assert all((0 <= v).all() and (v <= 1).all() for v in be.values())


def test_extra_diesel_pump_helps_most_when_power_is_lost(cfg):
    th = cfg.registry.at_quantile(1, 0.5)

    def base(pp, nd):
        return float(firewater_subtree(2, nd).probability(fire_protection_basic_events(th, pp, 2, nd))[0])

    gain_ucp = base(0.6, 1) - base(0.6, 2)
    gain_other = base(0.1, 1) - base(0.1, 2)
    assert gain_ucp > gain_other > 0
