import numpy as np
import pytest

from offshore_risk.event_tree import OUTCOMES, RELEASE_TREE, EventTree, Split

P = {"p_detect": 0.8, "p_isolate": 0.99, "p_ign": 0.1, "p_ign_iso": 0.05, "p_exp": 0.2, "p_fp_ok": 0.97}


def test_outcomes_sum_to_one():
    op = RELEASE_TREE.outcome_probabilities(P)
    assert sum(op.values()) == pytest.approx(1.0)
    assert set(op) == set(OUTCOMES)


def test_hand_computed_paths():
    op = RELEASE_TREE.outcome_probabilities(P)
    iso = 0.8 * 0.99
    uni = 1 - iso
    assert op["controlled_release"] == pytest.approx(iso * 0.95)
    assert op["uncontrolled_release"] == pytest.approx(uni * 0.9)
    assert op["minor_fire"] == pytest.approx(iso * 0.05 * 0.8 * 0.97)
    expected_cat = iso * 0.05 * 0.2 * 0.03 + uni * 0.1 * (0.2 * 0.03 + 0.8 * 0.03)
    assert op["catastrophic"] == pytest.approx(expected_cat)


def test_vectorised_and_sampling_frequencies():
    n = 400_000
    rng = np.random.default_rng(0)
    params = {k: np.full(n, v) for k, v in P.items()}
    leaves = RELEASE_TREE.sample_leaves(params, rng.random(n))
    out = RELEASE_TREE.leaf_outcome_index(leaves)
    op = RELEASE_TREE.outcome_probabilities(P)
    for j, o in enumerate(OUTCOMES):
        freq = (out == j).mean()
        assert freq == pytest.approx(op[o], abs=4 * np.sqrt(op[o] * (1 - op[o]) / n) + 1e-6)


def test_severity_ordered_sampling_is_monotone():
    """Better fire protection never maps the same random number to a worse outcome."""
    u = np.random.default_rng(1).random(50_000)
    worse = {**P, "p_fp_ok": 0.90}
    better = {**P, "p_fp_ok": 0.99}
    o_w = RELEASE_TREE.leaf_outcome_index(RELEASE_TREE.sample_leaves({k: np.full(len(u), v) for k, v in worse.items()}, u))
    o_b = RELEASE_TREE.leaf_outcome_index(RELEASE_TREE.sample_leaves({k: np.full(len(u), v) for k, v in better.items()}, u))
    # catastrophic outcomes can only decrease
    assert (o_b == OUTCOMES.index("catastrophic")).sum() <= (o_w == OUTCOMES.index("catastrophic")).sum()


def test_isolated_attribute():
    iso = RELEASE_TREE.leaf_attribute("Isolated")
    outs = [lf.outcome for lf in RELEASE_TREE.leaves]
    for flag, o in zip(iso, outs):
        if o in ("controlled_release", "minor_fire"):
            assert flag
        if o == "uncontrolled_release":
            assert not flag


def test_degenerate_probabilities():
    op = RELEASE_TREE.outcome_probabilities({**P, "p_ign": 0.0, "p_ign_iso": 0.0})
    assert op["controlled_release"] + op["uncontrolled_release"] == pytest.approx(1.0)
    op = RELEASE_TREE.outcome_probabilities(
        {**P, "p_detect": 1.0, "p_isolate": 1.0, "p_ign_iso": 1.0, "p_exp": 0.0, "p_fp_ok": 1.0}
    )
    assert op["minor_fire"] == pytest.approx(1.0)


def test_unknown_outcome_rejected():
    with pytest.raises(ValueError):
        EventTree(Split("q", 0.5, "a", "b"), ["a"])
