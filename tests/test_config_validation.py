import copy

import pytest

from offshore_risk import load_config
from offshore_risk.config_validation import ConfigError, validate_config


def broken(cfg, mutate):
    c = copy.deepcopy(cfg)
    mutate(c)
    with pytest.raises(ConfigError) as e:
        validate_config(c)
    return e.value.problems


def test_shipped_config_is_valid(cfg):
    validate_config(cfg)


def test_missing_config_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope")


def test_probability_above_one(cfg):
    probs = broken(
        cfg,
        lambda c: setattr(
            c,
            "registry",
            c.registry.with_distribution("p_ign_large", {"type": "triangular", "low": 0.5, "mode": 0.9, "high": 1.4}),
        ),
    )
    assert any("p_ign_large" in p and "exceed 1" in p for p in probs)


def test_negative_values(cfg):
    probs = broken(
        cfg,
        lambda c: setattr(
            c, "registry", c.registry.with_distribution("weather_rate", {"type": "normal", "mean": 0.4, "sd": 0.3})
        ),
    )
    assert any("weather_rate" in p and "negative" in p for p in probs)


def test_bad_basis(cfg):
    def m(c):
        c.registry.params["weather_rate"].spec["basis"] = "vibes"

    assert any("basis" in p for p in broken(cfg, m))


def test_cyclic_dependency(cfg):
    def m(c):
        c.asset["dependencies"].append({"platform": "UCP", "provider": "PPA", "service": "x", "impact": 0.5})

    assert any("cycle" in p for p in broken(cfg, m))


def test_unknown_platform_and_impact_range(cfg):
    def m(c):
        c.asset["dependencies"].append({"platform": "XYZ", "provider": "PPA", "impact": 0.5})
        c.asset["dependencies"][0]["impact"] = 1.5

    probs = broken(cfg, m)
    assert any("unknown platform" in p for p in probs)
    assert any("impact" in p for p in probs)


def test_mitigation_references(cfg):
    def m(c):
        c.mitigations["ptw_competence"]["effects"][0]["param"] = "typo_param"
        c.mitigations["deluge_coverage"]["effects"][0]["op"] = "explode"
        c.mitigations["firewater_pump"]["architecture"] = {"pumps": 3}

    probs = broken(cfg, m)
    assert any("typo_param" in p for p in probs)
    assert any("explode" in p for p in probs)
    assert any("'pumps'" in p for p in probs)


def test_scenario_references(cfg):
    def m(c):
        c.scenarios["S4_multi_platform_shutdown"]["forced_events"][0]["escalate_to"] = ["LQ"]
        c.scenarios["S3_major_hydrocarbon_release"]["forced_events"][0]["size"] = "huge"
        c.scenarios["S1_single_equipment_failure"]["forced_events"].append({"type": "meteor"})

    probs = broken(cfg, m)
    assert any("not a bridge neighbour" in p for p in probs)
    assert any("huge" in p for p in probs)
    assert any("meteor" in p for p in probs)


def test_driver_correlation_checks(cfg):
    def m(c):
        c.drivers["correlation"] = [[1, 0.99, -0.99], [0.99, 1, 0.99], [-0.99, 0.99, 1]]

    assert any("positive definite" in p for p in broken(cfg, m))


def test_appetite_metric_names(cfg):
    def m(c):
        c.appetite["limits"]["expected_annual_loss"]["metric"] = "eel"

    assert any("eel" in p for p in broken(cfg, m))


def test_all_problems_reported_together(cfg):
    def m(c):
        c.asset["bridges"].append(["PPA", "PPA"])
        c.appetite["near_threshold_fraction"] = 2

    assert len(broken(cfg, m)) >= 2
