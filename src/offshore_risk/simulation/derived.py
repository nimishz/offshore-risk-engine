"""Derive rates and protection-layer probabilities from epistemic parameters.

This is where the reliability and fault-tree modules feed the simulation:
theta (arrays, one entry per simulated world/year) -> event frequencies and
event-tree branch probabilities.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ..asset.network import AssetModel
from ..fault_tree.protection import (
    esd_pfd,
    fire_protection_basic_events,
    fire_protection_tree,
    firewater_subtree,
    gas_detection_pfd,
)
from ..reliability.models import nhpp_expected_failures, repairable_koon_failure_frequency

SIZES = ["small", "medium", "large"]


@dataclass
class Derived:
    release_rate: np.ndarray        # (n, n_platforms) per year
    release_causes: dict            # cause -> (n,) per platform-year at factor 1
    compressor_rate: np.ndarray     # expected train failures per year
    power_loss_rate: np.ndarray     # total power loss per year
    pipeline_rate: np.ndarray
    spurious_trip_rate: np.ndarray
    weather_rate: np.ndarray
    collision_rate: np.ndarray
    size_probs: np.ndarray          # (n, 3)
    p_detect: np.ndarray            # (n, 3)
    p_isolate: np.ndarray           # (n,)
    p_ign: np.ndarray               # (n, 3) unisolated
    p_exp: np.ndarray               # (n, 3) gas
    pfd_fp: np.ndarray              # (n, n_platforms) fire protection fails
    pfd_firewater: np.ndarray       # (n, n_platforms)
    pfd_gas_detection: np.ndarray
    pfd_esd: np.ndarray

    def rates(self) -> dict[str, np.ndarray]:
        return {
            "release": self.release_rate.sum(axis=1),
            "compressor": self.compressor_rate,
            "power_loss": self.power_loss_rate,
            "pipeline": self.pipeline_rate,
            "spurious_trip": self.spurious_trip_rate,
            "weather": self.weather_rate,
            "collision": self.collision_rate,
        }


def release_cause_rates(theta: Mapping) -> dict[str, np.ndarray]:
    """Cause rates per platform-year (equipment factor 1). OR gate on rates = sum."""
    return {
        "valve_leak": theta["rel_valve_rate"],
        "flange_seal": theta["rel_flange_seal_rate"],
        "corrosion": theta["rel_corrosion_rate"],
        # AND gate: jobs/yr x P(isolation failure | job) x P(release | failure)
        "maintenance_ptw": theta["ptw_jobs_per_year"] * theta["ptw_isolation_failure_prob"] * theta["p_release_given_ptw_failure"],
        # AND gate: demand rate x PFD of pressure protection
        "overpressure": theta["overpressure_demand_rate"] * theta["psv_pfd"],
    }


def derive(theta: Mapping, asset: AssetModel, architecture: Mapping | None = None) -> Derived:
    arch = dict(asset.protection)
    arch.update(architecture or {})
    hours = 8760.0

    causes = release_cause_rates(theta)
    per_unit = sum(causes.values())
    release_rate = per_unit[:, None] * asset.release_factor[None, :]

    comp = arch["compressors_installed"] * nhpp_expected_failures(
        theta["comp_age_years"], theta["comp_age_years"] + 1.0, theta["comp_weibull_shape"], theta["comp_weibull_scale_years"]
    )
    power = hours * repairable_koon_failure_frequency(
        arch["generators_required"], arch["generators_installed"], theta["gen_lambda_per_h"], theta["gen_mttr_h"], theta["gen_ccf_beta"]
    )

    p_large = theta["p_large_release"]
    p_med = theta["p_medium_release"] * np.ones_like(p_large)
    size_probs = np.stack([1.0 - p_large - p_med, p_med, p_large], axis=1)

    pfd_gd = gas_detection_pfd(theta, arch["gas_detectors_per_zone"], arch["gas_detection_voting_k"])
    cov = np.stack([theta["gd_coverage_small"], theta["gd_coverage_medium"], theta["gd_coverage_large"]], axis=1)
    p_detect = cov * (1.0 - pfd_gd)[:, None]
    pfd_e = esd_pfd(theta, arch["esd_valves_in_series"])

    ne, nd = int(arch["electric_firewater_pumps"]), int(arch["diesel_firewater_pumps"])
    fp_tree = fire_protection_tree(ne, nd)
    fw_tree = firewater_subtree(ne, nd)
    pfd_fp = np.zeros((len(p_large), asset.n))
    pfd_fw = np.zeros_like(pfd_fp)
    cache: dict[float, tuple] = {}
    for i, pp in enumerate(asset.p_power_loss_given_fire):
        if pp not in cache:
            be = fire_protection_basic_events(theta, pp, ne, nd)
            cache[pp] = (fp_tree.probability(be, "exact"), fw_tree.probability(be, "exact"))
        pfd_fp[:, i], pfd_fw[:, i] = cache[pp]

    return Derived(
        release_rate=release_rate,
        release_causes=causes,
        compressor_rate=np.asarray(comp, dtype=float),
        power_loss_rate=np.asarray(power, dtype=float),
        pipeline_rate=theta["pipeline_rate_per_km_yr"] * asset.pipeline_km,
        spurious_trip_rate=np.asarray(theta["spurious_trip_rate"], dtype=float),
        weather_rate=np.asarray(theta["weather_rate"], dtype=float),
        collision_rate=np.asarray(theta["collision_rate"], dtype=float),
        size_probs=size_probs,
        p_detect=p_detect,
        p_isolate=1.0 - pfd_e,
        p_ign=np.stack([theta["p_ign_small"], theta["p_ign_medium"], theta["p_ign_large"]], axis=1),
        p_exp=np.stack([theta["p_exp_small"], theta["p_exp_medium"], theta["p_exp_large"]], axis=1),
        pfd_fp=pfd_fp,
        pfd_firewater=pfd_fw,
        pfd_gas_detection=np.asarray(pfd_gd),
        pfd_esd=np.asarray(pfd_e),
    )
