"""Fault trees for the complex's protection layers, built from the asset architecture.

The fire-protection tree is drawn the way an engineer would draw it: each
electric firewater pump is 'unavailable' if it fails to start, is under
maintenance, fails to run, **or loses main power**, **or suffers a common-cause
failure**. The last two events therefore appear under every electric pump,
which is exactly the dependence that naive multiplication misses.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Mapping

import numpy as np

from ..reliability.models import beta_factor_split, pfd_koon
from .tree import AND, OR, BasicEvent, FaultTree


@lru_cache(maxsize=32)
def fire_protection_tree(n_electric: int = 2, n_diesel: int = 1) -> FaultTree:
    power = BasicEvent("POWER_LOSS", "Main power unavailable to electric firewater pumps")
    ccf_e = BasicEvent("CCF_ELEC_PUMPS", "Common-cause failure to start of electric pumps")
    ccf_d = BasicEvent("CCF_DIESEL_PUMPS", "Common-cause failure to start of diesel pumps")

    elec = []
    for i in range(n_electric):
        tag = chr(ord("A") + i)
        causes = [
            BasicEvent(f"EPUMP_{tag}_FTS", "fails to start (independent part)"),
            BasicEvent(f"EPUMP_{tag}_MAINT", "out of service for maintenance"),
            BasicEvent(f"EPUMP_{tag}_FTR", "fails to run for the mission time"),
            power,
        ]
        if n_electric > 1:
            causes.append(ccf_e)
        elec.append(OR(f"EPUMP_{tag}_UNAVAILABLE", *causes))
    diesel = []
    for j in range(n_diesel):
        tag = chr(ord("C") + j)
        causes = [
            BasicEvent(f"DPUMP_{tag}_FTS", "fails to start (independent part)"),
            BasicEvent(f"DPUMP_{tag}_MAINT", "out of service for maintenance"),
            BasicEvent(f"DPUMP_{tag}_FTR", "fails to run for the mission time"),
        ]
        if n_diesel > 1:
            causes.append(ccf_d)
        diesel.append(OR(f"DPUMP_{tag}_UNAVAILABLE", *causes))

    pump_groups = []
    if elec:
        pump_groups.append(elec[0] if len(elec) == 1 else AND("ALL_ELECTRIC_PUMPS_FAIL", *elec))
    if diesel:
        pump_groups.append(diesel[0] if len(diesel) == 1 else AND("ALL_DIESEL_PUMPS_FAIL", *diesel))
    pumps = pump_groups[0] if len(pump_groups) == 1 else AND("NO_FIREWATER_PUMP", *pump_groups)

    firewater = OR("FIREWATER_FAIL", BasicEvent("HEADER_FAIL", "ring main / header cannot deliver"), pumps)
    detection = AND(
        "NO_DELUGE_ACTIVATION",
        BasicEvent("FIRE_DETECTION_FAIL", "2oo3 flame detection + logic fails"),
        BasicEvent("MANUAL_ACTIVATION_FAIL", "operators do not activate manually"),
    )
    top = OR("FIRE_PROTECTION_FAIL", firewater, BasicEvent("DELUGE_VALVE_FAIL"), detection)
    return FaultTree(top, name="Fire protection fails on demand")


def firewater_subtree(n_electric: int = 2, n_diesel: int = 1) -> FaultTree:
    top = fire_protection_tree(n_electric, n_diesel).top
    fw = next(c for c in top.inputs if getattr(c, "name", "") == "FIREWATER_FAIL")
    return FaultTree(fw, name="Firewater fails on demand")


def fire_protection_basic_events(theta: Mapping, p_power_loss, n_electric: int, n_diesel: int) -> dict:
    """Basic-event probabilities for :func:`fire_protection_tree` from epistemic parameters.

    ``p_power_loss`` is the conditional probability that main power is lost
    given a fire at the location in question (it differs by platform).
    """
    T = theta["proof_test_interval_h"]
    ftr = -np.expm1(-theta["fw_pump_ftr_per_h"] * theta["fw_mission_time_h"])
    beta = theta["fw_elec_ccf_beta"]
    e_ind, e_ccf = beta_factor_split(theta["fw_elec_pump_fts"], beta)
    d_ind, d_ccf = beta_factor_split(theta["fw_diesel_pump_fts"], beta)
    p = {
        "POWER_LOSS": np.asarray(p_power_loss, dtype=float) * np.ones_like(T, dtype=float),
        "CCF_ELEC_PUMPS": e_ccf,
        "CCF_DIESEL_PUMPS": d_ccf,
        "HEADER_FAIL": theta["fw_header_fail"],
        "DELUGE_VALVE_FAIL": np.minimum(theta["deluge_lambda_du"] * T / 2.0, 1.0),
        "FIRE_DETECTION_FAIL": np.minimum(
            pfd_koon(2, 3, theta["fd_lambda_du"], T, theta["fd_beta"]) + theta["logic_solver_pfd"], 1.0
        ),
        "MANUAL_ACTIVATION_FAIL": theta["p_manual_activation_fail"],
    }
    for i in range(n_electric):
        tag = chr(ord("A") + i)
        p[f"EPUMP_{tag}_FTS"] = e_ind if n_electric > 1 else theta["fw_elec_pump_fts"]
        p[f"EPUMP_{tag}_MAINT"] = theta["fw_pump_maint_unavail"]
        p[f"EPUMP_{tag}_FTR"] = ftr
    for j in range(n_diesel):
        tag = chr(ord("C") + j)
        p[f"DPUMP_{tag}_FTS"] = d_ind if n_diesel > 1 else theta["fw_diesel_pump_fts"]
        p[f"DPUMP_{tag}_MAINT"] = theta["fw_pump_maint_unavail"]
        p[f"DPUMP_{tag}_FTR"] = ftr
    return p


def gas_detection_pfd(theta: Mapping, n: int = 3, k: int = 2):
    return pfd_koon(k, n, theta["gd_lambda_du"], theta["proof_test_interval_h"], theta["gd_beta"]) + theta[
        "logic_solver_pfd"
    ]


def esd_pfd(theta: Mapping, n_valves: int = 2):
    """Isolation fails if all valves fail to close (1ooN) or the logic solver fails."""
    return pfd_koon(1, n_valves, theta["esd_lambda_du"], theta["proof_test_interval_h"], theta["esd_beta"]) + theta[
        "logic_solver_pfd"
    ]
