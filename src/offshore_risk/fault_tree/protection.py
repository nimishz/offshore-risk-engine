"""Fault trees for the complex's protection layers, built from the asset architecture.

The fire-protection tree is drawn the way an engineer would draw it: each
electric firewater pump is 'unavailable' if it fails to start, is under
maintenance, fails to run, **or loses main power**, **or suffers a common-cause
failure**. The last two events therefore appear under every electric pump,
which is exactly the dependence that naive multiplication misses.

Basic-event names are numbered per pump group (``EPUMP_1``, ``EPUMP_2``,
``DPUMP_1`` ...) so that any architecture produces unique names.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache

import numpy as np

from ..reliability.models import beta_factor_split, pfd_koon
from .tree import AND, OR, BasicEvent, FaultTree


def _pump_group(prefix: str, n: int, shared: list[BasicEvent], ccf: BasicEvent, kind: str):
    """OR gate per pump (own failures + shared causes); AND across the pumps of the group."""
    pumps = []
    for i in range(1, n + 1):
        causes = [
            BasicEvent(f"{prefix}_{i}_FTS", f"{kind} pump {i} fails to start (independent part)"),
            BasicEvent(f"{prefix}_{i}_MAINT", f"{kind} pump {i} out of service for maintenance"),
            BasicEvent(f"{prefix}_{i}_FTR", f"{kind} pump {i} fails to run for the mission time"),
            *shared,
        ]
        if n > 1:
            causes.append(ccf)
        pumps.append(OR(f"{prefix}_{i}_UNAVAILABLE", *causes))
    if not pumps:
        return None
    return pumps[0] if n == 1 else AND(f"ALL_{kind.upper()}_PUMPS_FAIL", *pumps)


@lru_cache(maxsize=64)
def fire_protection_tree(n_electric: int = 2, n_diesel: int = 1) -> FaultTree:
    if n_electric < 0 or n_diesel < 0 or n_electric + n_diesel < 1:
        raise ValueError("the firewater system needs at least one pump")
    power = BasicEvent("POWER_LOSS", "Main power unavailable to electric firewater pumps")
    ccf_e = BasicEvent("CCF_ELEC_PUMPS", "Common-cause failure to start of the electric pumps")
    ccf_d = BasicEvent("CCF_DIESEL_PUMPS", "Common-cause failure to start of the diesel pumps")

    groups = [
        g
        for g in (
            _pump_group("EPUMP", n_electric, [power], ccf_e, "electric"),
            _pump_group("DPUMP", n_diesel, [], ccf_d, "diesel"),
        )
        if g is not None
    ]
    pumps = groups[0] if len(groups) == 1 else AND("NO_FIREWATER_PUMP", *groups)

    firewater = OR("FIREWATER_FAIL", BasicEvent("HEADER_FAIL", "ring main / header cannot deliver"), pumps)
    no_activation = AND(
        "NO_DELUGE_ACTIVATION",
        BasicEvent("FIRE_DETECTION_FAIL", "voted flame detection + logic solver fails"),
        BasicEvent("MANUAL_ACTIVATION_FAIL", "operators do not activate deluge manually"),
    )
    top = OR("FIRE_PROTECTION_FAIL", firewater, BasicEvent("DELUGE_VALVE_FAIL", "deluge valve set fails to open"), no_activation)
    return FaultTree(top, name="Fire protection fails on demand")


def firewater_subtree(n_electric: int = 2, n_diesel: int = 1) -> FaultTree:
    top = fire_protection_tree(n_electric, n_diesel).top
    fw = next(c for c in top.inputs if getattr(c, "name", "") == "FIREWATER_FAIL")
    return FaultTree(fw, name="Firewater fails on demand")


def _prob(x):
    return np.clip(np.asarray(x, dtype=float), 0.0, 1.0)


def fire_protection_basic_events(
    theta: Mapping,
    p_power_loss,
    n_electric: int,
    n_diesel: int,
    flame_detectors: int = 3,
    flame_voting_k: int = 2,
) -> dict:
    """Basic-event probabilities for :func:`fire_protection_tree` from epistemic parameters.

    ``p_power_loss`` is the conditional probability that main power is lost
    given a fire at the location in question (it differs by platform). All
    probabilities are clipped to [0, 1] so that scenario overrides cannot
    produce invalid inputs.
    """
    T = np.asarray(theta["proof_test_interval_h"], dtype=float)
    ftr = -np.expm1(-theta["fw_pump_ftr_per_h"] * theta["fw_mission_time_h"])
    e_ind, e_ccf = beta_factor_split(theta["fw_elec_pump_fts"], theta["fw_elec_ccf_beta"])
    d_ind, d_ccf = beta_factor_split(theta["fw_diesel_pump_fts"], theta["fw_diesel_ccf_beta"])
    p = {
        "POWER_LOSS": np.asarray(p_power_loss, dtype=float) * np.ones_like(T),
        "CCF_ELEC_PUMPS": e_ccf,
        "CCF_DIESEL_PUMPS": d_ccf,
        "HEADER_FAIL": theta["fw_header_fail"],
        "DELUGE_VALVE_FAIL": theta["deluge_lambda_du"] * T / 2.0,
        "FIRE_DETECTION_FAIL": pfd_koon(flame_voting_k, flame_detectors, theta["fd_lambda_du"], T, theta["fd_beta"])
        + theta["logic_solver_pfd"],
        "MANUAL_ACTIVATION_FAIL": theta["p_manual_activation_fail"],
    }
    for prefix, n, ind, full in (
        ("EPUMP", n_electric, e_ind, theta["fw_elec_pump_fts"]),
        ("DPUMP", n_diesel, d_ind, theta["fw_diesel_pump_fts"]),
    ):
        for i in range(1, n + 1):
            p[f"{prefix}_{i}_FTS"] = ind if n > 1 else full  # a single pump has no common-cause partner
            p[f"{prefix}_{i}_MAINT"] = theta["fw_pump_maint_unavail"]
            p[f"{prefix}_{i}_FTR"] = ftr
    return {k: _prob(v) for k, v in p.items()}


def gas_detection_pfd(theta: Mapping, n: int = 3, k: int = 2):
    """Voted gas detection hardware + logic solver (coverage is handled separately)."""
    return _prob(
        pfd_koon(k, n, theta["gd_lambda_du"], theta["proof_test_interval_h"], theta["gd_beta"]) + theta["logic_solver_pfd"]
    )


def esd_pfd(theta: Mapping, n_valves: int = 2):
    """Isolation fails if all valves fail to close (1ooN) or the logic solver fails."""
    return _prob(
        pfd_koon(1, n_valves, theta["esd_lambda_du"], theta["proof_test_interval_h"], theta["esd_beta"])
        + theta["logic_solver_pfd"]
    )
