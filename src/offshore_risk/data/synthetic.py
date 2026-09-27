"""Reproducible synthetic operating history for the fictional complex.

The records are generated from the model itself with a documented
'synthetic truth' (``SYNTHETIC_TRUTH``) that deliberately differs from the
prior medians for a few parameters. This lets the Bayesian module demonstrate
updating, but it is circular by construction: it validates the *mechanics* of
the analysis, not the realism of the parameters.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import REPO_ROOT, ModelConfig
from ..simulation.engine import SimulationSettings, Simulator

DATA_DIR = REPO_ROOT / "data" / "synthetic"
START_YEAR, N_YEARS = 2014, 12

# Parameters whose synthetic truth is set away from the prior median.
SYNTHETIC_TRUTH = {
    "fw_elec_pump_fts": 0.022,      # electric pumps start worse than the generic prior (0.01)
    "fw_diesel_pump_fts": 0.012,    # diesel pump better than its prior (0.02)
    "spurious_trip_rate": 4.5,      # more trips than the prior median (3.0)
}
SEED = 20140101


def truth_theta(cfg: ModelConfig, n: int) -> dict:
    th = cfg.registry.at_quantile(n, 0.5)
    for k, v in SYNTHETIC_TRUTH.items():
        th[k] = np.full(n, float(v))
    return th


def generate(cfg: ModelConfig, out_dir: Path | str = DATA_DIR, seed: int = SEED) -> dict[str, pd.DataFrame]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    sim = Simulator(cfg)
    res = sim.run(SimulationSettings(n_years=N_YEARS, seed=seed, record_events=True), theta=truth_theta(cfg, N_YEARS))
    ev = res.events.copy()
    ev["year"] = ev["year"] + START_YEAR
    doy = rng.integers(0, 365, len(ev))
    ev["date"] = pd.to_datetime(ev["year"].astype(str) + "-01-01") + pd.to_timedelta(doy, unit="D")
    ev = ev.sort_values("date").reset_index(drop=True)
    category = {"release": "process_safety", "compressor": "mechanical", "power_loss": "mechanical", "pipeline": "mechanical",
                "spurious_trip": "mechanical", "weather": "external", "collision": "external"}
    log = pd.DataFrame({
        "event_id": [f"SYN-{i:04d}" for i in range(1, len(ev) + 1)],
        "date": ev["date"].dt.date,
        "year": ev["year"],
        "platform": ev["platform"],
        "category": ev["source"].map(category),
        "subcategory": ev["source"],
        "outcome": ev["detail"],
        "downtime_h": (ev["downtime_days"] * 24).round(1),
        "lost_production_bbl": ev["lost_bbl"].round(0),
        "direct_cost_usd": ev["direct_usd"].round(-2),
        "business_interruption_usd": ev["business_interruption_usd"].round(-2),
        "liquid_released_bbl": ev["spill_bbl"].round(1),
    })

    # weekly start tests of the three firewater pumps
    rows = []
    tags = [("FWP-A", "electric_firewater_pump", SYNTHETIC_TRUTH["fw_elec_pump_fts"]),
            ("FWP-B", "electric_firewater_pump", SYNTHETIC_TRUTH["fw_elec_pump_fts"]),
            ("FWP-C", "diesel_firewater_pump", SYNTHETIC_TRUTH["fw_diesel_pump_fts"])]
    dates = pd.date_range(f"{START_YEAR}-01-06", f"{START_YEAR + N_YEARS - 1}-12-31", freq="7D")
    for tag, typ, p in tags:
        fails = rng.random(len(dates)) < p
        for d, f in zip(dates, fails):
            rows.append({"date": d.date(), "equipment_tag": tag, "equipment_type": typ, "result": "fail_to_start" if f else "pass"})
    tests = pd.DataFrame(rows).sort_values(["date", "equipment_tag"]).reset_index(drop=True)
    tests.insert(0, "test_id", [f"PT-{i:05d}" for i in range(1, len(tests) + 1)])

    reg_rows = [
        ("FWP-A", "UCP", "electric_firewater_pump", "Electric-motor firewater pump A (100 %)"),
        ("FWP-B", "UCP", "electric_firewater_pump", "Electric-motor firewater pump B (100 %)"),
        ("FWP-C", "SWI", "diesel_firewater_pump", "Diesel-driven standby firewater pump (100 %)"),
        ("GTG-1", "UCP", "gas_turbine_generator", "Gas turbine generator 1 (50 %)"),
        ("GTG-2", "UCP", "gas_turbine_generator", "Gas turbine generator 2 (50 %)"),
        ("GTG-3", "UCP", "gas_turbine_generator", "Gas turbine generator 3 (50 %)"),
        ("K-101", "UCP", "gas_compressor_train", "Gas compression train 1 (50 %)"),
        ("K-102", "UCP", "gas_compressor_train", "Gas compression train 2 (50 %)"),
        ("ESDV-A1", "PPA", "esd_valve", "Inlet ESD valve 1 (1oo2)"),
        ("ESDV-A2", "PPA", "esd_valve", "Inlet ESD valve 2 (1oo2)"),
        ("ESDV-B1", "PPB", "esd_valve", "Inlet ESD valve 1 (1oo2)"),
        ("ESDV-B2", "PPB", "esd_valve", "Inlet ESD valve 2 (1oo2)"),
        ("XV-DEL-A", "PPA", "deluge_valve", "Deluge valve set, process module"),
        ("XV-DEL-B", "PPB", "deluge_valve", "Deluge valve set, process module"),
        ("PL-EXP-01", "PPA", "export_pipeline", "30 km export pipeline"),
    ]
    register = pd.DataFrame(reg_rows, columns=["tag", "platform", "type", "description"])
    register["installed_year"] = 2008
    exposure = pd.DataFrame({"year": np.arange(START_YEAR, START_YEAR + N_YEARS), "complex_years": 1.0,
                             "compressor_train_years": 2.0, "pipeline_km_years": 30.0})
    annual = (log.groupby(["year", "subcategory"]).size().unstack(fill_value=0)
              .reindex(exposure["year"], fill_value=0).reset_index())

    files = {"incident_log": log, "proof_tests": tests, "equipment_register": register, "exposure": exposure}
    for name, df in files.items():
        df.to_csv(out / f"{name}.csv", index=False)
    processed = REPO_ROOT / "data" / "processed" if Path(out_dir) == DATA_DIR else out
    processed.mkdir(parents=True, exist_ok=True)
    annual.to_csv(processed / "annual_event_counts.csv", index=False)
    files["annual_event_counts"] = annual
    return files


def load(out_dir: Path | str = DATA_DIR) -> dict[str, pd.DataFrame]:
    d = Path(out_dir)
    return {n: pd.read_csv(d / f"{n}.csv") for n in ("incident_log", "proof_tests", "equipment_register", "exposure")}
