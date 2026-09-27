# Changelog

## 0.2.0 — review release

Correctness
- Fire-protection fault tree: basic events are now numbered within each pump group
  (`EPUMP_1…`, `DPUMP_1…`). Previously a third electric pump would have collided with the
  first diesel pump's name and been merged.
- Diesel firewater pumps have their own common-cause β-factor (`fw_diesel_ccf_beta`).
- Flame-detection voting is read from the asset configuration instead of being fixed at 2oo3.
- Restart costs apply only to platforms with process inventory (WHP, PPA, PPB, UCP), not to
  the living quarters or water-injection platforms.
- Every outage duration (not only fire/explosion outages) is capped at `max_event_downtime_days`.
- Escalation supports any number of bridge neighbours; the random-number layout adapts.
- Power-loss repair cost moved from code to `config/risk_parameters.yaml`.
- Platform-specific metrics use platform codes instead of hard-coded indices.
- Parameter `basis` labels corrected (`comp_weibull_scale_years` is an assumption;
  `spurious_trip_rate` is synthetic-calibrated).
- Basic-event probabilities are clipped to [0, 1]; release-size probabilities cannot exceed 1.

Robustness and security
- New `config_validation.py`: schema, range and cross-reference checks run on every
  `load_config`, reporting all problems together (`ConfigError`).
- `SimulationSettings` and caller-supplied parameter arrays are validated.
- Dashboard bound to `localhost` with usage statistics disabled (`.streamlit/config.toml`);
  seed input bounded.
- The library no longer forces a matplotlib backend (figures render in notebooks again).
- `numpy>=2.0` declared (the code uses `np.trapezoid`).

Quality
- Ruff lint and format configuration; codebase formatted.
- GitHub Actions CI: lint, tests on Python 3.10–3.12, quick end-to-end analysis.
- README and result-dependent docs generated from `outputs/results/` by
  `scripts/render_docs.py`, which also checks the narrative's qualitative claims.
- New tests for configuration validation, analysis layers, pump architectures, restart
  costs, duration caps and settings validation.
- All results regenerated.

## 0.1.0 — initial release

Complete model, analysis pipeline, dashboard, notebooks and documentation.
