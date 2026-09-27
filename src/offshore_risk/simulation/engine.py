"""Frequency-severity Monte Carlo engine for annual operational loss.

One simulated year:

1. epistemic parameters theta (fresh per year in 'pooled' mode, fixed at the
   medians in 'median' mode, supplied by the caller in nested runs);
2. mitigation / scenario transforms of theta; rates and PFDs derived from it;
3. year-level drivers (weather, logistics, integrity) under Model A or B,
   and an annual oil price;
4. for every event source: candidate events at the *reference* (unmitigated)
   rate, thinned to the actual rate, then consequences per event;
5. consequences -> downtime by platform -> barrels lost -> money.

Random numbers come from named streams seeded by (seed, chunk, stream name),
so a run with a mitigation re-uses exactly the same random numbers as the
baseline for everything the mitigation does not touch (common random numbers).
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.special import ndtri

from ..asset.network import AssetModel, lost_production_bbl
from ..config import ModelConfig
from ..distributions import beta_from_mean, make_distribution
from ..event_tree.hydrocarbon_release import OUTCOMES, RELEASE_TREE
from ..financial.loss_model import C, COMPONENTS, SCENARIO_ONLY, FinancialModel
from ..mitigation.controls import apply_mitigations, apply_overrides
from .dependence import sample_drivers
from .derived import SIZES, Derived, derive

SOURCES = ["release", "compressor", "power_loss", "pipeline", "spurious_trip", "weather", "collision", "scenario_event"]
S = {name: i for i, name in enumerate(SOURCES)}
O = {name: i for i, name in enumerate(OUTCOMES)}
COUNT_KEYS = ["releases", "fires", "explosions", "catastrophic", "escalations", "evacuations"]
_ISOLATED_LEAF = RELEASE_TREE.leaf_attribute("Isolated")
_EPS = 1e-12


def _ln(median, sigma, u):
    """Lognormal draw around a median (fast inverse transform)."""
    return np.asarray(median) * np.exp(sigma * ndtri(np.clip(u, _EPS, 1 - _EPS)))


@dataclass
class SimulationSettings:
    n_years: int = 20_000
    seed: int = 20260927
    dependence: str = "correlated"      # "independent" (Model A) | "correlated" (Model B)
    epistemic: str = "pooled"           # "pooled" | "median"
    chunk_size: int = 25_000
    include_env_cost: bool = False
    record_events: bool = False


@dataclass
class ScenarioSpec:
    name: str = "baseline"
    forced_events: list = field(default_factory=list)
    overrides: dict = field(default_factory=dict)
    architecture: dict = field(default_factory=dict)
    price_override: float | None = None
    logistics_multiplier: float = 1.0

    @classmethod
    def from_config(cls, key: str, spec: dict) -> "ScenarioSpec":
        return cls(
            name=key,
            forced_events=list(spec.get("forced_events", [])),
            overrides=dict(spec.get("overrides", {}) or {}),
            architecture=dict(spec.get("architecture", {}) or {}),
            price_override=spec.get("price_override"),
            logistics_multiplier=float(spec.get("logistics_multiplier", 1.0)),
        )


@dataclass
class SimulationResult:
    loss_by_source: np.ndarray
    loss_by_component: np.ndarray
    lost_bbl: np.ndarray
    downtime_days: np.ndarray        # lost production in equivalent days of full-complex output
    max_event_loss: np.ndarray
    env_proxy_bbl: np.ndarray
    counts: dict
    derived_means: dict
    settings: SimulationSettings
    mitigations: tuple = ()
    scenario: str = "baseline"
    events: pd.DataFrame | None = None

    @property
    def included_components(self) -> list[str]:
        return [c for c in COMPONENTS if self.settings.include_env_cost or c not in SCENARIO_ONLY]

    @property
    def total(self) -> np.ndarray:
        return self.loss_by_source.sum(axis=1)

    def component_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.loss_by_component, columns=COMPONENTS)

    def source_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.loss_by_source, columns=SOURCES)

    @property
    def n_years(self) -> int:
        return len(self.total)


class _Accumulator:
    def __init__(self, n: int, include_env: bool, record: bool):
        self.n = n
        self.by_source = np.zeros((n, len(SOURCES)))
        self.by_comp = np.zeros((n, len(COMPONENTS)))
        self.lost = np.zeros(n)
        self.spill = np.zeros(n)
        self.max_event = np.zeros(n)
        self.counts = {k: np.zeros(n, dtype=np.int64) for k in COUNT_KEYS}
        self.include = np.array([include_env or c not in SCENARIO_ONLY for c in COMPONENTS])
        self.record = record
        self.rows: list[pd.DataFrame] = []
        self.pf = np.ones(n)

    def add(self, source: str, year: np.ndarray, batch: dict, year_offset: int = 0):
        if len(year) == 0:
            return
        comps = batch["comps"]
        ev_total = comps[:, self.include].sum(axis=1)
        for j in range(comps.shape[1]):
            self.by_comp[:, j] += np.bincount(year, weights=comps[:, j], minlength=self.n)
        self.by_source[:, S[source]] += np.bincount(year, weights=ev_total, minlength=self.n)
        self.lost += np.bincount(year, weights=batch["lost"], minlength=self.n)
        self.spill += np.bincount(year, weights=batch["spill"], minlength=self.n)
        np.maximum.at(self.max_event, year, ev_total)
        for k, v in batch.get("flags", {}).items():
            self.counts[k] += np.bincount(year, weights=v.astype(float), minlength=self.n).astype(np.int64)
        if self.record:
            df = pd.DataFrame(
                {
                    "year": year + year_offset,
                    "source": source,
                    "platform": batch.get("platform", np.full(len(year), "")),
                    "detail": batch.get("detail", np.full(len(year), "")),
                    "downtime_days": batch.get("downtime", np.zeros(len(year))),
                    "lost_bbl": batch["lost"],
                    "direct_usd": comps[:, [C[c] for c in ("property_damage", "equipment_repair", "emergency_response", "logistics", "restart")]].sum(axis=1),
                    "business_interruption_usd": comps[:, C["business_interruption"]],
                    "total_usd": ev_total,
                    "spill_bbl": batch["spill"],
                }
            )
            self.rows.append(df)


class Simulator:
    """Monte Carlo simulator for one configuration + set of mitigations + scenario."""

    def __init__(self, cfg: ModelConfig, mitigations: Iterable[str] = (), scenario: ScenarioSpec | None = None):
        self.cfg = cfg
        self.asset = AssetModel.from_config(cfg.asset)
        self.fin = FinancialModel.from_config(cfg.financial)
        self.mitigations = tuple(mitigations)
        unknown = set(self.mitigations) - set(cfg.mitigations)
        if unknown:
            raise KeyError(f"unknown mitigations: {sorted(unknown)}")
        self.scenario = scenario or ScenarioSpec()
        self.cap_table = self.asset.capacity_table()
        self.q_total = self.cap_table[0]
        self.nb = self.asset.neighbour_matrix(3)
        self.al = cfg.aleatory
        self._price_dist = make_distribution(cfg.financial["oil_price"])
        idx = self.asset.index
        self.trip_idx = np.array([idx(c) for c in self.al["trip_platforms"]])
        tgt = self.al["collision_targets"]
        self.coll_idx = np.array([idx(c) for c in tgt])
        self.coll_cdf = np.cumsum(np.array(list(tgt.values()), dtype=float) / sum(tgt.values()))
        self.lq = idx("LQ")

    # ------------------------------------------------------------------ utils
    def _rng(self, seed: int, chunk: int, name: str) -> np.random.Generator:
        return np.random.default_rng([seed, chunk, zlib.crc32(name.encode())])

    def _outcome_table(self, theta: Mapping, key: str, n: int) -> np.ndarray:
        """(n, n_outcomes) table of an outcome attribute that may reference theta."""
        out = np.zeros((n, len(OUTCOMES)))
        for j, o in enumerate(OUTCOMES):
            v = self.al["outcomes"][o][key]
            out[:, j] = theta[v] if isinstance(v, str) else float(v)
        return out

    # ------------------------------------------------------------------ run
    def run(self, settings: SimulationSettings | None = None, theta: Mapping[str, np.ndarray] | None = None) -> SimulationResult:
        st = settings or SimulationSettings()
        n_total = st.n_years if theta is None else len(next(iter(theta.values())))
        parts = []
        derived_acc: dict[str, list] = {}
        for c, start in enumerate(range(0, n_total, st.chunk_size)):
            n = min(st.chunk_size, n_total - start)
            th = None if theta is None else {k: np.asarray(v)[start : start + n] for k, v in theta.items()}
            acc, dm = self._run_chunk(st, c, n, th, start)
            parts.append(acc)
            for k, v in dm.items():
                derived_acc.setdefault(k, []).append((v, n))
        cat = lambda attr: np.concatenate([getattr(p, attr) for p in parts])
        counts = {k: np.concatenate([p.counts[k] for p in parts]) for k in COUNT_KEYS}
        derived_means = {k: sum(v * n for v, n in lst) / n_total for k, lst in derived_acc.items()}
        lost = cat("lost")
        events = pd.concat([r for p in parts for r in p.rows], ignore_index=True) if st.record_events and any(p.rows for p in parts) else None
        pf = cat("pf")
        return SimulationResult(
            loss_by_source=cat("by_source"),
            loss_by_component=cat("by_comp"),
            lost_bbl=lost,
            downtime_days=lost / (self.q_total * pf),
            max_event_loss=cat("max_event"),
            env_proxy_bbl=cat("spill"),
            counts=counts,
            derived_means=derived_means,
            settings=st,
            mitigations=self.mitigations,
            scenario=self.scenario.name,
            events=events,
        )

    def _run_chunk(self, st: SimulationSettings, chunk: int, n: int, theta_given, offset: int):
        cfg, reg = self.cfg, self.cfg.registry
        rng = lambda name: self._rng(st.seed, chunk, name)

        # 1. epistemic parameters
        if theta_given is not None:
            theta = theta_given
        elif st.epistemic == "pooled":
            theta = reg.sample(rng("epistemic").random((n, len(reg))))
        elif st.epistemic == "median":
            theta = reg.at_quantile(n, 0.5)
        else:
            raise ValueError(f"unknown epistemic mode {st.epistemic!r}")

        # 2. mitigations and scenario overrides
        th_mod, arch, _ = apply_mitigations(theta, self.mitigations, cfg.mitigations, rng)
        th_mod = apply_overrides(th_mod, self.scenario.overrides)
        arch = {**arch, **self.scenario.architecture}
        d_ref = derive(theta, self.asset)
        d = derive(th_mod, self.asset, arch)

        # 3. year-level drivers and price
        drv = sample_drivers(n, rng("drivers"), cfg.drivers, st.dependence)
        fdrv = cfg.drivers["frequency_driver"]
        L = drv["logistics"] * self.scenario.logistics_multiplier
        price = self._price_dist.ppf(rng("price").random(n))
        if self.scenario.price_override is not None:
            price = np.full(n, float(self.scenario.price_override))

        ctx = {
            "theta": th_mod, "d": d, "L": L, "price": price,
            "pf": th_mod["production_rate_factor"], "v": th_mod["bi_value_fraction"],
            "dt_med": self._outcome_table(th_mod, "downtime_days", n),
            "cs_days": self._outcome_table(th_mod, "complex_shutdown_days", n),
            "dmg_mean": self._outcome_table(th_mod, "damage_fraction", n),
            "er_cost": self._outcome_table(th_mod, "er_cost_usd", n),
        }
        acc = _Accumulator(n, st.include_env_cost, st.record_events)
        acc.pf = np.asarray(th_mod["production_rate_factor"], dtype=float)

        # 4. event sources
        # -- hydrocarbon releases (per platform rates, one stream)
        r = rng("source:release")
        lam_c = np.maximum(d_ref.release_rate, d.release_rate)
        tot_c = lam_c.sum(axis=1)
        counts = r.poisson(tot_c * drv[fdrv["release"]])
        yr = np.repeat(np.arange(n), counts)
        u = r.random((len(yr), 15))
        cdf = np.cumsum(lam_c / np.where(tot_c > 0, tot_c, 1.0)[:, None], axis=1)
        plat = np.minimum((u[:, 0, None] > cdf[yr]).sum(axis=1), self.asset.n - 1)
        keep = u[:, 1] * lam_c[yr, plat] < d.release_rate[yr, plat]
        acc.add("release", yr[keep], self._release(ctx, yr[keep], plat[keep], u[keep, 2:]), offset)

        # -- simple sources: (name, reference rate, actual rate, driver, handler)
        simple = [
            ("compressor", d_ref.compressor_rate, d.compressor_rate, self._compressor),
            ("power_loss", d_ref.power_loss_rate, d.power_loss_rate, self._power_loss),
            ("pipeline", d_ref.pipeline_rate, d.pipeline_rate, self._pipeline),
            ("spurious_trip", d_ref.spurious_trip_rate, d.spurious_trip_rate, self._trip),
            ("weather", d_ref.weather_rate, d.weather_rate, self._weather),
            ("collision", d_ref.collision_rate, d.collision_rate, self._collision),
        ]
        for name, lam_ref, lam, handler in simple:
            r = rng(f"source:{name}")
            lc = np.maximum(lam_ref, lam)
            counts = r.poisson(lc * drv[fdrv[name]])
            yr = np.repeat(np.arange(n), counts)
            u = r.random((len(yr), 18))
            keep = u[:, 0] * lc[yr] < lam[yr]
            acc.add(name, yr[keep], handler(ctx, yr[keep], u[keep, 1:]), offset)

        # -- forced scenario events: one per simulated year
        for i, fe in enumerate(self.scenario.forced_events):
            r = rng(f"forced:{i}")
            yr = np.arange(n)
            u = r.random((n, 18))
            acc.add("scenario_event", yr, self._forced(ctx, yr, u, fe), offset)

        dm = {
            "pfd_fire_protection": d.pfd_fp.mean(axis=0),
            "pfd_firewater": d.pfd_firewater.mean(axis=0),
            "pfd_gas_detection": float(d.pfd_gas_detection.mean()),
            "pfd_esd": float(d.pfd_esd.mean()),
            **{f"rate_{k}": float(v.mean()) for k, v in d.rates().items()},
        }
        return acc, dm

    # ------------------------------------------------------------ consequences
    def _batch(self, n_ev: int) -> dict:
        return {"comps": np.zeros((n_ev, len(COMPONENTS))), "lost": np.zeros(n_ev), "spill": np.zeros(n_ev), "flags": {}}

    def _release(self, ctx, yr, plat, u, size=None, gas=None, outcome=None, escalate_to=None) -> dict:
        """Consequences of hydrocarbon releases. u has >= 13 columns."""
        th, d = ctx["theta"], ctx["d"]
        E = len(yr)
        b = self._batch(E)
        if E == 0:
            return b
        if size is None:
            size = (u[:, 0, None] > np.cumsum(d.size_probs[yr], axis=1)).sum(axis=1).clip(0, 2)
        else:
            size = np.full(E, SIZES.index(size))
        if gas is None:
            gas = u[:, 1] < self.asset.gas_fraction[plat]
        else:
            gas = np.full(E, bool(gas))
        params = {
            "p_detect": d.p_detect[yr, size],
            "p_isolate": d.p_isolate[yr],
            "p_ign": d.p_ign[yr, size],
            "p_ign_iso": d.p_ign[yr, size] * th["ign_isolation_factor"][yr],
            "p_exp": d.p_exp[yr, size] * np.where(gas, 1.0, th["liquid_explosion_factor"][yr]),
            "p_fp_ok": 1.0 - d.pfd_fp[yr, plat],
        }
        leaf = RELEASE_TREE.sample_leaves(params, u[:, 2])
        out = RELEASE_TREE.leaf_outcome_index(leaf)
        isolated = _ISOLATED_LEAF[leaf]
        if outcome is not None:
            out = np.full(E, O[outcome])
            isolated = np.isin(out, [O["controlled_release"], O["minor_fire"]])

        # escalation across bridges
        delayed = u[:, 3] < th["p_er_delayed"][yr]
        base = np.zeros(E)
        base[out == O["major_fire"]] = th["esc_major_fire"][yr][out == O["major_fire"]]
        base[out == O["explosion"]] = th["esc_explosion"][yr][out == O["explosion"]]
        base[out == O["catastrophic"]] = th["esc_catastrophic"][yr][out == O["catastrophic"]]
        p_esc = np.minimum(1.0, base * np.where(delayed, th["er_delay_escalation_multiplier"][yr], 1.0))
        nbm = self.nb[plat]
        esc = (u[:, 4:7] < p_esc[:, None]) & (nbm >= 0)
        if escalate_to:
            forced_idx = [self.asset.index(c) for c in escalate_to]
            esc = esc | np.isin(nbm, forced_idx)

        # downtime by platform
        sig, cap = float(self.al["downtime_sigma"]), float(self.al["max_event_downtime_days"])
        L = ctx["L"][yr]
        D = np.zeros((E, self.asset.n))
        dt_own = np.minimum(_ln(ctx["dt_med"][yr, out], sig, u[:, 7]) * L, cap)
        D[np.arange(E), plat] = dt_own
        dt_nb = np.minimum(_ln(th["dt_escalated_days"][yr], sig, u[:, 8]) * L, cap)
        for j in range(nbm.shape[1]):
            sel = esc[:, j]
            D[sel, nbm[sel, j]] = np.maximum(D[sel, nbm[sel, j]], dt_nb[sel])
        D = np.maximum(D, ctx["cs_days"][yr, out][:, None])
        lost = lost_production_bbl(D, self.cap_table) * ctx["pf"][yr]

        # money
        conc = float(self.al["damage_beta_concentration"])
        rv = self.asset.replacement_value
        mean = ctx["dmg_mean"][yr, out]
        frac = np.where(mean > 0, beta_from_mean(np.maximum(mean, 1e-9), conc, u[:, 9]), 0.0)
        prop = frac * rv[plat]
        frac_nb = beta_from_mean(float(self.al["escalated_damage_fraction"]), conc, u[:, 10])
        for j in range(nbm.shape[1]):
            prop += esc[:, j] * frac_nb * rv[np.maximum(nbm[:, j], 0)]
        evac = (out == O["catastrophic"]) | (esc & (nbm == self.lq)).any(axis=1)
        er = ctx["er_cost"][yr, out] * _ln(1.0, float(self.al["cost_sigma"]), u[:, 11]) + evac * self.fin.evacuation_cost
        repair_days = np.where(out >= O["minor_fire"], dt_own, 0.0) + (esc * dt_nb[:, None]).sum(axis=1)
        cm = b["comps"]
        cm[:, C["property_damage"]] = prop
        cm[:, C["emergency_response"]] = er
        cm[:, C["logistics"]] = self.fin.logistics_day_rate * repair_days
        cm[:, C["restart"]] = self.fin.restart_cost * (D > 0).sum(axis=1)
        cm[:, C["business_interruption"]] = self.fin.business_interruption(lost, ctx["price"][yr], ctx["v"][yr])
        med_spill = np.array([self.al["liquid_spill_bbl_median"][s] for s in SIZES])[size]
        mult = np.where(isolated, 1.0, float(self.al["liquid_spill_unisolated_multiplier"]))
        spill = np.where(gas, 0.0, _ln(med_spill * mult, sig, u[:, 12]))
        cm[:, C["environmental_scenario"]] = spill * self.fin.env_cost_per_bbl
        b.update(
            lost=lost,
            spill=spill,
            platform=np.array(self.asset.codes)[plat],
            detail=np.array(OUTCOMES)[out],
            downtime=dt_own,
            flags={
                "releases": np.ones(E, dtype=bool),
                "fires": out >= O["minor_fire"],
                "explosions": np.isin(out, [O["explosion"], O["catastrophic"]]),
                "catastrophic": out == O["catastrophic"],
                "escalations": esc.any(axis=1),
                "evacuations": evac,
            },
        )
        return b

    def _whole_complex_outage(self, ctx, yr, days, b):
        b["lost"] = days * self.q_total * ctx["pf"][yr]
        b["comps"][:, C["business_interruption"]] = self.fin.business_interruption(b["lost"], ctx["price"][yr], ctx["v"][yr])
        b["downtime"] = days
        return b

    def _compressor(self, ctx, yr, u, **_):
        th = ctx["theta"]
        b = self._batch(len(yr))
        sig = float(self.al["downtime_sigma"])
        dt = _ln(th["comp_repair_days"][yr], sig, u[:, 0]) * ctx["L"][yr]
        b["lost"] = th["comp_production_impact"][yr] * dt * self.q_total * ctx["pf"][yr]
        b["comps"][:, C["business_interruption"]] = self.fin.business_interruption(b["lost"], ctx["price"][yr], ctx["v"][yr])
        b["comps"][:, C["equipment_repair"]] = _ln(float(self.al["compressor_repair_cost_usd_median"]), float(self.al["cost_sigma"]), u[:, 1])
        b.update(downtime=dt, platform=np.full(len(yr), "UCP"), detail=np.full(len(yr), "compressor_train_failure"))
        return b

    def _power_loss(self, ctx, yr, u, restart_days_median=None, **_):
        th = ctx["theta"]
        b = self._batch(len(yr))
        med = th["power_restart_days"][yr] if restart_days_median is None else float(restart_days_median)
        dt = _ln(med, float(self.al["downtime_sigma"]), u[:, 0]) * ctx["L"][yr]
        self._whole_complex_outage(ctx, yr, dt, b)
        b["comps"][:, C["restart"]] = 4 * self.fin.restart_cost
        b["comps"][:, C["equipment_repair"]] = _ln(2.0e5, float(self.al["cost_sigma"]), u[:, 1])
        b.update(platform=np.full(len(yr), "UCP"), detail=np.full(len(yr), "total_power_loss"))
        return b

    def _pipeline(self, ctx, yr, u, **_):
        th = ctx["theta"]
        b = self._batch(len(yr))
        sig = float(self.al["downtime_sigma"])
        dt = np.minimum(_ln(th["pipeline_repair_days"][yr], sig, u[:, 0]) * ctx["L"][yr], float(self.al["max_event_downtime_days"]))
        self._whole_complex_outage(ctx, yr, dt, b)
        b["comps"][:, C["equipment_repair"]] = _ln(th["pipeline_repair_cost_usd"][yr], float(self.al["cost_sigma"]), u[:, 1])
        b["comps"][:, C["logistics"]] = self.fin.logistics_day_rate * dt
        b["comps"][:, C["restart"]] = 2 * self.fin.restart_cost
        b["spill"] = _ln(float(self.al["pipeline_spill_bbl_median"]), sig, u[:, 2])
        b["comps"][:, C["environmental_scenario"]] = b["spill"] * self.fin.env_cost_per_bbl
        b.update(platform=np.full(len(yr), "export"), detail=np.full(len(yr), "pipeline_loss_of_containment"))
        return b

    def _trip(self, ctx, yr, u, **_):
        b = self._batch(len(yr))
        E = len(yr)
        p = self.trip_idx[np.minimum((u[:, 0] * len(self.trip_idx)).astype(int), len(self.trip_idx) - 1)]
        dt = _ln(float(self.al["trip_downtime_hours_median"]) / 24.0, float(self.al["downtime_sigma"]), u[:, 1])
        D = np.zeros((E, self.asset.n))
        D[np.arange(E), p] = dt
        b["lost"] = lost_production_bbl(D, self.cap_table) * ctx["pf"][yr]
        b["comps"][:, C["business_interruption"]] = self.fin.business_interruption(b["lost"], ctx["price"][yr], ctx["v"][yr])
        b["comps"][:, C["restart"]] = self.fin.restart_cost
        b.update(downtime=dt, platform=np.array(self.asset.codes)[p], detail=np.full(E, "spurious_trip"))
        return b

    def _weather(self, ctx, yr, u, shutin_days_median=None, damage=True, **_):
        th = ctx["theta"]
        b = self._batch(len(yr))
        med = float(self.al["weather_shutin_days_median"]) if shutin_days_median is None else float(shutin_days_median)
        dt = _ln(med, float(self.al["downtime_sigma"]), u[:, 0]) * ctx["L"][yr]
        self._whole_complex_outage(ctx, yr, dt, b)
        dmg = (u[:, 1] < th["weather_damage_prob"][yr]) if damage else np.zeros(len(yr), dtype=bool)
        b["comps"][:, C["property_damage"]] = dmg * _ln(float(self.al["weather_damage_usd_median"]), float(self.al["cost_sigma"]), u[:, 2])
        b["comps"][:, C["emergency_response"]] = self.fin.weather_demanning_cost
        b["comps"][:, C["restart"]] = 4 * self.fin.restart_cost
        b.update(platform=np.full(len(yr), "complex"), detail=np.where(dmg, "shut_in_with_damage", "precautionary_shut_in"))
        return b

    def _collision(self, ctx, yr, u, **_):
        th = ctx["theta"]
        E = len(yr)
        b = self._batch(E)
        if E == 0:
            return b
        tgt = self.coll_idx[np.minimum((u[:, 0, None] > self.coll_cdf).sum(axis=1), len(self.coll_idx) - 1)]
        sig = float(self.al["downtime_sigma"])
        dt = _ln(float(self.al["collision_downtime_days_median"]), sig, u[:, 1]) * ctx["L"][yr]
        D = np.zeros((E, self.asset.n))
        D[np.arange(E), tgt] = dt
        b["lost"] = lost_production_bbl(D, self.cap_table) * ctx["pf"][yr]
        b["comps"][:, C["business_interruption"]] = self.fin.business_interruption(b["lost"], ctx["price"][yr], ctx["v"][yr])
        b["comps"][:, C["property_damage"]] = _ln(float(self.al["collision_damage_usd_median"]), float(self.al["cost_sigma"]), u[:, 2])
        b["comps"][:, C["logistics"]] = self.fin.logistics_day_rate * dt
        b["comps"][:, C["restart"]] = self.fin.restart_cost
        b.update(downtime=dt, platform=np.array(self.asset.codes)[tgt], detail=np.full(E, "vessel_impact"))
        # riser rupture -> large release on the struck platform (if it holds hydrocarbons)
        rup = (u[:, 3] < th["p_riser_rupture_given_collision"][yr]) & (self.asset.release_factor[tgt] > 0)
        if rup.any():
            rb = self._release(ctx, yr[rup], tgt[rup], u[rup, 4:], size="large")
            for k in ("comps", "lost", "spill"):
                b[k][rup] = b[k][rup] + rb[k]
            b["flags"] = {k: np.zeros(E, dtype=bool) for k in rb["flags"]}
            for k, v in rb["flags"].items():
                b["flags"][k][rup] = v
            det = b["detail"].astype(object)
            det[rup] = "vessel_impact+riser_release:" + rb["detail"].astype(object)
            b["detail"] = det
        return b

    def _forced(self, ctx, yr, u, fe: dict) -> dict:
        kind = fe["type"]
        if kind == "release":
            plat = np.full(len(yr), self.asset.index(fe["platform"]))
            return self._release(
                ctx, yr, plat, u, size=fe.get("size"),
                gas=None if fe.get("phase") is None else fe["phase"] == "gas",
                outcome=fe.get("outcome"), escalate_to=fe.get("escalate_to"),
            )
        handlers = {"compressor": self._compressor, "power_loss": self._power_loss, "pipeline": self._pipeline,
                    "weather": self._weather, "spurious_trip": self._trip, "collision": self._collision}
        extra = {k: v for k, v in fe.items() if k != "type"}
        return handlers[kind](ctx, yr, u, **extra)


def simulate(cfg: ModelConfig, n_years: int = 20_000, seed: int = 20260927, dependence: str = "correlated",
             mitigations: Iterable[str] = (), scenario: ScenarioSpec | None = None, **kw) -> SimulationResult:
    """Convenience wrapper."""
    st = SimulationSettings(n_years=n_years, seed=seed, dependence=dependence, **kw)
    return Simulator(cfg, mitigations, scenario).run(st)
