"""Streamlit dashboard for the Offshore Asset Risk & Resilience Engine.

    streamlit run dashboard/app.py

The dashboard contains no model logic: it calls the library and caches results.
Heavy analyses (tornado, nested run) are read from outputs/results/ produced by
scripts/run_analysis.py; everything else is simulated live at the chosen size.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # allows running without `pip install -e .`
    sys.path.insert(0, str(ROOT / "src"))

from offshore_risk import load_config  # noqa: E402
from offshore_risk.appetite.framework import evaluate_appetite, exposure_metrics  # noqa: E402
from offshore_risk.asset import AssetModel  # noqa: E402
from offshore_risk.bayesian.calibration import apply_posteriors, calibrate  # noqa: E402
from offshore_risk.financial.loss_model import DIRECT  # noqa: E402
from offshore_risk.financial.metrics import METRIC_NOTES, exceedance_probability, risk_metrics, tail_allocation  # noqa: E402
from offshore_risk.mitigation.evaluate import evaluate_mitigations  # noqa: E402
from offshore_risk.scenarios.analysis import run_scenarios  # noqa: E402
from offshore_risk.simulation import SOURCES, SimulationSettings, Simulator  # noqa: E402
from offshore_risk.visualization import plots  # noqa: E402

RES = ROOT / "outputs" / "results"
SEED = 20260927

st.set_page_config(page_title="Offshore Risk Engine", layout="wide")


# ------------------------------------------------------------------ cached model calls
@st.cache_resource
def get_cfg(use_posteriors: bool):
    cfg = load_config()
    return apply_posteriors(cfg, calibrate(cfg)) if use_posteriors else cfg


@st.cache_data(show_spinner="Simulating…")
def run(use_posteriors: bool, n: int, seed: int, dependence: str, mitigations: tuple):
    cfg = get_cfg(use_posteriors)
    r = Simulator(cfg, mitigations).run(SimulationSettings(n_years=n, seed=seed, dependence=dependence))
    return {
        "total": r.total,
        "by_source": r.loss_by_source,
        "comp": r.component_frame().mean(),
        "downtime": r.downtime_days,
        "max_event": r.max_event_loss,
        "derived": r.derived_means,
        "counts": {k: v.mean() for k, v in r.counts.items()},
        "appetite_metrics": exposure_metrics(r),
    }


@st.cache_data(show_spinner="Running stress scenarios…")
def scenarios(use_posteriors: bool, n: int, seed: int, dependence: str):
    df, res = run_scenarios(get_cfg(use_posteriors), n_years=n, seed=seed, dependence=dependence)
    return df, {k: v.total for k, v in res.items()}, {k: v.downtime_days for k, v in res.items()}


@st.cache_data(show_spinner="Evaluating controls…")
def mitigations_eval(use_posteriors: bool, n: int, seed: int, dependence: str):
    df, _ = evaluate_mitigations(get_cfg(use_posteriors), n_years=n, seed=seed, dependence=dependence)
    return df


def money(x):
    return plots.musd(x)


def read_result(name):
    p = RES / name
    return pd.read_csv(p) if p.exists() else None


# ------------------------------------------------------------------ sidebar
st.sidebar.title("Model settings")
n_years = st.sidebar.select_slider("Simulated years", [10_000, 20_000, 50_000, 100_000], value=20_000)
dependence = st.sidebar.radio(
    "Dependence model",
    ["correlated", "independent"],
    format_func=lambda s: "B: correlated drivers" if s == "correlated" else "A: independent drivers",
)
use_post = st.sidebar.checkbox("Use Bayesian posteriors (synthetic records)", value=True)
seed = int(st.sidebar.number_input("Random seed", min_value=0, max_value=2**32 - 1, value=SEED, step=1))
cfg = get_cfg(use_post)
mit_keys = list(cfg.mitigations)
chosen = st.sidebar.multiselect("Controls in place", mit_keys, format_func=lambda k: cfg.mitigations[k]["name"])
st.sidebar.caption(
    "Fictional asset. All parameters are documented assumptions (docs/data_dictionary.md). "
    "Results are not predictions for any real facility."
)

base = run(use_post, n_years, seed, dependence, ())
cur = run(use_post, n_years, seed, dependence, tuple(chosen)) if chosen else base
m = risk_metrics(cur["total"])
asset = AssetModel.from_config(cfg.asset)

st.title("Offshore Asset Risk & Resilience Engine")
st.caption(
    f"{cfg.asset['complex_name']} · {n_years:,} simulated years · "
    f"{'Model B (correlated)' if dependence == 'correlated' else 'Model A (independent)'} · "
    f"{'posterior-updated' if use_post else 'prior'} parameters" + (f" · controls: {', '.join(chosen)}" if chosen else "")
)

tabs = st.tabs(["Executive", "Asset", "Scenarios", "Monte Carlo", "Mitigation", "Sensitivity"])

# ------------------------------------------------------------------ executive
with tabs[0]:
    c = st.columns(5)
    c[0].metric(
        "Expected annual loss",
        money(m["eal"]),
        None if not chosen else money(m["eal"] - base["total"].mean()),
        delta_color="inverse",
    )
    c[1].metric("Median year", money(m["median"]))
    c[2].metric("P95 (1-in-20)", money(m["p95"]))
    c[3].metric("P99 (1-in-100)", money(m["p99"]))
    c[4].metric("ES99 (mean of worst 1 %)", money(m["es99"]))
    st.caption(
        f"Monte Carlo standard error of EAL: {money(m['se_eal'])}. Mean > median because a few years carry most of the loss."
    )

    st.subheader("Risk appetite (hypothetical limits)")
    app = evaluate_appetite(cfg.appetite, cur["appetite_metrics"])
    icon = {"Within appetite": "🟢", "Near threshold": "🟡", "Outside appetite": "🔴"}
    show = app.assign(
        status=app["status"].map(lambda s: f"{icon[s]} {s}"),
        value=[money(v) if u.startswith("USD") else f"{v:.3g}" for v, u in zip(app["value"], app["unit"])],
        limit=[money(v) if u.startswith("USD") else f"{v:.3g}" for v, u in zip(app["limit"], app["unit"])],
        utilisation=app["utilisation"].map(lambda x: f"{x:.0%}"),
    )
    st.dataframe(show[["description", "value", "limit", "utilisation", "status"]], hide_index=True, width="stretch")
    st.caption("Limits are illustrative management thresholds for the fictional operator, not industry standards.")
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Top risk contributors")
        ta = tail_allocation(cur["by_source"], SOURCES, 0.99)
        st.pyplot(plots.contribution_bars(ta), clear_figure=True)
    with right:
        st.subheader("Where the money goes")
        st.pyplot(plots.component_bars(cur["comp"], DIRECT), clear_figure=True)
    st.caption(
        "Share of EAL answers 'what costs us every year'; share of ES99 answers 'what makes a year catastrophic'. They rank sources differently."
    )

# ------------------------------------------------------------------ asset
with tabs[1]:
    st.subheader("Platform interdependencies")
    st.pyplot(plots.asset_diagram(asset, asset.single_outage_table()), clear_figure=True)
    a1, a2 = st.columns(2)
    with a1:
        st.markdown("**Production lost if one platform is down (dependency effects included)**")
        so = asset.single_outage_table()
        st.dataframe(
            so.assign(fraction_of_complex=so["fraction_of_complex"].map("{:.0%}".format))[
                ["platform", "own_production_bopd", "complex_production_lost_bopd", "fraction_of_complex", "platforms_affected"]
            ].rename(
                columns={
                    "own_production_bopd": "own output (bbl/d)",
                    "complex_production_lost_bopd": "complex output lost (bbl/d)",
                    "fraction_of_complex": "share of complex",
                    "platforms_affected": "also affected",
                }
            ),
            hide_index=True,
            width="stretch",
        )
    with a2:
        st.markdown("**Protection-layer reliability (mean over parameter uncertainty)**")
        d = cur["derived"]
        rel = pd.DataFrame(
            {
                "platform": asset.codes,
                "fire protection PFD": d["pfd_fire_protection"],
                "firewater PFD": d["pfd_firewater"],
                "release rate /yr": [
                    cur["derived"]["rate_release"] * f / asset.release_factor.sum() for f in asset.release_factor
                ],
            }
        )
        st.dataframe(
            rel.style.format({"fire protection PFD": "{:.4f}", "firewater PFD": "{:.4f}", "release rate /yr": "{:.3f}"}),
            hide_index=True,
            width="stretch",
        )
        st.markdown(
            f"Gas detection PFD **{d['pfd_gas_detection']:.2e}** · ESD isolation PFD **{d['pfd_esd']:.2e}** · "
            f"total power loss **{d['rate_power_loss']:.3f}/yr** · compressor failures **{d['rate_compressor']:.2f}/yr**"
        )
    ftn = read_result("fault_tree_naive_vs_exact.csv")
    if ftn is not None:
        st.markdown(
            "**Why architecture matters:** firewater PFD computed exactly vs by naive multiplication (shared power supply ignored)"
        )
        st.dataframe(
            ftn[
                [
                    "platform",
                    "p_power_loss_given_fire",
                    "firewater_pfd_exact",
                    "firewater_pfd_naive_independent",
                    "naive_underestimate_factor",
                ]
            ].style.format(
                {
                    "firewater_pfd_exact": "{:.4f}",
                    "firewater_pfd_naive_independent": "{:.4f}",
                    "naive_underestimate_factor": "{:.1f}×",
                }
            ),
            hide_index=True,
            width="stretch",
        )
    ev = pd.Series(cur["counts"])
    st.markdown("**Simulated events per year:** " + " · ".join(f"{k} {v:.3g}" for k, v in ev.items()))

# ------------------------------------------------------------------ scenarios
with tabs[2]:
    n_sc = min(n_years, 20_000)
    sdf, sc_tot, sc_dt = scenarios(use_post, n_sc, seed, dependence)
    st.subheader("Stress scenarios (conditional on the initiating event)")
    st.pyplot(plots.scenario_comparison(sdf), clear_figure=True)
    show = sdf.copy()
    for col in ("mean_event_loss", "annual_eal", "annual_median", "annual_p95", "annual_p99", "incremental_mean_loss"):
        show[col] = show[col].map(lambda v: "—" if pd.isna(v) else money(v))
    show["annual_frequency"] = [
        ("—" if k == "baseline" else "state (conditional)") if pd.isna(v) else f"{v:.2e} /yr"
        for k, v in zip(sdf["scenario"], sdf["annual_frequency"])
    ]
    show["return_period_years"] = sdf["return_period_years"].map(lambda v: "—" if pd.isna(v) else f"{v:,.0f}")
    st.dataframe(
        show[
            [
                "title",
                "initiating_event",
                "annual_frequency",
                "return_period_years",
                "mean_event_loss",
                "annual_eal",
                "annual_p95",
                "annual_p99",
                "mean_downtime_days",
                "p95_downtime_days",
                "pfd_fire_protection_PPA",
            ]
        ].rename(columns={"pfd_fire_protection_PPA": "fire-protection PFD (PPA)"}),
        hide_index=True,
        width="stretch",
    )
    pick = st.selectbox("Scenario detail", [k for k in cfg.scenarios], format_func=lambda k: cfg.scenarios[k]["title"])
    spec = cfg.scenarios[pick]
    cc = st.columns([1, 1.3])
    with cc[0]:
        st.markdown(f"**Initiating event:** {spec['initiating_event']}")
        st.markdown("**Assumptions:**\n" + "\n".join(f"- {a}" for a in spec.get("assumptions", [])))
    with cc[1]:
        st.pyplot(
            plots.exceedance_curves(
                {"Baseline": sc_tot["baseline"], spec["title"]: sc_tot[pick]}, f"Annual loss exceedance: {spec['title']}"
            ),
            clear_figure=True,
        )
    st.caption(
        f"Scenarios use {n_sc:,} years with the same random numbers as the baseline, so differences are the scenario's own effect."
    )

# ------------------------------------------------------------------ monte carlo
with tabs[3]:
    st.subheader("Annual loss distribution")
    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(plots.loss_histogram(cur["total"], m), clear_figure=True)
    with c2:
        curves = {"Current settings": cur["total"]}
        if chosen:
            curves = {"No controls": base["total"], "With selected controls": cur["total"]}
        other = run(use_post, n_years, seed, "independent" if dependence == "correlated" else "correlated", tuple(chosen))
        curves["Other dependence model"] = other["total"]
        st.pyplot(plots.exceedance_curves(curves), clear_figure=True)
    mt = pd.DataFrame(
        [
            {"metric": k, "value": money(v) if k != "n" else f"{v:,} years", "meaning": METRIC_NOTES.get(k, "")}
            for k, v in m.items()
            if k in METRIC_NOTES or k == "n"
        ]
    )
    st.dataframe(mt, hide_index=True, width="stretch")
    thr = st.multiselect("Exceedance thresholds (USD M)", [10, 25, 50, 100, 250, 500, 1000], default=[25, 50, 100, 250])
    ex = exceedance_probability(cur["total"], [t * 1e6 for t in thr])
    st.write({f"P(loss > ${int(k / 1e6)}M)": f"{v:.2%} (1 in {1 / v:,.0f} yr)" if v > 0 else "0" for k, v in ex.items()})
    st.pyplot(plots.component_bars(cur["comp"], DIRECT), clear_figure=True)

# ------------------------------------------------------------------ mitigation
with tabs[4]:
    n_m = min(n_years, 20_000)
    mdf = mitigations_eval(use_post, n_m, seed, dependence)
    st.subheader("Individual controls (each measured against the baseline, common random numbers)")
    st.pyplot(plots.mitigation_effects(mdf), clear_figure=True)
    show = mdf[
        [
            "name",
            "capex_usd",
            "annualised_cost_usd",
            "reduction_eal",
            "reduction_eal_se",
            "reduction_p95",
            "reduction_es99",
            "residual_eal",
            "cost_per_usd_eal_reduction",
            "benefit_cost_ratio",
        ]
    ].copy()
    for col in (
        "capex_usd",
        "annualised_cost_usd",
        "reduction_eal",
        "reduction_eal_se",
        "reduction_p95",
        "reduction_es99",
        "residual_eal",
    ):
        show[col] = show[col].map(money)
    show["cost_per_usd_eal_reduction"] = mdf["cost_per_usd_eal_reduction"].map(lambda v: f"{v:.2f}" if np.isfinite(v) else "∞")
    show["benefit_cost_ratio"] = mdf["benefit_cost_ratio"].map(lambda v: f"{v:.1f}")
    st.dataframe(show, hide_index=True, width="stretch")
    if chosen:
        st.markdown(
            f"**Selected portfolio** ({', '.join(chosen)}): EAL {money(base['total'].mean())} → {money(cur['total'].mean())}; "
            f"P95 {money(np.quantile(base['total'], 0.95))} → {money(np.quantile(cur['total'], 0.95))}; "
            f"capex {money(sum(cfg.mitigations[k]['capex_usd'] for k in chosen))} (budget {money(cfg.budget_usd)})."
        )
    opt = RES / "optimisation.json"
    if opt.exists():
        import json

        o = json.loads(opt.read_text())
        st.subheader(f"Budget-constrained optimum (MILP, budget {money(cfg.budget_usd)})")
        rows = [
            {
                "objective": k,
                "selected controls": ", ".join(v["selected"]),
                "capex": money(v["capex"]),
                "EAL reduction": money(v["simulated_eal_reduction"]),
                "ES99 reduction": money(v["simulated_es99_reduction"]),
                "annualised cost": money(v["annualised_cost"]),
            }
            for k, v in o.items()
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(
            "From scripts/run_analysis.py. The optimal portfolio depends on the objective: minimising expected loss favours "
            "controls against frequent production losses; minimising tail loss favours detection and isolation."
        )

# ------------------------------------------------------------------ sensitivity
with tabs[5]:
    tor = read_result("tornado.csv")
    rc = read_result("rank_correlation_eal.csv")
    s1 = read_result("first_order_indices_eal.csv")
    summ = read_result("parameter_importance_summary.csv")
    if tor is None:
        st.info("Run `python scripts/run_analysis.py` to produce the sensitivity results.")
    else:
        st.subheader("Which assumptions matter most?")
        metric = st.radio("Tornado metric", ["eal", "p99"], horizontal=True, format_func=str.upper)
        st.pyplot(plots.tornado(tor, metric), clear_figure=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Combined ranking (tornado, Spearman, first-order index)**")
            st.dataframe(
                summ[["parameter", "eal_swing", "spearman_rho", "S1", "mean_rank"]].style.format(
                    {"eal_swing": lambda v: money(v), "spearman_rho": "{:.2f}", "S1": "{:.3f}", "mean_rank": "{:.1f}"}
                ),
                hide_index=True,
                width="stretch",
            )
        with c2:
            st.markdown("**Rank correlation with per-world ES99 (tail)**")
            rc99 = read_result("rank_correlation_es99.csv")
            st.dataframe(
                rc99.head(10)[["parameter", "spearman_rho", "p_value"]].style.format(
                    {"spearman_rho": "{:.2f}", "p_value": "{:.3f}"}
                ),
                hide_index=True,
                width="stretch",
            )
        st.caption(
            "Tornado: one parameter at P10/P90, others at medians (local). Spearman and S1: from the nested run where all parameters vary (global). "
            "Precomputed by scripts/run_analysis.py with posterior-updated parameters."
        )
