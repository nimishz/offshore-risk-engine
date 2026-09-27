"""Run the full analysis and write outputs/results/*.csv|json and outputs/figures/*.png.

    python scripts/run_analysis.py            # full run (~3-5 min on a laptop)
    python scripts/run_analysis.py --quick    # reduced sample sizes for a smoke test

All random numbers derive from SEED, so results are reproducible.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from offshore_risk import load_config
from offshore_risk.appetite.framework import evaluate_appetite, exposure_metrics
from offshore_risk.asset import AssetModel
from offshore_risk.bayesian.calibration import apply_posteriors, calibrate, summary_table
from offshore_risk.bayesian.conjugate import BetaModel, GammaModel, sequential_update
from offshore_risk.config import REPO_ROOT
from offshore_risk.data import synthetic
from offshore_risk.data.dictionary import build as build_dictionary
from offshore_risk.fault_tree.protection import fire_protection_basic_events, fire_protection_tree, firewater_subtree
from offshore_risk.financial.loss_model import DIRECT, COMPONENTS
from offshore_risk.financial.metrics import METRIC_NOTES, bootstrap_ci, exceedance_probability, expected_shortfall, metrics_table, risk_metrics, tail_allocation
from offshore_risk.mitigation.evaluate import evaluate_mitigations
from offshore_risk.optimization.portfolio import PortfolioOptimizer
from offshore_risk.scenarios.analysis import run_scenarios
from offshore_risk.sensitivity.analysis import first_order_indices, importance_summary, rank_correlation, tornado
from offshore_risk.simulation import SOURCES, simulate
from offshore_risk.simulation.nested import run_nested
from offshore_risk.visualization import plots

SEED = 20260927
OUT = REPO_ROOT / "outputs"
FIG, RES = OUT / "figures", OUT / "results"


def log(msg, t0=[time.time()]):
    print(f"[{time.time() - t0[0]:6.1f}s] {msg}", flush=True)


def main(quick: bool = False):
    FIG.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    N_MAIN = 20_000 if quick else 100_000
    N_CMP = 10_000 if quick else 50_000
    N_OUT, N_IN = (80, 500) if quick else (400, 1000)
    N_TOR = 5_000 if quick else 20_000
    summary: dict = {"seed": SEED, "n_main": N_MAIN}

    cfg0 = load_config()
    build_dictionary(cfg0)
    if not (REPO_ROOT / "data/synthetic/incident_log.csv").exists():
        synthetic.generate(cfg0)
    data = synthetic.load()
    log("data dictionary + synthetic data ready")

    # ------------------------------------------------------------ Bayesian
    cal = calibrate(cfg0, data)
    tab = summary_table(cal)
    tab.to_csv(RES / "bayesian_updates.csv", index=False)
    cfg = apply_posteriors(cfg0, cal)
    c = cal["fw_elec_pump_fts"]
    grid = np.linspace(1e-5, 0.06, 600)
    plots.save(plots.bayes_update(c["prior"], c["posterior"], lambda x: BetaModel.likelihood_density(x, c["failures"], c["exposure"]), grid,
                                  f"Electric firewater pump fail-to-start: {c['failures']} failures in {c['exposure']} weekly tests",
                                  "P(fail to start) per demand", truth=synthetic.SYNTHETIC_TRUTH["fw_elec_pump_fts"]), FIG / "bayes_pump_fts.png")
    c = cal["spurious_trip_rate"]
    grid = np.linspace(0.01, 9, 600)
    plots.save(plots.bayes_update(c["prior"], c["posterior"], lambda x: GammaModel.likelihood_density(x, c["failures"], c["exposure"]), grid,
                                  f"Spurious trip rate: {c['failures']} trips in {c['exposure']:.0f} years", "Trips per year",
                                  truth=synthetic.SYNTHETIC_TRUTH["spurious_trip_rate"]), FIG / "bayes_trip_rate.png")
    tests = data["proof_tests"]
    tests["year"] = pd.to_datetime(tests["date"]).dt.year
    el = tests[tests["equipment_type"] == "electric_firewater_pump"].groupby("year")["result"].agg(k=lambda s: (s != "pass").sum(), n="size")
    seq = sequential_update(cal["fw_elec_pump_fts"]["prior"], el["k"], el["n"], labels=list(el.index))
    seq.to_csv(RES / "bayes_sequential_pump.csv", index=False)
    seq["period"] = ["prior"] + [str(i) for i in range(1, len(seq))]
    plots.save(plots.sequential_posterior(seq, "Electric pump fail-to-start: posterior as records accumulate", "P(fail to start)",
                                          truth=synthetic.SYNTHETIC_TRUTH["fw_elec_pump_fts"]), FIG / "bayes_sequential.png")
    summary["bayesian"] = tab.set_index("parameter")[["prior_mean", "posterior_mean", "ci_width_reduction"]].to_dict("index")
    log("Bayesian calibration")

    # ------------------------------------------------------------ reliability & fault tree
    asset = AssetModel.from_config(cfg.asset)
    th = cfg.registry.at_quantile(1, 0.5)
    ft_rows = []
    for plat, pp in zip(asset.codes, asset.p_power_loss_given_fire):
        be = fire_protection_basic_events(th, pp, 2, 1)
        fw, fp = firewater_subtree(2, 1), fire_protection_tree(2, 1)
        ft_rows.append({"platform": plat, "p_power_loss_given_fire": pp,
                        "firewater_pfd_exact": float(fw.probability(be, "exact")[0]),
                        "firewater_pfd_naive_independent": float(fw.probability(be, "independent")[0]),
                        "firewater_pfd_rare_event": float(fw.probability(be, "rare_event")[0]),
                        "fire_protection_pfd_exact": float(fp.probability(be, "exact")[0]),
                        "fire_protection_pfd_naive_independent": float(fp.probability(be, "independent")[0])})
    ft = pd.DataFrame(ft_rows)
    ft["naive_underestimate_factor"] = ft["firewater_pfd_exact"] / ft["firewater_pfd_naive_independent"]
    ft.to_csv(RES / "fault_tree_naive_vs_exact.csv", index=False)
    be = fire_protection_basic_events(th, asset.p_power_loss_given_fire[asset.index("PPA")], 2, 1)
    fpt = fire_protection_tree(2, 1)
    imp = fpt.importance(be)
    imp.to_csv(RES / "fault_tree_importance_PPA.csv", index=False)
    mcs = [{"cut_set": " AND ".join(sorted(cs)), "order": len(cs),
            "probability": float(np.prod([float(np.asarray(be[e]).ravel()[0]) for e in sorted(cs)]))} for cs in fpt.minimal_cut_sets()]
    (pd.DataFrame(mcs).sort_values(["probability", "cut_set"], ascending=[False, True])
     .to_csv(RES / "fault_tree_cut_sets_PPA.csv", index=False))
    (RES / "fault_tree_fire_protection.txt").write_text(fpt.to_text(), encoding="utf-8")
    plots.save(plots.fault_tree_importance(imp), FIG / "fault_tree_importance.png")
    shape, scale = float(th["comp_weibull_shape"][0]), float(th["comp_weibull_scale_years"][0])
    from offshore_risk.reliability.models import weibull_mean

    plots.save(plots.reliability_curves(shape, scale, 1.0 / float(weibull_mean(shape, scale))), FIG / "reliability_weibull_vs_exponential.png")
    single = asset.single_outage_table()
    single.to_csv(RES / "platform_single_outage.csv", index=False)
    plots.save(plots.asset_diagram(asset, single), FIG / "asset_dependencies.png")
    summary["fault_tree"] = ft.set_index("platform")[["firewater_pfd_exact", "firewater_pfd_naive_independent", "fire_protection_pfd_exact"]].to_dict("index")
    log("fault trees / reliability")

    # ------------------------------------------------------------ main Monte Carlo (Model B, posterior-updated)
    res_b = simulate(cfg, n_years=N_MAIN, seed=SEED, dependence="correlated")
    res_a = simulate(cfg, n_years=N_MAIN, seed=SEED, dependence="independent")
    res_prior = simulate(cfg0, n_years=N_MAIN, seed=SEED, dependence="correlated")
    L = res_b.total
    m = risk_metrics(L)
    m["eal_ci90"] = bootstrap_ci(L, np.mean, n_boot=200)
    m["p99_ci90"] = bootstrap_ci(L, lambda x: np.quantile(x, 0.99), n_boot=200)
    m["es99_ci90"] = bootstrap_ci(L, lambda x: expected_shortfall(x, 0.99), n_boot=200)
    thresholds = [25e6, 50e6, 100e6, 250e6, 500e6]
    m["exceedance"] = exceedance_probability(L, thresholds)
    summary["metrics_model_b"] = m
    mt = metrics_table({"Model A (independent)": res_a.total, "Model B (correlated)": res_b.total, "Model B, priors only (no Bayesian update)": res_prior.total})
    mt.to_csv(RES / "metrics_comparison.csv")
    pd.DataFrame([{"metric": k, "value_musd": (v / 1e6 if k not in ("n",) else v), "meaning": METRIC_NOTES.get(k, "")}
                  for k, v in risk_metrics(L).items()]).to_csv(RES / "risk_metrics.csv", index=False)
    lec = pd.DataFrame({"loss": np.geomspace(1e6, 2e9, 160)})
    for name, r in (("model_a", res_a), ("model_b", res_b)):
        xs = np.sort(r.total)
        lec[name] = 1 - np.searchsorted(xs, lec["loss"], side="right") / len(xs)
    lec.to_csv(RES / "loss_exceedance_curve.csv", index=False)
    plots.save(plots.loss_histogram(L, m, "Annual loss distribution (Model B, 100,000 simulated years)" if not quick else "Annual loss distribution"), FIG / "loss_histogram.png")
    plots.save(plots.exceedance_curves({"Model A: independent drivers": res_a.total, "Model B: correlated drivers": res_b.total},
                                       "Loss exceedance curve: effect of dependence"), FIG / "lec_model_a_vs_b.png")
    ta = tail_allocation(res_b.loss_by_source, SOURCES, 0.99)
    ta.to_csv(RES / "tail_allocation.csv", index=False)
    plots.save(plots.contribution_bars(ta), FIG / "contribution_mean_vs_tail.png")
    comp = res_b.component_frame().mean()
    comp.to_csv(RES / "loss_components.csv", header=["mean_usd"])
    plots.save(plots.component_bars(comp, DIRECT), FIG / "loss_components.png")
    ab = {}
    for q in (0.5, 0.9, 0.95, 0.99, 0.995):
        a_, b_ = np.quantile(res_a.total, q), np.quantile(res_b.total, q)
        ab[f"q{q}"] = {"model_a": a_, "model_b": b_, "ratio": b_ / a_}
    ab["es99"] = {"model_a": expected_shortfall(res_a.total, .99), "model_b": expected_shortfall(res_b.total, .99)}
    ab["eal"] = {"model_a": res_a.total.mean(), "model_b": res_b.total.mean()}
    summary["dependence"] = ab
    summary["counts_per_year"] = {k: float(v.mean()) for k, v in res_b.counts.items()}
    summary["derived_means"] = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in res_b.derived_means.items()}
    summary["downtime_days"] = {"mean": float(res_b.downtime_days.mean()), "p95": float(np.quantile(res_b.downtime_days, 0.95))}
    log("main Monte Carlo")

    # ------------------------------------------------------------ appetite
    app = evaluate_appetite(cfg.appetite, exposure_metrics(res_b))
    app.to_csv(RES / "risk_appetite.csv", index=False)
    plots.save(plots.appetite_table_fig(app), FIG / "risk_appetite.png")
    summary["appetite"] = app.set_index("limit_id")[["value", "limit", "status"]].to_dict("index")

    # ------------------------------------------------------------ nested (epistemic vs aleatory)
    nr = run_nested(cfg, n_outer=N_OUT, n_inner=N_IN, seed=SEED)
    nr.worlds.to_csv(RES / "nested_worlds.csv", index=False)
    ns = nr.metric_summary()
    ns.to_csv(RES / "nested_metric_uncertainty.csv", index=False)
    summary["nested"] = ns.set_index("metric").to_dict("index")
    plots.save(plots.exceedance_band(nr, L), FIG / "lec_epistemic_band.png")
    plots.save(plots.epistemic_metric_hist(nr, "eal"), FIG / "epistemic_eal.png")
    log("nested run")

    # ------------------------------------------------------------ sensitivity
    tor = tornado(cfg, n_years=N_TOR, seed=SEED)
    tor.to_csv(RES / "tornado.csv", index=False)
    plots.save(plots.tornado(tor, "eal"), FIG / "tornado_eal.png")
    plots.save(plots.tornado(tor, "p99", title="Tornado: one-at-a-time sensitivity of P99"), FIG / "tornado_p99.png")
    rc = rank_correlation(nr, "eal")
    rc.to_csv(RES / "rank_correlation_eal.csv", index=False)
    rc99 = rank_correlation(nr, "es99")
    rc99.to_csv(RES / "rank_correlation_es99.csv", index=False)
    s1 = first_order_indices(nr, "eal")
    s1.to_csv(RES / "first_order_indices_eal.csv", index=False)
    s1_99 = first_order_indices(nr, "es99")
    s1_99.to_csv(RES / "first_order_indices_es99.csv", index=False)
    plots.save(plots.importance_bars(s1, "S1", title="First-order variance share of EAL across worlds",
                                     xlabel=f"S1 (given-data estimate; null level ≈ {s1.attrs['null_level']:.3f})"), FIG / "sobol_first_order_eal.png")
    imp_sum = importance_summary(tor, rc, s1)
    imp_sum.to_csv(RES / "parameter_importance_summary.csv", index=False)
    summary["sensitivity_top"] = imp_sum.head(8)[["parameter", "eal_swing", "spearman_rho", "S1"]].to_dict("records")
    summary["sensitivity_top_es99"] = rc99.head(6)[["parameter", "spearman_rho"]].to_dict("records")
    log("sensitivity")

    # ------------------------------------------------------------ scenarios
    sdf, _ = run_scenarios(cfg, n_years=N_CMP // 2, seed=SEED)
    sdf.to_csv(RES / "scenarios.csv", index=False)
    plots.save(plots.scenario_comparison(sdf), FIG / "scenarios.png")
    summary["scenarios"] = sdf.drop(columns=["assumptions"]).to_dict("records")
    log("scenarios")

    # ------------------------------------------------------------ mitigation & optimisation
    mdf, _ = evaluate_mitigations(cfg, n_years=N_CMP, seed=SEED, baseline=None)
    mdf.to_csv(RES / "mitigations.csv", index=False)
    plots.save(plots.mitigation_effects(mdf), FIG / "mitigations.png")
    summary["mitigations"] = mdf[["mitigation", "capex_usd", "annualised_cost_usd", "reduction_eal", "reduction_eal_se",
                                  "reduction_es99", "cost_per_usd_eal_reduction", "benefit_cost_ratio"]].to_dict("records")
    opt = PortfolioOptimizer(cfg, n_years=N_CMP // 2, seed=SEED)
    port = {}
    for obj in ("eal", "es99", "net_benefit"):
        r = opt.optimise(objective=obj)
        port[obj] = {k: v for k, v in r.items() if k not in ("a", "b", "selected_idx")}
        port[obj]["standalone_sum"] = float(r["a"][list(r["selected_idx"])].sum())
    opt.coefficient_table("eal").to_csv(RES / "optimisation_interactions_eal.csv")
    (RES / "optimisation.json").write_text(json.dumps(port, indent=2, default=float))
    summary["optimisation"] = port
    log("mitigation + optimisation")

    (RES / "summary.json").write_text(json.dumps(summary, indent=2, default=lambda o: float(o) if np.isscalar(o) else str(o)))
    log(f"done -> {RES}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
