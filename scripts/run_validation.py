"""Model-level validation that is too slow for the unit tests.

Writes docs/validation_results.md, outputs/results/convergence.csv and
outputs/figures/convergence.png.

    python scripts/run_validation.py            # full (~3 min)
    python scripts/run_validation.py --quick
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")  # headless figure output

import argparse
import json
import time

import numpy as np
import pandas as pd

from offshore_risk import load_config
from offshore_risk.asset import AssetModel
from offshore_risk.bayesian.calibration import apply_posteriors, calibrate
from offshore_risk.bayesian.conjugate import BetaModel
from offshore_risk.config import REPO_ROOT
from offshore_risk.event_tree import OUTCOMES, RELEASE_TREE
from offshore_risk.financial.metrics import expected_shortfall, risk_metrics
from offshore_risk.optimization.portfolio import PortfolioOptimizer
from offshore_risk.simulation import SOURCES, ScenarioSpec, SimulationSettings, Simulator, simulate
from offshore_risk.simulation.derived import derive
from offshore_risk.visualization import plots

SEED = 20260927
RES, FIG = REPO_ROOT / "outputs" / "results", REPO_ROOT / "outputs" / "figures"


def md_table(df: pd.DataFrame, floatfmt="{:,.3g}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(out)


def convergence(cfg, sizes, seeds):
    rows = []
    for n in sizes:
        for s in seeds:
            L = simulate(cfg, n_years=n, seed=s).total
            m = risk_metrics(L)
            rows.append(
                {"n_years": n, "seed": s, **{k: m[k] for k in ("eal", "median", "p95", "p99", "es99")}, "se_eal": m["se_eal"]}
            )
    df = pd.DataFrame(rows)
    agg = df.groupby("n_years")[["eal", "p95", "p99", "es99"]].agg(["mean", "std"])
    cv = pd.DataFrame({m: agg[(m, "std")] / agg[(m, "mean")] for m in ("eal", "p95", "p99", "es99")})
    return df, cv


def event_tree_check(cfg, n_years):
    """Sampled release outcome shares vs analytic event-tree probabilities (medians, median-theta run)."""
    asset = AssetModel.from_config(cfg.asset)
    th = cfg.registry.at_quantile(1, 0.5)
    d = derive(th, asset)
    res = Simulator(cfg).run(SimulationSettings(n_years=n_years, seed=3, epistemic="median", record_events=True))
    ev = res.events[res.events["source"] == "release"]
    rows = []
    rates = d.release_rate[0]
    # analytic mix over platforms, sizes and phases
    tot = {o: 0.0 for o in OUTCOMES}
    for i, lam in enumerate(rates):
        if lam == 0:
            continue
        g = asset.gas_fraction[i]
        for s in range(3):
            for gas, wp in ((True, g), (False, 1 - g)):
                p = {
                    "p_detect": d.p_detect[0, s],
                    "p_isolate": d.p_isolate[0],
                    "p_ign": d.p_ign[0, s],
                    "p_ign_iso": d.p_ign[0, s] * th["ign_isolation_factor"][0],
                    "p_exp": d.p_exp[0, s] * (1 if gas else th["liquid_explosion_factor"][0]),
                    "p_fp_ok": 1 - d.pfd_fp[0, i],
                }
                op = RELEASE_TREE.outcome_probabilities(p)
                for o in OUTCOMES:
                    tot[o] += lam / rates.sum() * d.size_probs[0, s] * wp * float(op[o])
    n = len(ev)
    for o in OUTCOMES:
        k = int((ev["detail"] == o).sum())
        se = np.sqrt(tot[o] * (1 - tot[o]) / n)
        rows.append(
            {
                "outcome": o,
                "analytic": tot[o],
                "simulated": k / n,
                "events": k,
                "z_score": (k / n - tot[o]) / se if se > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows), n


def bayes_coverage(n_rep=2000, seed=5):
    """If truth ~ prior, 90 % posterior intervals should cover the truth ~90 % of the time."""
    rng = np.random.default_rng(seed)
    prior = BetaModel.from_mean(0.01, 100)
    hits = 0
    for _ in range(n_rep):
        p = rng.beta(prior.a, prior.b)
        k = rng.binomial(624, p)
        lo, hi = prior.update(k, 624).interval(0.9)
        hits += lo <= p <= hi
    return hits / n_rep


def main(quick=False):
    t0 = time.time()
    cfg = apply_posteriors(load_config(), calibrate(load_config()))
    sizes = [10_000, 50_000, 100_000]
    seeds = [1, 2, 3] if quick else [1, 2, 3, 4, 5]
    conv, cv = convergence(cfg, sizes, seeds)
    conv.to_csv(RES / "convergence.csv", index=False)
    plots.save(plots.convergence(conv), FIG / "convergence.png")
    print(f"convergence {time.time() - t0:.0f}s")

    et, n_rel = event_tree_check(cfg, 50_000 if quick else 200_000)
    print(f"event tree {time.time() - t0:.0f}s")

    # dependence: same frequencies, different tails
    a = simulate(cfg, n_years=100_000, seed=SEED, dependence="independent")
    b = simulate(cfg, n_years=100_000, seed=SEED, dependence="correlated")
    dep = pd.DataFrame(
        [
            {"check": "mean releases per year", "model_a": a.counts["releases"].mean(), "model_b": b.counts["releases"].mean()},
            {
                "check": "mean compressor loss (USD)",
                "model_a": a.loss_by_source[:, SOURCES.index("compressor")].mean(),
                "model_b": b.loss_by_source[:, SOURCES.index("compressor")].mean(),
            },
            {"check": "EAL (USD)", "model_a": a.total.mean(), "model_b": b.total.mean()},
            {"check": "P90 (USD)", "model_a": np.quantile(a.total, 0.9), "model_b": np.quantile(b.total, 0.9)},
            {"check": "P95 (USD)", "model_a": np.quantile(a.total, 0.95), "model_b": np.quantile(b.total, 0.95)},
            {"check": "P99 (USD)", "model_a": np.quantile(a.total, 0.99), "model_b": np.quantile(b.total, 0.99)},
            {"check": "ES99 (USD)", "model_a": expected_shortfall(a.total, 0.99), "model_b": expected_shortfall(b.total, 0.99)},
        ]
    )
    dep["ratio_b_over_a"] = dep["model_b"] / dep["model_a"]
    # stronger dependence stress: correlations scaled up
    strong = cfg.copy()
    strong.drivers["correlation"] = [[1.0, 0.9, 0.7], [0.9, 1.0, 0.8], [0.7, 0.8, 1.0]]
    s = simulate(strong, n_years=100_000, seed=SEED, dependence="correlated")
    dep_strong = {
        "P95": np.quantile(s.total, 0.95) / np.quantile(a.total, 0.95),
        "P99": np.quantile(s.total, 0.99) / np.quantile(a.total, 0.99),
        "ES99": expected_shortfall(s.total, 0.99) / expected_shortfall(a.total, 0.99),
        "EAL": s.total.mean() / a.total.mean(),
    }
    print(f"dependence {time.time() - t0:.0f}s")

    # optimisation: MILP vs exhaustive enumeration
    opt = PortfolioOptimizer(cfg, n_years=10_000 if quick else 25_000, seed=SEED)
    opt_rows = []
    for obj in ("eal", "es99"):
        r = opt.optimise(objective=obj)
        best, best_v, table = opt.exhaustive(objective=obj)
        opt_rows.append(
            {
                "objective": obj,
                "milp_selection": " + ".join(r["selected"]),
                "milp_value_simulated": r["simulated_value"],
                "milp_value_predicted": r["predicted_value"],
                "exhaustive_best": " + ".join(best),
                "exhaustive_value": best_v,
                "feasible_portfolios": len(table),
                "milp_rank": int((table["value"] > r["simulated_value"] + 1e-6).sum()) + 1,
                "capex": r["capex"],
                "budget": r["budget"],
            }
        )
    optdf = pd.DataFrame(opt_rows)
    print(f"optimisation {time.time() - t0:.0f}s")

    cov = bayes_coverage(500 if quick else 2000)

    # stress edge cases
    edge = []
    sc = ScenarioSpec(overrides={"bi_value_fraction": {"op": "set", "value": 0.0}})
    r0 = Simulator(cfg, scenario=sc).run(SimulationSettings(n_years=5000, seed=1))
    edge.append(
        {
            "case": "bi_value_fraction = 0",
            "expected": "business interruption = 0",
            "result": f"BI total = {r0.loss_by_component[:, 5].sum():.0f}",
        }
    )
    sc = ScenarioSpec(
        overrides={
            "p_ign_small": {"op": "set", "value": 0.0},
            "p_ign_medium": {"op": "set", "value": 0.0},
            "p_ign_large": {"op": "set", "value": 0.0},
        }
    )
    r1 = Simulator(cfg, scenario=sc).run(SimulationSettings(n_years=20000, seed=1))
    edge.append(
        {
            "case": "all ignition probabilities = 0",
            "expected": "no fires or explosions",
            "result": f"fires = {r1.counts['fires'].sum()}, explosions = {r1.counts['explosions'].sum()}",
        }
    )
    sc = ScenarioSpec(overrides={"fw_header_fail": {"op": "set", "value": 1.0}})
    r2 = Simulator(cfg, scenario=sc).run(SimulationSettings(n_years=2000, seed=1))
    edge.append(
        {
            "case": "firewater header always failed",
            "expected": "firewater PFD = 1",
            "result": f"PFD firewater (PPA) = {r2.derived_means['pfd_firewater'][1]:.3f}",
        }
    )
    r3 = Simulator(cfg).run(SimulationSettings(n_years=1, seed=1))
    edge.append({"case": "one simulated year", "expected": "runs", "result": f"loss = {r3.total[0]:,.0f}"})
    edgedf = pd.DataFrame(edge)

    summary = {
        "mode": "quick" if quick else "full",
        "seeds": seeds,
        "convergence_cv": {str(int(k)): {m: float(v) for m, v in row.items()} for k, row in cv.iterrows()},
        "event_tree_releases": int(n_rel),
        "event_tree_max_abs_z": float(et["z_score"].abs().max()),
        "dependence": {
            r["check"]: {"model_a": float(r["model_a"]), "model_b": float(r["model_b"]), "ratio": float(r["ratio_b_over_a"])}
            for _, r in dep.iterrows()
        },
        "dependence_strong": {k: float(v) for k, v in dep_strong.items()},
        "optimisation": optdf.to_dict("records"),
        "bayes_coverage": float(cov),
        "bayes_coverage_reps": 500 if quick else 2000,
    }
    (RES / "validation_summary.json").write_text(json.dumps(summary, indent=2, default=float))

    lines = [
        "# Validation results",
        "",
        f"_Generated by `scripts/run_validation.py` ({'quick' if quick else 'full'} mode). Unit tests (`pytest`) cover the analytical checks; this file covers model-level behaviour._",
        "",
        "## 1. Monte Carlo convergence",
        "",
        f"Seeds: {seeds}. Coefficient of variation of each estimate across seeds (lower = more stable):",
        "",
        md_table(cv.reset_index().rename(columns={"n_years": "simulated years"}).astype({"simulated years": str}), "{:.3f}"),
        "",
        "Estimates per run (USD millions):",
        "",
        md_table(
            conv.assign(**{c: conv[c] / 1e6 for c in ("eal", "median", "p95", "p99", "es99", "se_eal")}).astype(
                {"n_years": str, "seed": str}
            ),
            "{:.2f}",
        ),
        "",
        "![convergence](../outputs/figures/convergence.png)",
        "",
        "## 2. Event tree: analytic vs simulated outcome shares",
        "",
        f"Median parameters, {n_rel:,} simulated releases. |z| < 3 expected for every row.",
        "",
        md_table(et, "{:.4g}"),
        "",
        "## 3. Dependence models A vs B",
        "",
        "Frequencies are identical by construction; EAL rises slightly under B because frequency (integrity driver) and duration "
        "(logistics driver) co-move, so E[frequency × duration] > E[frequency] × E[duration].",
        "",
        md_table(dep, "{:,.4g}"),
        "",
        "Stress test with much stronger driver correlation (0.7–0.9), ratio to Model A: "
        + ", ".join(f"{k} {v:.3f}" for k, v in dep_strong.items()),
        "",
        "## 4. Optimisation: MILP vs exhaustive enumeration",
        "",
        md_table(optdf, "{:,.4g}"),
        "",
        "`milp_rank` = 1 means the MILP choice is also the best portfolio found by simulating every feasible combination.",
        "",
        "## 5. Bayesian interval coverage",
        "",
        f"Truth drawn from the Beta prior, 624 demands per synthetic data set: nominal 90 % posterior intervals covered the truth in **{cov:.1%}** of {500 if quick else 2000} repetitions.",
        "",
        "## 6. Edge cases",
        "",
        md_table(edgedf),
        "",
    ]
    (REPO_ROOT / "docs" / "validation_results.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
