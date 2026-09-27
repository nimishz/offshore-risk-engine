"""Render README.md and the result-dependent docs from docs/templates/ and outputs/results/.

Every number quoted in the documentation is a ``{{token}}`` filled from the
result files, so text and results cannot drift apart. Before writing, the
script also checks the *qualitative* claims the narrative makes (e.g. "the
tail is dominated by releases and the pipeline"); if a model change breaks a
claim, rendering fails and the text has to be revisited.

    python scripts/render_docs.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from itertools import combinations

import numpy as np
import pandas as pd

from offshore_risk import load_config
from offshore_risk.asset import AssetModel
from offshore_risk.config import REPO_ROOT
from offshore_risk.reliability.models import repairable_koon_failure_frequency

RES = REPO_ROOT / "outputs" / "results"
TEMPLATES = REPO_ROOT / "docs" / "templates"
TARGETS = {
    "README.md": REPO_ROOT / "README.md",
    "validation.md": REPO_ROOT / "docs" / "validation.md",
    "research_notes.md": REPO_ROOT / "docs" / "research_notes.md",
    "assumptions.md": REPO_ROOT / "docs" / "assumptions.md",
}
LABELS = {
    "release": "Hydrocarbon releases (incl. fires/explosions)",
    "compressor": "Compressor failures",
    "weather": "Extreme weather shut-ins",
    "spurious_trip": "Spurious trips",
    "pipeline": "Export pipeline",
    "power_loss": "Total power loss",
    "collision": "Vessel collision",
}


def m(x: float, nd: int | None = None) -> str:
    """USD millions with sensible precision."""
    v = x / 1e6
    if nd is None:
        nd = 0 if abs(v) >= 100 else 1 if abs(v) >= 1 else 2
    return f"{v:,.{nd}f}"


def pct(x: float, nd: int = 0) -> str:
    if 0 < x < 0.5 * 10 ** (-nd) / 100:
        return f"<{10 ** (-nd):g} %"
    return f"{100 * x:.{nd}f} %"


def md(df: pd.DataFrame, align: str | None = None) -> str:
    cols = list(df.columns)
    align = align or "l" + "r" * (len(cols) - 1)
    sep = ["---:" if a == "r" else "---" for a in align]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(sep) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


class ClaimError(AssertionError):
    pass


def claim(ok: bool, text: str):
    if not ok:
        raise ClaimError(f"documented claim no longer holds: {text}")


def test_count() -> str:
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"], cwd=REPO_ROOT, capture_output=True, text=True)
    hit = re.search(r"(\d+) tests? collected", out.stdout)
    return hit.group(1) if hit else "?"


def build_tokens() -> dict[str, str]:
    cfg = load_config()
    asset = AssetModel.from_config(cfg.asset)
    s = json.loads((RES / "summary.json").read_text())
    v = json.loads((RES / "validation_summary.json").read_text())
    mb = s["metrics_model_b"]
    t: dict[str, str] = {}

    # ---------------------------------------------------------------- headline distribution
    t["seed"] = str(s["seed"])
    t["n_main"] = f"{s['n_main']:,}"
    for k in ("eal", "median", "p75", "p90", "p95", "p99", "es95", "es99"):
        t[k] = m(mb[k])
    for k in ("eal", "p99", "es99"):
        lo, hi = mb[f"{k}_ci90"]
        t[f"{k}_ci"] = f"{m(lo)}–{m(hi)}"
    t["median_share"] = pct(mb["median"] / mb["eal"])
    t["top1_share"] = pct(0.01 * mb["es99"] / mb["eal"])
    ex = {float(k): val for k, val in mb["exceedance"].items()}
    for thr in (100e6, 500e6):
        t[f"p_gt_{int(thr / 1e6)}"] = pct(ex[thr], 2 if ex[thr] < 0.01 else 1)
        t[f"rp_{int(thr / 1e6)}"] = f"{1 / ex[thr]:,.0f}"
    comp = pd.read_csv(RES / "loss_components.csv", index_col=0)["mean_usd"]
    t["bi_share"] = pct(comp["business_interruption"] / comp.sum())
    t["bi_musd"] = m(comp["business_interruption"])

    # ---------------------------------------------------------------- mean vs tail
    ta = pd.read_csv(RES / "tail_allocation.csv").set_index("source")
    rows = [(LABELS.get(k, k), pct(r["share_of_eal"]), pct(r["share_of_es99"])) for k, r in ta.iterrows()]
    t["table_tail"] = md(pd.DataFrame(rows, columns=["Source", "Share of EAL", "Share of ES99"]))
    frequent = ta.reindex(["compressor", "weather", "spurious_trip", "power_loss"])["share_of_eal"].sum()
    frequent_tail = ta.reindex(["compressor", "weather", "spurious_trip", "power_loss"])["share_of_es99"].sum()
    majors = ta.reindex(["release", "pipeline"])["share_of_es99"].sum()
    t["frequent_share_eal"] = pct(frequent)
    t["frequent_share_es99"] = pct(frequent_tail)
    t["major_share_es99"] = pct(majors)
    claim(majors > 0.8 and frequent_tail < 0.1, "releases + pipeline dominate ES99; frequent events do not")
    claim(frequent > 0.4, "frequent small events are a large share of the EAL")

    # ---------------------------------------------------------------- fault tree
    ft = pd.read_csv(RES / "fault_tree_naive_vs_exact.csv").set_index("platform")
    t["fw_exact_ppa"] = f"{ft.loc['PPA', 'firewater_pfd_exact']:.4f}"
    t["fw_naive_ppa"] = f"{ft.loc['PPA', 'firewater_pfd_naive_independent']:.4f}"
    t["fw_factor"] = f"{ft.loc['PPA', 'naive_underestimate_factor']:.1f}"
    t["fw_exact_ucp"] = f"{ft.loc['UCP', 'firewater_pfd_exact']:.3f}"
    imp = pd.read_csv(RES / "fault_tree_importance_PPA.csv")
    t["fv_top_event"] = imp.iloc[0]["basic_event"]
    t["fv_top"] = pct(imp.iloc[0]["fussell_vesely"])
    claim(ft.loc["PPA", "naive_underestimate_factor"] > 1.5, "naive multiplication materially underestimates firewater PFD")

    # ---------------------------------------------------------------- dependence
    mc = pd.read_csv(RES / "metrics_comparison.csv", index_col=0)
    a, b = mc.iloc[0], mc.iloc[1]
    rows = []
    for name, r in (("Model A (independent drivers)", a), ("Model B (correlated drivers)", b)):
        rows.append((name, *(f"{r[k]:,.1f}" if r[k] < 100 else f"{r[k]:,.0f}" for k in ("eal", "p90", "p95", "p99", "es99"))))
    t["table_dependence"] = md(pd.DataFrame(rows, columns=["USD M", "EAL", "P90", "P95", "P99", "ES99"]))
    for k in ("eal", "p90", "p95", "p99", "es99"):
        t[f"dep_{k}"] = f"{100 * (b[k] / a[k] - 1):+.0f} %"
    t["prior_eal"] = f"{mc.iloc[2]['eal']:.1f}"
    vd = v["dependence"]
    t["dep_freq_ratio"] = f"{vd['mean releases per year']['ratio']:.3f}"
    ds = v["dependence_strong"]
    t["strong_p95"] = f"{100 * (ds['P95'] - 1):+.0f} %"
    t["strong_es99"] = f"{100 * (ds['ES99'] - 1):+.0f} %"
    claim(b["p95"] / a["p95"] > 1.03, "correlation raises P95")
    claim(abs(b["es99"] / a["es99"] - 1) < 0.05, "correlation leaves ES99 roughly unchanged")

    # ---------------------------------------------------------------- nested
    ns = pd.read_csv(RES / "nested_metric_uncertainty.csv").set_index("metric")
    rows = [(k.upper(), m(ns.loc[k, "p05"]), m(ns.loc[k, "median"]), m(ns.loc[k, "p95"])) for k in ("eal", "p99", "es99")]
    t["table_nested"] = md(pd.DataFrame(rows, columns=["Metric (USD M)", "5 % of worlds", "median world", "95 % of worlds"]))
    t["nested_eal_range"] = f"{m(ns.loc['eal', 'p05'])}–{m(ns.loc['eal', 'p95'])}"
    t["nested_eal_ratio"] = f"{ns.loc['eal', 'ratio_p95_p05']:.1f}"
    lo_r, hi_r = sorted((ns.loc["p99", "ratio_p95_p05"], ns.loc["es99", "ratio_p95_p05"]))
    t["nested_tail_ratio"] = f"about {lo_r:.1f}" if round(lo_r, 1) == round(hi_r, 1) else f"{lo_r:.1f}–{hi_r:.1f}"
    n_worlds = len(pd.read_csv(RES / "nested_worlds.csv"))
    t["n_worlds"] = str(n_worlds)

    # ---------------------------------------------------------------- sensitivity
    imp_s = pd.read_csv(RES / "parameter_importance_summary.csv")
    rows = [
        (i + 1, f"`{r.parameter}`", m(r.eal_swing, 1), f"{r.spearman_rho:+.2f}", f"{r.S1:.2f}")
        for i, r in enumerate(imp_s.head(8).itertuples())
    ]
    t["table_sensitivity"] = md(
        pd.DataFrame(rows, columns=["Rank", "Parameter", "EAL swing P10→P90 (USD M)", "Spearman ρ (EAL)", "First-order index"]),
        "rlrrr",
    )
    t["top_param"] = imp_s.iloc[0]["parameter"]
    rc = pd.read_csv(RES / "rank_correlation_eal.csv")
    t["rho_se"] = f"{1 / np.sqrt(n_worlds):.2f}"
    t["top_rho"] = f"{rc.iloc[0]['spearman_rho']:.2f}"
    t["second_rho"] = f"{rc.iloc[1]['abs_rho']:.2f}"
    s1 = pd.read_csv(RES / "first_order_indices_eal.csv")
    t["top_s1"] = f"{s1.iloc[0]['S1']:.2f}"
    rc99 = pd.read_csv(RES / "rank_correlation_es99.csv")
    t["tail_params"] = ", ".join(f"`{p}`" for p in rc99.head(6)["parameter"])
    tor = pd.read_csv(RES / "tornado.csv")
    t["tornado_base"] = m(tor["eal_base"].iloc[0])
    claim(imp_s.iloc[0]["parameter"] == "bi_value_fraction", "bi_value_fraction is the most influential assumption")
    claim(
        rc.iloc[0]["abs_rho"] - rc.iloc[1]["abs_rho"] > 2 / np.sqrt(n_worlds),
        "the top parameter is clearly separated from the rest",
    )
    gaps = -np.diff(rc["abs_rho"].to_numpy()[1:8])
    claim(gaps.max() < 1 / np.sqrt(n_worlds), "adjacent ranks 2-8 differ by less than one sampling SE of rho")
    prot = {
        "fw_elec_pump_fts",
        "fw_diesel_pump_fts",
        "fw_header_fail",
        "deluge_lambda_du",
        "fd_lambda_du",
        "p_manual_activation_fail",
    }
    claim(not prot & set(imp_s.head(8)["parameter"]), "protection-system reliability parameters are not among the top drivers")

    # ---------------------------------------------------------------- scenarios
    sc = pd.read_csv(RES / "scenarios.csv").set_index("scenario")
    rows = []
    for k, r in sc.drop(index="baseline").iterrows():
        f = (
            "state"
            if np.isnan(r["annual_frequency"])
            else (f"{r['annual_frequency']:.2g} /yr" if r["annual_frequency"] >= 1e-3 else f"{r['annual_frequency']:.1e} /yr")
        )
        ev = f"+{m(r['incremental_mean_loss'], 2)} M/yr" if np.isnan(r["mean_event_loss"]) else m(r["mean_event_loss"])
        rows.append((f"{k.split('_')[0]} {r['title']}", f, ev, m(r["annual_p99"])))
    t["table_scenarios"] = md(
        pd.DataFrame(
            rows,
            columns=["Scenario", "Annual frequency", "Mean loss of the event (USD M)", "Conditional P99 of the year (USD M)"],
        )
    )
    s2 = sc.loc["S2_firewater_degradation"]
    t["s2_pfd_base"] = f"{sc.loc['baseline', 'pfd_fire_protection_PPA']:.3f}"
    t["s2_pfd"] = f"{s2['pfd_fire_protection_PPA']:.3f}"
    t["s2_incr"] = m(s2["incremental_mean_loss"], 2)
    for k in ("n_scenarios", "n_mitigations", "n_optimisation", "n_tornado"):
        t[k] = f"{s[k]:,}"
    t["n_inner"] = f"{s['nested_sizes']['n_inner']:,}"
    claim(s2["incremental_mean_loss"] < 0.02 * sc.loc["baseline", "annual_eal"], "firewater degradation barely moves the EAL")

    # ---------------------------------------------------------------- mitigations and optimisation
    mt = pd.read_csv(RES / "mitigations.csv")
    rows = [
        (
            r["name"],
            m(r["capex_usd"], 1),
            f"{m(r['reduction_eal'], 2)} (±{m(r['reduction_eal_se'], 2)})",
            m(r["reduction_es99"]),
            f"{r['cost_per_usd_eal_reduction']:.2f}",
        )
        for _, r in mt.iterrows()
    ]
    t["table_mitigations"] = md(
        pd.DataFrame(
            rows,
            columns=[
                "Control",
                "Capex (USD M)",
                "EAL reduction, USD M/yr (±SE)",
                "ES99 reduction (USD M)",
                "Annualised cost per USD of EAL reduced",
            ],
        )
    )
    best_ce = mt.sort_values("cost_per_usd_eal_reduction").iloc[0]
    best_tail = mt.sort_values("reduction_es99", ascending=False).iloc[0]
    t["best_ce_control"] = best_ce["name"]
    t["best_ce_ratio"] = f"{best_ce['cost_per_usd_eal_reduction']:.2f}"
    t["best_tail_control"] = best_tail["name"]
    t["se_min"] = m(mt["reduction_eal_se"].min(), 3)
    t["se_max"] = m(mt["reduction_eal_se"].max(), 2)
    sd = float(mb["sd"])
    t["se_indep"] = m(np.sqrt(2) * sd / np.sqrt(50_000), 2)
    fp_rows = mt.set_index("mitigation")["residual_pfd_fire_protection_worst_producer"]
    t["fp_pfd_pump"] = f"{fp_rows['firewater_pump']:.3f}"
    t["fp_pfd_deluge"] = f"{fp_rows['deluge_coverage']:.3f}"
    t["fp_pfd_base"] = f"{s['appetite']['fire_protection_unavailability']['value']:.3f}"
    t["gen_ratio"] = f"{mt.set_index('mitigation').loc['generator_redundancy', 'cost_per_usd_eal_reduction']:.1f}"
    claim(best_ce["mitigation"] == "gas_detection_upgrade", "gas detection is the most cost-effective control")
    claim(best_tail["mitigation"] == "gas_detection_upgrade", "gas detection is the best tail reducer")
    low = mt.set_index("mitigation").loc[
        ["firewater_pump", "deluge_coverage", "emergency_response"], "cost_per_usd_eal_reduction"
    ]
    claim((low > 1).all(), "firewater, deluge and emergency-response controls cost more than they save in EAL")

    opt = json.loads((RES / "optimisation.json").read_text())
    names = {k: cfg.mitigations[k]["name"] for k in cfg.mitigations}
    rows = []
    for obj, title in (
        ("eal", "Minimise expected loss"),
        ("es99", "Minimise ES99 (tail)"),
        ("net_benefit", "Maximise net benefit (EAL reduction − annualised cost)"),
    ):
        o = opt[obj]
        rows.append(
            (
                title,
                " + ".join(names[k] for k in o["selected"]),
                m(o["capex"]),
                f"{m(o['simulated_eal_reduction'], 2)} M/yr",
                m(o["simulated_es99_reduction"]),
            )
        )
    t["table_portfolio"] = md(
        pd.DataFrame(
            rows, columns=["Objective", "Selected controls", "Capex (USD M)", "EAL reduction", "ES99 reduction (USD M)"]
        ),
        "llrrr",
    )
    claim(set(opt["eal"]["selected"]) != set(opt["es99"]["selected"]), "the optimal portfolio depends on the objective")
    costs = [float(cfg.mitigations[k]["capex_usd"]) for k in cfg.mitigations]
    t["n_feasible"] = str(
        sum(1 for r in range(len(costs) + 1) for c in combinations(costs, r) if sum(c) <= cfg.budget_usd + 1e-9)
    )
    t["budget"] = m(cfg.budget_usd)
    vo = {r["objective"]: r for r in v["optimisation"]}
    claim(all(r["milp_rank"] == 1 for r in vo.values()), "the MILP choice equals the exhaustive optimum")

    # ---------------------------------------------------------------- appetite
    ap = pd.read_csv(RES / "risk_appetite.csv")

    def fmt(val, unit):
        if unit.startswith("USD"):
            return f"{m(val)} M"
        if unit.startswith("days"):
            return f"{val:.1f} d"
        return f"{val:.3g}"

    rows = [
        (r.description, fmt(r.value, r.unit), fmt(r.limit, r.unit), f"{r.status} ({pct(r.utilisation)})") for r in ap.itertuples()
    ]
    t["table_appetite"] = md(pd.DataFrame(rows, columns=["Limit (hypothetical)", "Value", "Limit", "Status"]), "lrrl")

    # ---------------------------------------------------------------- counts, reliability, validation
    cnt = s["counts_per_year"]
    t["fires_per_year"] = f"{cnt['fires']:.3f}"
    t["releases_per_year"] = f"{s['derived_means']['rate_release']:.2f}"
    t["compressor_per_year"] = f"{s['derived_means']['rate_compressor']:.2f}"
    t["power_per_year"] = f"{s['derived_means']['rate_power_loss']:.2f}"
    th = cfg.registry.at_quantile(1, 0.5)
    tot = 8760 * repairable_koon_failure_frequency(2, 3, th["gen_lambda_per_h"], th["gen_mttr_h"], th["gen_ccf_beta"])[0]
    ccf = 8760 * th["gen_ccf_beta"][0] * th["gen_lambda_per_h"][0]
    t["power_median"] = f"{tot:.2f}"
    t["power_ccf_share"] = pct(ccf / tot)
    bu = pd.read_csv(RES / "bayesian_updates.csv").set_index("parameter")
    t["trip_post"] = f"{bu.loc['spurious_trip_rate', 'posterior_mean']:.1f}"
    t["pump_post"] = pct(bu.loc["fw_elec_pump_fts", "posterior_mean"], 1)
    t["pump_k"] = str(int(bu.loc["fw_elec_pump_fts", "observed_failures"]))
    t["pump_n"] = f"{int(bu.loc['fw_elec_pump_fts', 'exposure']):,}"
    t["pump_ci_red"] = pct(bu.loc["fw_elec_pump_fts", "ci_width_reduction"])
    t["n_tests_total"] = f"{int(bu.loc['fw_elec_pump_fts', 'exposure'] + bu.loc['fw_diesel_pump_fts', 'exposure']):,}"
    single = pd.read_csv(RES / "platform_single_outage.csv").set_index("platform")
    t["ppa_outage"] = pct(single.loc["PPA", "fraction_of_complex"])
    t["ppa_own"] = pct(asset.production_bopd[asset.index("PPA")] / asset.total_production)

    cvv = v["convergence_cv"]
    rows = [(f"{int(k):,}", *(pct(cvv[k][mm], 1) for mm in ("eal", "p95", "p99", "es99"))) for k in sorted(cvv, key=int)]
    t["table_convergence"] = md(pd.DataFrame(rows, columns=["Simulated years", "EAL", "P95", "P99", "ES99"]))
    t["cv100_eal"] = pct(cvv["100000"]["eal"], 1)
    t["cv100_p95"] = pct(cvv["100000"]["p95"], 1)
    t["cv100_p99"] = pct(cvv["100000"]["p99"], 1)
    t["cv100_es99"] = pct(cvv["100000"]["es99"], 1)
    t["cv10_p99"] = pct(cvv["10000"]["p99"], 1)
    t["cv10_es99"] = pct(cvv["10000"]["es99"], 1)
    claim(cvv["100000"]["eal"] < 0.01 and cvv["100000"]["p95"] < 0.01, "EAL and P95 are stable to within 1 % at 100k years")
    t["n_seeds"] = str(len(v["seeds"]))
    t["et_releases"] = f"{v['event_tree_releases']:,}"
    t["et_max_z"] = f"{v['event_tree_max_abs_z']:.1f}"
    t["coverage"] = pct(v["bayes_coverage"], 1)
    t["coverage_reps"] = f"{v['bayes_coverage_reps']:,}"
    t["n_tests"] = test_count()
    claim(v["event_tree_max_abs_z"] < 3, "event-tree shares inside the engine match the analytic tree")
    claim(abs(v["bayes_coverage"] - 0.9) < 0.03, "Bayesian 90 % intervals have ~90 % coverage")
    return t


def render(tokens: dict[str, str]) -> None:
    for name, target in TARGETS.items():
        text = (TEMPLATES / name).read_text(encoding="utf-8")
        missing = sorted(set(re.findall(r"\{\{(\w+)\}\}", text)) - set(tokens))
        if missing:
            raise KeyError(f"{name}: no value for tokens {missing}")
        out = re.sub(r"\{\{(\w+)\}\}", lambda mt: tokens[mt.group(1)], text)
        banner = f"<!-- Generated by scripts/render_docs.py from docs/templates/{name}. Edit the template, not this file. -->\n"
        target.write_text(banner + out, encoding="utf-8")
        print(f"rendered {target.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    render(build_tokens())
