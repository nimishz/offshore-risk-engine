"""Build and execute the example notebooks in notebooks/ (they only call the library)."""

from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"
SETUP = """import numpy as np, pandas as pd, matplotlib.pyplot as plt
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 20)
from offshore_risk import load_config
cfg = load_config()"""

NOTEBOOKS = {
    "01_data_exploration": [
        (
            "md",
            "# 01 · Synthetic data exploration\n\nThe records in `data/synthetic/` are generated **by the model itself** (`scripts/generate_synthetic_data.py`) from a documented synthetic truth. Exploring them checks that the generator behaves sensibly; it says nothing about real offshore facilities.",
        ),
        (
            "code",
            SETUP
            + "\nfrom offshore_risk.data import synthetic\nd = synthetic.load()\nlog, tests = d['incident_log'], d['proof_tests']\nlog.head()",
        ),
        ("code", "log.groupby(['year', 'subcategory']).size().unstack(fill_value=0)"),
        (
            "code",
            "summary = log.groupby('subcategory').agg(events=('event_id', 'size'), median_downtime_h=('downtime_h', 'median'),\n    total_lost_bbl=('lost_production_bbl', 'sum'), direct_cost_usd=('direct_cost_usd', 'sum'))\nsummary.sort_values('events', ascending=False)",
        ),
        ("code", "log['outcome'][log.subcategory == 'release'].value_counts()"),
        (
            "code",
            "t = tests.assign(fail=tests.result != 'pass').groupby('equipment_type')['fail'].agg(['sum', 'size', 'mean'])\nt.columns = ['failures', 'tests', 'observed_rate']; t",
        ),
        (
            "md",
            "**Observation.** Twelve years contain about a dozen releases, at most a minor fire and no major fire or explosion. A frequency estimated from this record alone would say 'major fires never happen'; this is why generic priors and event-tree logic, not only the operator's own history, are needed for major-accident risk. By contrast, spurious trips and pump start tests are numerous enough for the records to dominate the prior (see notebook 03).",
        ),
    ],
    "02_fault_tree_analysis": [
        (
            "md",
            "# 02 · Fault tree and event tree analysis\n\nThe fire-protection fault tree shares two basic events (main power, common-cause failure of the electric pumps) between branches. This notebook shows why that matters.",
        ),
        (
            "code",
            SETUP
            + "\nfrom offshore_risk.fault_tree.protection import fire_protection_tree, firewater_subtree, fire_protection_basic_events\nft = fire_protection_tree(2, 1)\nprint(ft.to_text())\nprint('repeated events:', ft.repeated_events)",
        ),
        (
            "code",
            "th = cfg.registry.at_quantile(1, 0.5)   # all parameters at their medians\nrows = []\nfor pp in (0.05, 0.10, 0.60):\n    be = fire_protection_basic_events(th, pp, 2, 1)\n    fw = firewater_subtree(2, 1)\n    rows.append({'P(power lost | fire)': pp, **{m: float(fw.probability(be, m)[0]) for m in ('independent', 'exact', 'rare_event', 'mcub', 'enumeration')}})\npd.DataFrame(rows)",
        ),
        (
            "md",
            "`independent` multiplies gate probabilities as if the two pump branches were independent; it underestimates firewater unavailability by roughly a factor of two here because the shared power supply is counted twice. `exact` (Shannon decomposition) agrees with brute-force `enumeration`.",
        ),
        (
            "code",
            "be = fire_protection_basic_events(th, 0.10, 2, 1)\ncs = [(' AND '.join(sorted(c)), float(np.prod([np.ravel(be[e])[0] for e in sorted(c)]))) for c in ft.minimal_cut_sets()]\npd.DataFrame(cs, columns=['minimal cut set', 'probability']).sort_values('probability', ascending=False).head(10)",
        ),
        ("code", "ft.importance(be).head(8)"),
        (
            "md",
            "## Event tree for a hydrocarbon release\nBranch probabilities for a medium gas release on PPA, at median parameters, derived from the reliability and fault-tree modules.",
        ),
        (
            "code",
            "from offshore_risk.asset import AssetModel\nfrom offshore_risk.simulation.derived import derive\nfrom offshore_risk.event_tree import RELEASE_TREE\nasset = AssetModel.from_config(cfg.asset); d = derive(th, asset); i = asset.index('PPA'); s = 1\np = {'p_detect': d.p_detect[0, s], 'p_isolate': d.p_isolate[0], 'p_ign': d.p_ign[0, s], 'p_ign_iso': d.p_ign[0, s] * th['ign_isolation_factor'][0],\n     'p_exp': d.p_exp[0, s], 'p_fp_ok': 1 - d.pfd_fp[0, i]}\nprint({k: round(float(v), 5) for k, v in p.items()})\nRELEASE_TREE.path_table(p)",
        ),
        ("code", "pd.Series({k: float(v) for k, v in RELEASE_TREE.outcome_probabilities(p).items()})"),
        (
            "md",
            "## Illustrative top event: major hydrocarbon fire on PPA within one year\nAND of (at least one release), (ignition), (failure of prevention/mitigation) is exactly what the event tree already computes per release; the annual frequency is the release rate times the conditional probability of the major-fire outcomes.",
        ),
        (
            "code",
            "rate = d.release_rate[0, i]\nmix = {o: 0.0 for o in ('major_fire', 'explosion', 'catastrophic')}\nfor s in range(3):\n    for gas, w in ((True, asset.gas_fraction[i]), (False, 1 - asset.gas_fraction[i])):\n        pp = {'p_detect': d.p_detect[0, s], 'p_isolate': d.p_isolate[0], 'p_ign': d.p_ign[0, s], 'p_ign_iso': d.p_ign[0, s] * th['ign_isolation_factor'][0],\n              'p_exp': d.p_exp[0, s] * (1 if gas else th['liquid_explosion_factor'][0]), 'p_fp_ok': 1 - d.pfd_fp[0, i]}\n        op = RELEASE_TREE.outcome_probabilities(pp)\n        for o in mix: mix[o] += d.size_probs[0, s] * w * float(op[o])\nfreq = rate * sum(mix.values())\nprint(f'release rate {rate:.3f}/yr; P(major fire or worse | release) {sum(mix.values()):.4f}; frequency {freq:.2e}/yr (1 in {1/freq:,.0f} yr)')",
        ),
    ],
    "03_reliability_analysis": [
        ("md", "# 03 · Reliability and Bayesian updating"),
        (
            "code",
            SETUP
            + "\nfrom offshore_risk.reliability import models as rm\nth = cfg.registry.at_quantile(1, 0.5)\nshape, scale = float(th['comp_weibull_shape'][0]), float(th['comp_weibull_scale_years'][0])\nages = np.arange(0, 6)\npd.DataFrame({'age since overhaul (y)': ages, 'expected failures next year (Weibull NHPP)': rm.nhpp_expected_failures(ages, ages + 1, shape, scale),\n              'constant-rate equivalent': 1 / float(rm.weibull_mean(shape, scale))})",
        ),
        (
            "md",
            "With a wear-out shape (β > 1) the expected number of failures in the coming year depends on time since overhaul; a constant-rate model would give the same answer at every age and could not value maintenance timing.",
        ),
        (
            "code",
            "lam, T = 2.5e-6, 8760\nrows = [{'architecture': a, 'beta': b, 'PFDavg': float(rm.pfd_koon(k, n, lam, T, b))} for a, k, n in (('1oo1', 1, 1), ('1oo2', 1, 2), ('2oo3', 2, 3)) for b in (0.0, 0.05, 0.10)]\npd.DataFrame(rows).pivot(index='architecture', columns='beta', values='PFDavg')",
        ),
        (
            "md",
            "Common cause dominates: for 1oo2 valves the β = 10 % term is several times the independent term, so adding redundancy without addressing common cause buys little.",
        ),
        (
            "code",
            "lam_g, mttr, beta = 1.5e-4, 24, 0.05\npd.DataFrame([{'generators': f'{k}oo{n}', 'independent /yr': 8760 * rm.repairable_koon_failure_frequency(k, n, lam_g, mttr, 0),\n              'with CCF /yr': 8760 * rm.repairable_koon_failure_frequency(k, n, lam_g, mttr, beta)} for k, n in ((2, 3), (2, 4))])",
        ),
        (
            "md",
            "A fourth generator removes most of the *independent* loss-of-power frequency but leaves the common-cause term (shared fuel gas, switchboard) untouched, which is why that control scores poorly in the mitigation analysis.",
        ),
        ("md", "## Bayesian updating from synthetic records"),
        (
            "code",
            "from offshore_risk.bayesian.calibration import calibrate, summary_table\ncal = calibrate(cfg)\nsummary_table(cal)",
        ),
        (
            "code",
            "from offshore_risk.bayesian.conjugate import BetaModel\nfrom offshore_risk.visualization import plots\nc = cal['fw_elec_pump_fts']; grid = np.linspace(1e-5, 0.06, 500)\nplots.bayes_update(c['prior'], c['posterior'], lambda x: BetaModel.likelihood_density(x, c['failures'], c['exposure']), grid, 'Electric firewater pump fail-to-start', 'P(fail to start)')",
        ),
        (
            "md",
            "The generic prior (mean 1 %) was optimistic for these synthetic pumps; about 1,250 tests move the posterior to ~2.7 % and cut the 90 % interval width roughly in half. The posterior (not the prior) is what the main analysis uses.",
        ),
    ],
    "04_monte_carlo": [
        ("md", "# 04 · Monte Carlo loss distribution, dependence and uncertainty layers"),
        (
            "code",
            SETUP
            + "\nfrom offshore_risk.bayesian.calibration import apply_posteriors\nfrom offshore_risk.simulation import simulate, SOURCES\nfrom offshore_risk.financial.metrics import risk_metrics, tail_allocation\nfrom offshore_risk.visualization import plots\ncfg = apply_posteriors(cfg)\nb = simulate(cfg, n_years=50_000, dependence='correlated')\na = simulate(cfg, n_years=50_000, dependence='independent')\npd.DataFrame({'Model A': risk_metrics(a.total), 'Model B': risk_metrics(b.total)}).T[['eal', 'median', 'p90', 'p95', 'p99', 'es99']] / 1e6",
        ),
        ("code", "plots.exceedance_curves({'Model A (independent)': a.total, 'Model B (correlated)': b.total})"),
        (
            "md",
            "Correlated year-level drivers raise the middle-to-upper part of the distribution (P90–P95, where years with several moderate events accumulate) but hardly change the extreme tail, which is set by single major-accident events.",
        ),
        ("code", "tail_allocation(b.loss_by_source, SOURCES, 0.99)"),
        ("code", "b.component_frame().mean().to_frame('mean USD/yr')"),
        (
            "md",
            "## Epistemic vs aleatory\nEach 'world' fixes the uncertain parameters; the spread within a world is year-to-year randomness, the spread between worlds is what we do not know.",
        ),
        (
            "code",
            "from offshore_risk.simulation.nested import run_nested\nnr = run_nested(cfg, n_outer=150, n_inner=800)\nnr.metric_summary()",
        ),
        ("code", "plots.exceedance_band(nr, b.total)"),
    ],
    "05_sensitivity_analysis": [
        (
            "md",
            "# 05 · Which assumptions matter most?\nThe full-size results are produced by `scripts/run_analysis.py`; this notebook reads them and reproduces a small version of each method.",
        ),
        ("code", SETUP + "\nres = '../outputs/results/'\npd.read_csv(res + 'parameter_importance_summary.csv')"),
        (
            "code",
            "tor = pd.read_csv(res + 'tornado.csv')\ntor[['parameter', 'value_low', 'value_high', 'eal_low', 'eal_high', 'eal_swing', 'p99_swing']].head(10)",
        ),
        ("code", "pd.read_csv(res + 'rank_correlation_es99.csv').head(8)"),
        ("md", "## Small live example (tornado on four parameters)"),
        (
            "code",
            "from offshore_risk.sensitivity.analysis import tornado\nfrom offshore_risk.visualization import plots\nsmall = tornado(cfg, n_years=5000, params=['bi_value_fraction', 'pipeline_rate_per_km_yr', 'p_ign_large', 'fw_header_fail'])\nplots.tornado(small, 'eal')",
        ),
        (
            "md",
            "**Reading.** The single most influential assumption is economic: the value of a deferred barrel (`bi_value_fraction`). Engineering parameters that drive *frequent* losses (release cause rates, compressor life, repair times) dominate the EAL; for the tail, the pipeline failure rate and repair duration join the release cause rates at the top. Below the top parameter the ranking is noisy (rank-correlation sampling error ≈ ±0.05 with 400 worlds), so read ranks 2–8 as a group. Protection-system reliability parameters (firewater, deluge) barely move either metric at this asset's assumed fire frequency — they matter for safety and for a catastrophic tail that is dominated by a handful of simulated years.",
        ),
    ],
}


def _single_figure(src: str) -> str:
    """Assign a trailing plots.* call so the inline backend shows the figure once, not twice."""
    lines = src.rstrip().split("\n")
    if lines[-1].startswith("plots."):
        lines[-1] = "fig = " + lines[-1]
    return "\n".join(lines)


def build(execute: bool = True):
    NB.mkdir(exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        nb = nbf.v4.new_notebook()
        nb.cells = [
            nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(_single_figure(src)) for kind, src in cells
        ]
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        if execute:
            ExecutePreprocessor(timeout=900, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(NB)}})
        nbf.write(nb, NB / f"{name}.ipynb")
        print("built", name)


if __name__ == "__main__":
    build()
