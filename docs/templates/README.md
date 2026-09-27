# Offshore Asset Risk & Resilience Engine

A quantitative operational-risk model for a **fictional** six-platform offshore
oil and gas complex. It follows operational hazards from failure frequencies,
through protection-layer reliability and event sequences, to physical damage,
production interruption and an annual loss distribution. It then asks which
assumptions drive that distribution and which controls reduce it most for a
fixed budget.

> **Scope.** The asset, every parameter and all "historical" records are
> fictional or synthetic. No confidential or company data is used, and no value
> should be read as a statistic for any real facility or operator. The model
> demonstrates methods and their sensitivities; it does not predict real
> accidents. Every parameter carries its basis, rationale and uncertainty range
> in the [data dictionary](docs/data_dictionary.md).

---

## Contents

1. [Why this project exists](#1-why-this-project-exists)
2. [The risk problem](#2-the-risk-problem)
3. [System architecture](#3-system-architecture)
4. [Methodology](#4-methodology)
5. [Data methodology](#5-data-methodology)
6. [Results](#6-results)
7. [Dashboard](#7-dashboard)
8. [Key findings](#8-key-findings)
9. [Limitations](#9-limitations)
10. [How to run](#10-how-to-run)
11. [Testing and quality checks](#11-testing-and-quality-checks)
12. [Future improvements](#12-future-improvements)

## 1. Why this project exists

Offshore fire-protection and HSE engineering usually asks *"is this barrier
adequate?"* Enterprise risk management asks *"what is our exposure, how
uncertain is it, and where should the next dollar go?"* This repository connects
the two. The safety engineer's tools (fault trees, event trees, PFD
calculations, common-cause modelling) feed the ERM function's tools (loss
distributions, VaR and expected shortfall, appetite limits, portfolio
optimisation, sensitivity analysis). Each layer is simple enough to check by hand.

## 2. The risk problem

For the fictional complex *Meridian* (40 kbbl/d, six bridge-linked platforms):

1. What does the **distribution** of annual operational loss look like, not just its mean?
2. Which sources drive the **average year**, and which drive the **tail**?
3. Where does **system architecture** (shared power, common cause) undermine apparent redundancy?
4. How much of the uncertainty is **randomness**, and how much is **not knowing the parameters**?
5. With a **USD {{budget}} M** budget, which controls should be funded? Does the answer depend on the risk measure?

## 3. System architecture

```text
config/*.yaml ─► load_config ─► validate_config (schema, ranges, cross-references)
                    │  ParameterRegistry: value, unit, distribution, basis, rationale
                    │  θ (epistemic draw)
   reliability ─► fault_tree ─► simulation/derived ◄── bayesian (posteriors from synthetic records)
   (PFD, k-of-n,   (exact,       (rates, PFDs,
    Weibull NHPP,   cut sets,     branch probabilities)
    CCF)            importance)        │
                                       ▼
   asset (dependency graph → 64-state capacity table)   event_tree (release sequences)
                                       │
                                       ▼
             simulation/engine: year drivers (Model A / B) → thinned Poisson events
             → event-tree outcomes, escalation, downtime → financial/loss_model
                                       │
      ┌──────────────┬─────────────┬───┴────────┬──────────────┬──────────────┐
   metrics        nested        scenarios    mitigation →   sensitivity    appetite
 (EAL, VaR, ES,  (epistemic vs  (7 stress    optimization   (tornado,      (hypothetical
  LEC, Euler     aleatory)      cases)       (MILP)         Spearman, S1)   limits)
  allocation)
```

Business logic lives in `src/offshore_risk/`; the dashboard, notebooks and
plotting code only call it. Full design: [docs/methodology.md](docs/methodology.md).

```text
offshore-risk-engine/
├── config/          asset, risk parameters, mitigations, scenarios, risk appetite (YAML, validated on load)
├── data/            raw/ (empty by design), synthetic/ (generated), processed/
├── src/offshore_risk/
│   ├── reliability/  fault_tree/  event_tree/  bayesian/  asset/
│   ├── simulation/   financial/   scenarios/   appetite/  mitigation/
│   ├── optimization/ sensitivity/ data/        visualization/
│   └── config.py, config_validation.py, distributions.py
├── scripts/         generate data, run analysis, run validation, render docs, build notebooks
├── notebooks/       01–05, executed; thin wrappers around the library
├── dashboard/       Streamlit app
├── tests/           pytest suite ({{n_tests}} tests)
├── outputs/         results/ (CSV, JSON) and figures/ (PNG) from the full run
└── docs/            methodology, assumptions, data dictionary, validation, limitations, research notes
```

## 4. Methodology

| Layer | Method | Why it is needed |
|---|---|---|
| Failure frequency | Release rate = sum of cause rates (valve, flange, corrosion, PTW × isolation failure, overpressure demand × PSV PFD); Weibull power-law NHPP for compressors; repairable 2oo3 + β-factor for power | Controls act on causes; wear-out makes overhaul timing matter; redundancy vs common cause |
| Protection layers | PFDavg (simplified IEC 61508-6 forms) for k-oo-n voted groups with β-factor CCF; fire-protection fault tree evaluated **exactly** by Shannon decomposition on shared events | The electric firewater pumps share main power; naive multiplication understates firewater unavailability {{fw_factor}}× |
| Event sequences | Release event tree: detection → isolation → ignition → explosion → fire protection → escalation across bridges | Converts barrier reliability into outcome probabilities |
| Interdependency | Directed dependency graph (power, well fluids, export route, manning, water injection) → capacity for all 64 down-states | A PPA outage costs {{ppa_outage}} of output, not its own {{ppa_own}} |
| Uncertainty | Epistemic parameters (per world) vs aleatory variability (per year and event); pooled and **two-loop nested** simulation | Separates "what we don't know" from "what is random" |
| Dependence | Year-level drivers (weather, logistics, integrity): Model A independent, Model B Gaussian copula with identical marginals | Isolates the effect of co-movement on the tail |
| Finance | Direct (damage, repair, emergency response, logistics, restart) vs indirect (lost bbl × price × value fraction); environmental proxy monetised only as an explicit scenario assumption | Keeps speculative costs out of the totals |
| Bayesian | Beta-Binomial (pump start tests) and Gamma-Poisson (trip rate) on synthetic records; posteriors feed the model | Shows how evidence narrows uncertainty |
| Decisions | Controls as parameter transforms, re-simulated with **common random numbers**; MILP with pairwise interaction terms, checked against exhaustive search | Measured, not assumed, risk reduction |
| Sensitivity | Tornado (P10/P90, common random numbers), Spearman on nested worlds, binned first-order variance index | Answers "which assumptions matter most?" |

### Key equations

* Release frequency (platform *p*): $\lambda_p = k_p[\lambda_{valve}+\lambda_{flange}+\lambda_{corr}+n_{jobs}q_{PTW}p_{rel}+\lambda_{dem}\mathrm{PFD}_{PSV}]$
* Weibull NHPP expected failures in the coming year: $\Lambda = ((a+1)/\eta)^\beta-(a/\eta)^\beta$
* PFD of a *k*oo*n* group: $\binom{n}{m}\frac{((1-\beta)\lambda_{DU}T)^m}{m+1}+\beta\frac{\lambda_{DU}T}{2}$, $m=n-k+1$
* Exact fault-tree probability with a shared event *x*: $P = p_x P(\cdot|x{=}1)+(1-p_x)P(\cdot|x{=}0)$
* Business interruption: lost bbl × oil price × value fraction; lost bbl $=\sum_k (t_k-t_{k-1})(Q-C(S_k))$
* $\mathrm{ES}_\alpha = E[L \mid L\ge \mathrm{VaR}_\alpha]$; Euler allocation $\mathrm{ES}_\alpha=\sum_s E[L_s\mid L\ge\mathrm{VaR}_\alpha]$
* Portfolio: $\max \sum a_i x_i+\sum b_{ij}y_{ij}$ s.t. $\sum c_i x_i\le B$, with $y_{ij}$ linearising $x_ix_j$

## 5. Data methodology

* **No confidential data.** `data/raw/` is empty by design.
* **Parameters** are assumptions recorded in YAML with definition, unit,
  distribution, basis (`assumption`, `order_of_magnitude`, `engineering_convention`,
  `synthetic_calibrated`), rationale and uncertainty. The configuration is
  validated on load (required fields, probability ranges, cross-references, an
  acyclic dependency graph). The [data dictionary](docs/data_dictionary.md) is
  generated from the same YAML.
* **Synthetic records** (12 years: incident log, {{n_tests_total}} pump start tests,
  equipment register, exposure) are generated by the model from a documented
  synthetic truth with a fixed seed. They exercise the Bayesian module; the
  argument is circular by construction and is labelled as such.
* **No invented citations.** "Order of magnitude" parameters name the *type* of
  public source whose range they sit in (regulator release statistics,
  reliability handbooks, pipeline loss-of-containment compilations). The
  numbers themselves are assumptions.
* **Numbers in this README are generated** from `outputs/results/` by
  `scripts/render_docs.py`, which also checks that the qualitative statements
  below still hold.

## 6. Results

Unless stated otherwise: `python scripts/run_analysis.py`, seed {{seed}}, Model B,
posterior-updated parameters, {{n_main}} simulated years. All money in USD millions.

### Annual loss distribution

| Metric | USD M | Reading |
|---|---:|---|
| Expected annual loss (EAL) | **{{eal}}** (90 % CI {{eal_ci}}) | long-run average; basis for cost-benefit |
| Median year | {{median}} | a typical year is {{median_share}} of the EAL |
| P75 / P90 | {{p75}} / {{p90}} | |
| P95 = VaR95 | **{{p95}}** | 1-in-20-year loss |
| P99 = VaR99 | **{{p99}}** ({{p99_ci}}) | 1-in-100-year loss |
| ES95 / ES99 | {{es95}} / **{{es99}}** ({{es99_ci}}) | mean of the worst 5 % / 1 % of years |
| P(loss > 100 M) | {{p_gt_100}} | about 1 year in {{rp_100}} |
| P(loss > 500 M) | {{p_gt_500}} | about 1 year in {{rp_500}} |

Intervals are 90 % bootstrap intervals for Monte Carlo error only; parameter
uncertainty is shown separately below.

![Loss distribution](outputs/figures/loss_histogram.png)

Business interruption is {{bi_share}} of expected loss (USD {{bi_musd}} M/yr). Direct costs
(restart, repair, damage, emergency response, logistics) make up the rest. The
worst 1 % of years carry {{top1_share}} of all loss.

### The average year and the tail are driven by different things

{{table_tail}}

![Mean vs tail contributions](outputs/figures/contribution_mean_vs_tail.png)

### Architecture matters: shared power under redundant pumps

At median parameters, the firewater probability of failure on demand for a
fire on a production platform is **{{fw_exact_ppa}}** when evaluated exactly and
**{{fw_naive_ppa}}** if the electric pumps are (wrongly) treated as independent.
The shared main-power supply accounts for the difference. For a fire on the
utility platform, where main power is likely to be lost, the exact value is
{{fw_exact_ucp}}. The most important basic event in the fire-protection fault tree
is `{{fv_top_event}}` (Fussell–Vesely {{fv_top}}).

### Dependence: Model A vs Model B

{{table_dependence}}

Correlated year-level drivers change P90 by {{dep_p90}} and P95 by {{dep_p95}}.
That is where bad years are built from several moderate events. ES99 changes by
only {{dep_es99}}, because the extreme tail is made of single major accidents. Event
frequencies are identical in both models (ratio {{dep_freq_ratio}}, validated).

![LEC A vs B](outputs/figures/lec_model_a_vs_b.png)

### Uncertainty vs risk

A two-loop run ({{n_worlds}} parameter "worlds" × {{n_inner}} years) separates the two:

{{table_nested}}

Parameter uncertainty alone spans a factor of about {{nested_eal_ratio}} in the EAL and
{{nested_tail_ratio}} in the tail metrics (P99, ES99). The randomness of a year can only be absorbed
(resilience, insurance, contingency). The uncertainty about the parameters can
be *reduced* with data, and the sensitivity analysis shows where to collect it.

![Epistemic band](outputs/figures/lec_epistemic_band.png)

### Which assumptions matter most?

{{table_sensitivity}}

`{{top_param}}` (the economic value of a deferred barrel) is clearly first (ρ = {{top_rho}},
first-order index {{top_s1}}). The ordering of the next few parameters is **not
robust**: from rank 2 onwards (ρ = {{second_rho}} and below), neighbouring correlations
differ by less than the sampling error of ρ (≈ ±{{rho_se}} with {{n_worlds}} worlds), so ranks
2–8 are best read as one group of second-order drivers. For the tail (ES99), the
leading parameters are {{tail_params}}. Protection-system reliability
parameters do not appear among the leading drivers at this asset's fire
frequency (~{{fires_per_year}} fires/yr). The tornado's base case (all parameters at their medians) has an EAL of
{{tornado_base}}, below the pooled EAL, because the parameter distributions are right-skewed.

![Tornado](outputs/figures/tornado_eal.png)

### Stress scenarios (conditional)

{{table_scenarios}}

S2 is instructive. Degrading the firewater system raises the fire-protection
PFD from {{s2_pfd_base}} to {{s2_pfd}} but adds only USD {{s2_incr}} M to the expected annual loss. An
expected-value view alone would under-rate firewater integrity. (Scenarios use
{{n_scenarios}} years with the same random numbers as the baseline.)

![Scenarios](outputs/figures/scenarios.png)

### Controls and the budget

{{table_mitigations}}

**The optimal portfolio depends on the objective.** The MILP result is confirmed by exhaustive search over all {{n_feasible}} feasible portfolios:

{{table_portfolio}}

![Mitigations](outputs/figures/mitigations.png)

### Risk appetite

{{table_appetite}}

All limits are hypothetical management thresholds for the fictional operator,
not industry or regulatory criteria.

## 7. Dashboard

`streamlit run dashboard/app.py` opens six views: Executive, Asset, Scenarios,
Monte Carlo, Mitigation and Sensitivity. The sidebar sets the sample size,
dependence model, Bayesian updating, seed and the controls in place, and every
view updates from the library. Started with `--server.address localhost` it is
reachable only from your own machine; usage statistics are disabled
(`.streamlit/config.toml`). The same app deploys unchanged to Streamlit Community Cloud. Screenshots use the
default 20,000 simulated years, so their figures differ slightly from the
results above.

| Executive | Scenarios |
|---|---|
| ![](docs/img/dashboard_executive.png) | ![](docs/img/dashboard_scenarios.png) |
| **Mitigation** | **Sensitivity** |
| ![](docs/img/dashboard_mitigation.png) | ![](docs/img/dashboard_sensitivity.png) |

## 8. Key findings

These apply to this fictional asset under these assumptions.

1. **The economic assumption outweighs the engineering ones.** What a deferred
   barrel is worth moves the EAL more than any failure rate. It should be pinned
   down first.
2. **Mean and tail are different problems.** Compressors, weather, trips and
   power losses are {{frequent_share_eal}} of the EAL but {{frequent_share_es99}} of ES99.
   Releases and the export pipeline are {{major_share_es99}} of ES99.
3. **Redundancy is weaker than it looks where support systems are shared.**
   Modelling the common power supply raises firewater unavailability {{fw_factor}}×. A
   fourth generator costs USD {{gen_ratio}} per dollar of EAL reduced, because the
   common-cause loss of power ({{power_ccf_share}} of the total at median parameters) is untouched.
4. **Detection and isolation are the economically decisive barrier.**
   {{best_ce_control}} is the most cost-effective control (USD {{best_ce_ratio}} per dollar
   of EAL reduced) and the largest tail reducer. Undetected releases become
   unisolated ones, which shut the complex down even when they do not ignite.
5. **Firewater, deluge and emergency-response controls look poor on expected
   value** but reduce fire-protection unavailability (from {{fp_pfd_base}} to
   {{fp_pfd_pump}} with an extra pump, {{fp_pfd_deluge}} with deluge duplication). Funding them
   is a safety and appetite decision, not an expected-value one.
6. **Correlation between risks thickens the upper-middle of the distribution**
   (P95 {{dep_p95}}), not the extreme tail (ES99 {{dep_es99}}).
7. **Parameter uncertainty is as large as the risk signal** (EAL {{nested_eal_range}} M across
   plausible worlds), so collecting data on the top-ranked parameters has real value.

## 9. Limitations

The main ones are listed here; the full list is in
[docs/limitations.md](docs/limitations.md).

* Economic risk only: no individual or societal risk.
* No consequence physics: fire and explosion outcomes are classes with assumed damage and downtime.
* Simplified PFD formulas and β-factor CCF.
* A Gaussian copula, which has no tail dependence.
* Gross, pre-insurance losses.
* Blowouts and subsea equipment are not modelled.
* The synthetic data is generated by the model itself.

## 10. How to run

```bash
git clone <repository-url> offshore-risk-engine && cd offshore-risk-engine
python -m venv .venv && source .venv/bin/activate      # Python 3.10+
pip install -r requirements.txt && pip install -e .

python scripts/generate_synthetic_data.py   # data/synthetic/*.csv (fixed seed)
python scripts/build_data_dictionary.py     # docs/data_dictionary.md
python scripts/run_analysis.py              # outputs/results + outputs/figures (~2 min)
python scripts/run_validation.py            # docs/validation_results.md (~2 min)
python scripts/render_docs.py               # README.md and docs/ numbers from outputs/
python scripts/build_notebooks.py           # executes notebooks/01–05
streamlit run dashboard/app.py --server.address localhost   # local only
```

`make all` runs the whole chain. From Python:

```python
from offshore_risk import load_config
from offshore_risk.simulation import simulate
from offshore_risk.financial.metrics import risk_metrics

cfg = load_config()                       # validated; raises ConfigError listing every problem
res = simulate(cfg, n_years=50_000, dependence="correlated", mitigations=["gas_detection_upgrade"])
print(risk_metrics(res.total))
```

To change an assumption, edit `config/*.yaml` and re-run; there are no
parameter values in the model code. Set `OFFSHORE_RISK_CONFIG` to point at an
alternative configuration directory.

## 11. Testing and quality checks

```bash
pytest                                   # {{n_tests}} tests
ruff check . && ruff format --check .    # lint and formatting
```

The tests check the model against:
* closed forms;
* brute-force enumeration (fault trees, the optimiser);
* numerical integration (Bayesian posteriors, PFDs);
* independent Monte Carlo and discrete-event simulation (standby, 1oo2, repairable 2oo3);
* engine invariants (reproducibility, accounting identities, common-random-number monotonicity, duration caps);
* configuration validation.

Model-level validation is in [docs/validation.md](docs/validation.md): convergence at 10k/50k/100k years, the event
tree inside the engine, the dependence models, MILP vs exhaustive search,
Bayesian coverage and edge cases. At 100k years the EAL and P95 vary by
{{cv100_eal}} and {{cv100_p95}} across seeds, P99 by {{cv100_p99}}. GitHub Actions runs lint,
tests and a quick end-to-end analysis on every push
(`.github/workflows/ci.yml`).

## 12. Future improvements

* Consequence modelling from physics (jet-fire length, explosion overpressure) instead of outcome classes.
* Individual and societal risk (PLL, F-N curves), with temporary-refuge impairment.
* Alpha-factor CCF, partial-stroke and imperfect proof testing, and a SIL verification mode.
* Hierarchical Bayesian models pooling across similar equipment, with test-coverage correction.
* A t-copula or event-level common causes (storm → collision) for tail dependence.
* An insurance programme (deductibles, limits, BI waiting periods) to move from gross to net loss.
* Multi-year simulation with decline curves, deferred-production recovery and maintenance schedules.
* A full Saltelli/Sobol' design for interaction effects.

---

*Licence: MIT. Fictional asset; synthetic data; for research and education.*
