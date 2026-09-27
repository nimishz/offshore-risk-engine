# Offshore Asset Risk & Resilience Engine

A quantitative operational-risk model for a **fictional** six-platform offshore
oil and gas complex. It follows operational hazards from failure frequencies,
through protection-layer reliability and event sequences, to physical damage,
production interruption and an annual loss distribution. It then asks which
assumptions drive that distribution and which controls reduce it most for a
fixed budget.

> **Scope and honesty statement.** The asset, every parameter and all
> "historical" records are fictional or synthetic. No confidential or company
> data is used, and no value should be read as a statistic for any real facility
> or operator. The model shows how the methods work and what they are sensitive
> to. It does not predict real accidents. Every parameter carries its basis,
> rationale and uncertainty range in the [data dictionary](docs/data_dictionary.md).

---

## 1. Why this project exists

Offshore fire-protection and HSE engineering usually asks *"is this barrier
adequate?"* Enterprise risk management asks *"what is our exposure, how
uncertain is it, and where should the next dollar go?"* This repository connects
the two. It takes the tools a safety engineer uses (fault trees, event trees,
PFD calculations, common-cause modelling) and feeds them into the tools an ERM
function uses (loss distributions, VaR/expected shortfall, appetite limits,
portfolio optimisation, sensitivity analysis). Each layer is kept transparent
enough to check by hand.

## 2. The risk problem

For the fictional complex *Meridian* (40 kbbl/d, bridge-linked):

1. What does the **distribution** of annual operational loss look like, not just its mean?
2. Which sources drive the **average year**, and which drive the **tail**?
3. Where does **system architecture** (shared power, common cause) undermine apparent redundancy?
4. How much of the uncertainty is **randomness** and how much is **not knowing the parameters**?
5. With **USD 5 M**, which controls should be funded? Does the answer depend on the risk measure?

## 3. System architecture

```text
config/*.yaml ─► ParameterRegistry (value, unit, distribution, basis, rationale)
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

Business logic lives in `src/offshore_risk/`; `dashboard/`, `notebooks/` and
`visualization/` only call it. Full design: [docs/methodology.md](docs/methodology.md).

```text
offshore-risk-engine/
├── config/          asset, risk parameters, mitigations, scenarios, risk appetite (YAML)
├── data/            raw/ (empty by design), synthetic/ (generated), processed/
├── src/offshore_risk/
│   ├── reliability/  fault_tree/  event_tree/  bayesian/  asset/
│   ├── simulation/   financial/   scenarios/   appetite/  mitigation/
│   ├── optimization/ sensitivity/ data/        visualization/
├── scripts/         generate data, run analysis, run validation, build notebooks/dictionary
├── notebooks/       01–05, executed; thin wrappers around the library
├── dashboard/       Streamlit app
├── tests/           pytest suite (106 tests)
├── outputs/         results/ (CSV, JSON) and figures/ (PNG) from the full run
└── docs/            methodology, assumptions, data dictionary, validation, limitations, research notes
```

## 4. Methodology in brief

| Layer | Method | Why it is needed |
|---|---|---|
| Failure frequency | Release rate = sum of cause rates (valve, flange, corrosion, PTW × isolation failure, overpressure demand × PSV PFD); Weibull power-law NHPP for compressors; repairable 2oo3 + β-factor for power | Controls act on causes; wear-out makes overhaul timing matter; redundancy vs common cause |
| Protection layers | PFDavg (IEC 61508-6-type simplified forms) for k-oo-n voted groups with β-factor CCF; fire-protection fault tree evaluated **exactly** by Shannon decomposition on shared events | The two electric firewater pumps share main power; naive multiplication halves the firewater PFD |
| Event sequences | Release event tree: detection → isolation → ignition → explosion → fire protection → escalation across bridges | Converts barrier reliability into outcome probabilities |
| Interdependency | Directed dependency graph (power, well fluids, export route, manning, water injection) → capacity for all 64 down-states | A PPA outage costs 78 % of output, not 55 %; a UCP outage costs 100 % |
| Uncertainty | Epistemic parameters (per world) vs aleatory variability (per year/event); pooled and **two-loop nested** simulation | Separates "what we don't know" from "what is random" |
| Dependence | Year-level drivers (weather, logistics, integrity): Model A independent, Model B Gaussian copula with identical marginals | Isolates the effect of co-movement on the tail |
| Finance | Direct (damage, repair, emergency response, logistics, restart) vs indirect (lost bbl × price × value fraction); environmental proxy only as an explicit scenario assumption | Keeps speculative costs out of the totals |
| Bayesian | Beta-Binomial (pump start tests), Gamma-Poisson (trip rate) on synthetic records; posteriors feed the model | Shows how evidence narrows uncertainty |
| Decisions | Controls as parameter transforms, re-simulated with **common random numbers**; MILP with pairwise interaction terms, checked against exhaustive search | Measured, not assumed, risk reduction |
| Sensitivity | Tornado (P10/P90, CRN), Spearman on nested worlds, binned first-order variance index | Answers "which assumptions matter most?" |

### Key equations

* Release frequency (platform *p*): $\lambda_p = k_p[\lambda_{valve}+\lambda_{flange}+\lambda_{corr}\phi_{insp}+n_{jobs}q_{PTW}p_{rel}+\lambda_{dem}\mathrm{PFD}_{PSV}]$
* Weibull NHPP expected failures in the coming year: $\Lambda = ((a+1)/\eta)^\beta-(a/\eta)^\beta$
* PFD of a *k*oo*n* group: $\binom{n}{m}\frac{((1-\beta)\lambda_{DU}T)^m}{m+1}+\beta\frac{\lambda_{DU}T}{2}$, $m=n-k+1$
* Exact fault-tree probability with a shared event *x*: $P = p_x P(\cdot|x{=}1)+(1-p_x)P(\cdot|x{=}0)$
* Business interruption: lost bbl × oil price × value fraction; lost bbl $=\sum_k (t_k-t_{k-1})(Q-C(S_k))$
* $\mathrm{ES}_\alpha = E[L \mid L\ge \mathrm{VaR}_\alpha]$; Euler allocation $\mathrm{ES}_\alpha=\sum_s E[L_s\mid L\ge\mathrm{VaR}_\alpha]$
* Portfolio: $\max \sum a_i x_i+\sum b_{ij}y_{ij}$ s.t. $\sum c_i x_i\le B$, $y_{ij}$ linearising $x_ix_j$

## 5. Data methodology

* **No confidential data.** `data/raw/` is empty by design.
* **Parameters** are assumptions recorded in YAML with definition, unit,
  distribution, basis (`assumption`, `order_of_magnitude`, `engineering_convention`,
  `synthetic_calibrated`), rationale and uncertainty. The
  [data dictionary](docs/data_dictionary.md) is generated from that YAML, so the
  two cannot drift apart.
* **Synthetic records** (12 years: incident log, 1,878 pump start tests,
  equipment register, exposure) are generated by the model from a documented
  synthetic truth (fixed seed). They exercise the Bayesian module; the argument
  is circular by construction and is labelled as such.
* **No invented citations.** "Order of magnitude" parameters name the *type* of
  public source whose range they sit in (regulator release statistics,
  reliability handbooks, pipeline loss-of-containment compilations); the numbers
  themselves are assumptions.

## 6. Example results

All figures below come from `python scripts/run_analysis.py` (seed 20260927,
Model B, posterior-updated parameters, 100,000 simulated years unless stated).

### Annual loss distribution

| Metric | USD M | Reading |
|---|---:|---|
| Expected annual loss (EAL) | **22.6** (90 % CI 22.3–22.9) | long-run average; basis for cost-benefit |
| Median year | 13.6 | a typical year is ~60 % of the EAL |
| P75 / P90 | 24.9 / 42.4 | |
| P95 = VaR95 | **61.5** | 1-in-20-year loss |
| P99 = VaR99 | **180** (173–187) | 1-in-100-year loss |
| ES95 / ES99 | 145 / **357** (343–372) | average of the worst 5 % / 1 % of years |
| P(loss > 100 M) | 2.2 % | about 1 year in 45 |
| P(loss > 500 M) | 0.15 % | about 1 year in 680 |

![Loss distribution](outputs/figures/loss_histogram.png)

Business interruption is 82 % of expected loss (USD 18.6 M/yr); direct costs
(restart, repair, damage, emergency response, logistics) are the rest.

### The average year and the tail are driven by different things

| Source | Share of EAL | Share of ES99 |
|---|---:|---:|
| Hydrocarbon releases (incl. fires/explosions) | 36 % | **64 %** |
| Compressor failures | 23 % | 2 % |
| Extreme weather shut-ins | 14 % | 1 % |
| Spurious trips | 13 % | 1 % |
| Export pipeline | 10 % | **30 %** |
| Power loss, vessel collision | 4 % | <1 % |

![Mean vs tail contributions](outputs/figures/contribution_mean_vs_tail.png)

### Architecture matters: shared power under redundant pumps

At median parameters, the firewater system's probability of failure on demand
for a fire on a production platform is **0.0060** when evaluated exactly, and
**0.0028** if the two electric pumps are (wrongly) treated as independent. The
shared main-power supply is the reason. For a fire on the utility platform,
where main power is likely to be lost, the exact value rises to 0.025.
Deluge-valve failure dominates the fire-protection fault tree (Fussell–Vesely
68 %).

### Dependence: Model A vs Model B

| | EAL | P90 | P95 | P99 | ES99 |
|---|---:|---:|---:|---:|---:|
| Model A (independent drivers) | 22.1 | 40.3 | 56.5 | 174 | 358 |
| Model B (correlated drivers) | 22.6 | 42.4 | 61.5 | 180 | 357 |

Correlated year-level drivers raise the P90–P95 region by 5–9 %, where bad
years are built from several moderate events. They leave the extreme tail (ES99)
unchanged, because that tail is made of single major accidents. Event
frequencies are the same in both models (validated).

![LEC A vs B](outputs/figures/lec_model_a_vs_b.png)

### Uncertainty vs risk

A two-loop run (400 parameter "worlds" × 1,000 years) separates the two:

| Metric (USD M) | 5 % of worlds | median world | 95 % of worlds |
|---|---:|---:|---:|
| EAL | 14.9 | 21.7 | 33.1 |
| P99 | 79 | 166 | 293 |
| ES99 | 174 | 316 | 542 |

Parameter uncertainty alone spans a factor of about 2 in the EAL and 3–4 in
the tail metrics. The randomness of a year can only be absorbed (resilience,
insurance, contingency). The uncertainty about the parameters can be *reduced*
with data, and the sensitivity analysis shows where to collect it.

![Epistemic band](outputs/figures/lec_epistemic_band.png)

### Which assumptions matter most?

| Rank | Parameter | EAL swing P10→P90 | Spearman ρ (EAL) | First-order index |
|---:|---|---:|---:|---:|
| 1 | `bi_value_fraction` (economic value of a deferred barrel) | 8.3 M | 0.53 | 0.25 |
| 2 | `pipeline_rate_per_km_yr` | 3.2 M | 0.32 | 0.10 |
| 3 | `comp_weibull_scale_years` (compressor life) | 3.2 M | −0.28 | 0.08 |
| 4 | `rel_flange_seal_rate` | 3.5 M | 0.20 | 0.07 |
| 5 | `ptw_isolation_failure_prob` | 4.0 M | 0.19 | 0.06 |
| 6 | `rel_valve_rate` | 2.8 M | 0.25 | 0.07 |
| 7 | `weather_rate` | 3.4 M | 0.20 | 0.05 |

For the tail (ES99) the ranking shifts toward the pipeline failure rate and
repair duration, the large-release fraction and explosion probability.
Protection-system reliability parameters barely move either metric at this
asset's fire frequency (~0.02 fires/yr).

![Tornado](outputs/figures/tornado_eal.png)

### Stress scenarios (conditional)

| Scenario | Annual frequency | Mean loss of the event | Conditional P99 of the year |
|---|---:|---:|---:|
| S1 Single compressor failure | 0.73 /yr | 7.1 M | 193 M |
| S2 Firewater degradation (state) | — | +0.08 M/yr | 186 M |
| S3 Large gas release on PPA | 0.011 /yr | 16 M | 502 M |
| S4 Explosion on PPB escalating to PPA and UCP | 1.8e-5 /yr | 431 M | 1,173 M |
| S5 Severe storm + compressor failure, delayed logistics | 0.009 /yr | 40 M | 336 M |
| S6 Common-cause power failure (5-day blackout) | 0.082 /yr | 10.6 M | 199 M |
| S7 Major fire on PPA at 110 USD/bbl, delayed repair | 0.002 /yr | 430 M | 1,560 M |

S2 is instructive: degrading the firewater system raises the fire-protection
PFD from 0.024 to 0.030 but adds only USD 0.08 M to the expected annual loss. An
expected-value view alone would under-rate firewater integrity.

![Scenarios](outputs/figures/scenarios.png)

### Controls and the USD 5 M budget

| Control | Capex | EAL reduction (±SE) | ES99 reduction | Annualised cost per USD of EAL reduced |
|---|---:|---:|---:|---:|
| Critical spares (repair time) | 2.5 M | 2.29 M (±0.03) | 32 M | 0.21 |
| Gas-detection coverage | 1.5 M | 1.88 M (±0.07) | 41 M | **0.11** |
| Compressor preventive maintenance | 1.2 M | 1.59 M (±0.02) | 2 M | 0.31 |
| Risk-based inspection | 0.8 M | 0.85 M (±0.05) | 16 M | 0.53 |
| Permit-to-work competence | 0.5 M | 0.72 M (±0.05) | 16 M | 0.38 |
| Fourth generator | 3.5 M | 0.28 M (±0.01) | 0.3 M | 1.84 |
| Deluge coverage | 1.8 M | 0.10 M (±0.04) | 9 M | 2.41 |
| Extra diesel firewater pump | 2.2 M | 0.09 M (±0.04) | 8 M | 2.91 |
| Faster emergency response | 0.6 M | 0.01 M (±0.01) | 1 M | 18.2 |

**The optimal portfolio depends on the objective** (MILP, confirmed by exhaustive search over all 114 feasible portfolios):

| Objective | Selected controls | Capex | EAL reduction | ES99 reduction |
|---|---|---:|---:|---:|
| Minimise expected loss | inspection + compressor PM + critical spares + PTW | 5.0 M | **4.83 M/yr** | 64 M |
| Minimise ES99 (tail) | gas detection + critical spares + PTW | 4.5 M | 4.73 M/yr | **96 M** |
| Maximise net benefit (EAL reduction − annualised cost) | gas detection + critical spares + PTW | 4.5 M | 4.73 M/yr | 96 M |

![Mitigations](outputs/figures/mitigations.png)

### Risk appetite (hypothetical limits)

| Limit | Value | Limit | Status |
|---|---:|---:|---|
| EAL | 22.6 M | 25 M | Near threshold (90 %) |
| P95 annual loss | 61.5 M | 70 M | Near threshold (88 %) |
| P95 lost production (full-complex days) | 30.8 d | 25 d | **Outside appetite** (123 %) |
| 1-in-100 single-event loss | 154 M | 250 M | Within (62 %) |
| Firewater PFD (production platform) | 0.0074 | 0.01 | Within (74 %) |
| Fire-protection PFD (production platform) | 0.024 | 0.05 | Within (48 %) |

## 7. Dashboard

`streamlit run dashboard/app.py` opens six views: Executive, Asset, Scenarios,
Monte Carlo, Mitigation, Sensitivity. The sidebar sets the sample size,
dependence model, Bayesian updating, seed and the controls in place, and every
view updates from the library. (Screenshots use the default 20,000 simulated years, so
figures differ slightly from the 100,000-year results above.)

| Executive | Scenarios |
|---|---|
| ![](docs/img/dashboard_executive.png) | ![](docs/img/dashboard_scenarios.png) |
| **Mitigation** | **Sensitivity** |
| ![](docs/img/dashboard_mitigation.png) | ![](docs/img/dashboard_sensitivity.png) |

## 8. Key findings (for this fictional asset and these assumptions)

1. **The economic assumption beats the engineering ones.** What a deferred
   barrel is worth moves the EAL more than any failure rate. Pin it down first.
2. **Mean and tail are different problems.** Compressors, weather and trips
   cost money every year but almost never make a year catastrophic. Releases
   and the export pipeline do.
3. **Redundancy is weaker than it looks** where support systems are shared.
   The firewater PFD doubles once the common power supply is modelled. A fourth
   generator helps little because the common cause stays.
4. **Detection and isolation are the economically decisive barrier.** Better gas
   detection is the most cost-effective control and the best tail-reducer.
   Undetected releases become unisolated ones, which shut the complex down even
   when they do not ignite.
5. **Firewater, deluge and emergency-response controls look poor on expected
   value** but reduce fire-protection unavailability and the catastrophic tail.
   Choosing them is a safety and appetite decision, not a cost-benefit one.
6. **Correlation between risks thickens the upper-middle of the distribution**
   (P90–P95 +5–9 %). It does not thicken the extreme tail, which is set by single events.
7. **Parameter uncertainty is as large as the risk signal** (EAL 15–33 M across
   plausible worlds), so data collection on the top-ranked parameters has real value.

## 9. Limitations

The main ones: economic risk only (no individual or societal risk); no
consequence physics (fire/explosion outcomes are classes with assumed damage
and downtime); simplified PFD formulas; β-factor CCF; Gaussian copula with no
tail dependence; gross pre-insurance losses; blowouts and subsea equipment not
modelled; synthetic data generated by the model itself. Full list:
[docs/limitations.md](docs/limitations.md).

## 10. How to run

```bash
git clone <this repo> && cd offshore-risk-engine
python -m venv .venv && source .venv/bin/activate      # Python 3.10+
pip install -r requirements.txt && pip install -e .

python scripts/generate_synthetic_data.py   # data/synthetic/*.csv (fixed seed)
python scripts/build_data_dictionary.py     # docs/data_dictionary.md
python scripts/run_analysis.py              # outputs/results + outputs/figures (~2 min)
python scripts/run_validation.py            # docs/validation_results.md (~2 min)
python scripts/build_notebooks.py           # executes notebooks/01–05
streamlit run dashboard/app.py
```

or `make all`. Minimal use from Python:

```python
from offshore_risk import load_config
from offshore_risk.simulation import simulate
from offshore_risk.financial.metrics import risk_metrics

cfg = load_config()
res = simulate(cfg, n_years=50_000, dependence="correlated", mitigations=["gas_detection_upgrade"])
print(risk_metrics(res.total))
```

To change an assumption, edit `config/*.yaml` and re-run. Nothing is hard-coded
in the model code.

## 11. Testing

```bash
pytest            # 106 tests, ~15 seconds
```

The tests check closed forms, brute-force enumeration (fault trees, optimiser),
numerical integration (Bayesian posteriors, PFDs), independent Monte Carlo and
discrete-event simulation (standby, 1oo2, repairable 2oo3), and engine
invariants (reproducibility, accounting identities, common-random-number
monotonicity). Model-level validation (convergence at 10k/50k/100k years, event
tree inside the engine, dependence models, MILP vs exhaustive search, Bayesian
coverage, edge cases) is in [docs/validation.md](docs/validation.md). At 100k
years the EAL and P95 are stable to within 0.5 % across seeds, P99 to within about 4 %.

## 12. Future improvements

* Consequence modelling from physics (jet-fire length, explosion overpressure) instead of outcome classes.
* Individual and societal risk (PLL, F-N curves), with temporary-refuge impairment.
* Alpha-factor CCF; partial-stroke and imperfect proof testing; SIL verification mode.
* Hierarchical Bayesian models pooling across similar equipment; test-coverage correction.
* t-copula or event-level common causes (storm → collision) for tail dependence.
* Insurance programme (deductibles, limits, BI waiting periods) to move from gross to net loss.
* Multi-year simulation with decline curves, deferred-production recovery and maintenance schedules.
* Full Saltelli/Sobol' design for interaction effects.

---

*Licence: MIT. Fictional asset; synthetic data; for research and education.*
