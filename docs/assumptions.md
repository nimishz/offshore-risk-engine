# Assumptions

The complete parameter list, with units, distributions and a one-line rationale for
each value, is in [`data_dictionary.md`](data_dictionary.md) (generated from
`config/risk_parameters.yaml`). This document explains the assumptions that shape
the model's *structure* and the groups of parameters that matter most.

> **Nothing in this repository describes a real facility or a real operator's data.**
> No value is taken from confidential sources. Where a value is described as
> "order of magnitude", it has been placed within the general range reported by
> the *type* of public source named (for example, regulator hydrocarbon-release
> statistics, reliability handbooks such as OREDA, pipeline loss-of-containment
> compilations). The specific numbers are the author's assumptions and must not
> be cited as coming from those sources.

## A. Asset and dependency assumptions

| # | Assumption | Why | Effect if wrong |
|---|---|---|---|
| A1 | Six bridge-linked platforms; only PPA (22 kbbl/d) and PPB (18 kbbl/d) are credited with production | Keeps the production accounting simple; wells on WHP are credited to the processing platforms | Rescales business interruption linearly |
| A2 | All production depends 100 % on main power from UCP | Typical of a complex with centralised power generation | UCP is the single most critical platform (a UCP outage stops 100 % of output) |
| A3 | PPB exports through PPA's metering/pipeline launcher with 50 % bypass capability | Creates an explicit asymmetric dependency ("Platform A failure affects downstream processing") | PPA outages cost 78 % of output, PPB outages 45 % |
| A4 | Loss of LQ removes 90 % of production capability (de-manning) | Production cannot continue for long without accommodation; a flotel restores part | LQ is rarely lost, so small effect on EAL |
| A5 | Water injection loss reduces production by 15 % (as a constant, not a decline curve) | Simplification of reservoir pressure support | Small |
| A6 | Escalation only to first-order bridge neighbours | Second-order escalation (A → B → C) is rare and would need a time-dependent fire model | Understates the most extreme tail |
| A7 | Overlapping outages in one year are added, not merged | Occurrence-basis accounting | Slight overstatement in rare multi-event years |

## B. Frequency assumptions

* **Hydrocarbon releases** are built from causes (valve leaks, flanges/seals,
  corrosion, maintenance/permit-to-work failures, overpressure), each an
  epistemic parameter, summed (OR gate on rates) and scaled by a relative
  leak-source count per platform. Total ≈ 1.2 releases per year across the
  complex (all sizes), 5 % large, 25 % medium. The cause split is an assumption;
  only the total order of magnitude is anchored to the general range of public
  release statistics.
* **Compressor failures** use a Weibull power-law NHPP (shape 1.2–2.2, scale
  2.5–5 years, two trains, two years since overhaul) giving ≈ 0.73 train
  failures per year.
* **Total power loss** is derived from generator reliability (2oo3, ~1.3 trips
  per machine-year, 8–72 h restore) plus a β-factor common cause; the common
  cause contributes roughly 70 % of the total (≈ 0.09 /yr at median parameters, ≈ 0.13 /yr averaged over parameter uncertainty).
* **Spurious trips** are frequent (prior median 3 /yr, posterior ≈ 4.7 /yr after
  the synthetic records) and short (median 8 h).
* **External events** (weather, collision) have rates chosen for a fictional
  basin with a cyclone season. They are illustrative only.

## C. Protection-layer assumptions

* Detection is limited by **geometric coverage** (60 % small, 85 % medium, 95 %
  large releases), not by detector hardware (voted 2oo3 PFD ≈ 1e-3). This is a
  deliberate modelling choice that reflects the usual finding of detector
  mapping studies; the numbers are assumptions.
* All safety functions are proof-tested annually with perfect tests; PFDavg uses
  simplified IEC 61508-6-type formulas without repair time or diagnostic
  coverage. The 1oo2 approximation is slightly conservative (validated against
  exact integration in the tests).
* **Power loss given a fire** depends on location (60 % on UCP, 10 % on process
  platforms, 5 % elsewhere) and feeds the firewater fault tree.
* Electric firewater pump fail-to-start and diesel pump fail-to-start use
  Bayesian posteriors from the synthetic weekly start tests (see §E).

## D. Consequence and financial assumptions

* Downtime medians per event-tree outcome (e.g. major fire 45–180 days) are
  epistemic; each event's actual downtime is lognormal around that median
  (aleatory, log-sd 0.6) and is multiplied by the year's logistics factor.
* Damage is a Beta-distributed fraction of the platform replacement value
  (fictional values, USD 250–900 M per platform).
* **Business interruption** = lost barrels × annual oil price × value fraction.
  The oil price is lognormal (median 75 USD/bbl, σ = 0.25): an assumption for
  illustration, not a forecast. The value fraction (0.3–0.8) reflects that most
  deferred production is recovered later. **This is the single most influential
  assumption in the model** (see sensitivity results); an operator would need
  its own economics to set it.
* Losses are **gross** (before insurance) and **pre-tax**.
* Environmental consequences are recorded as a physical proxy (bbl of liquid
  released). A monetary value is produced only under an explicit scenario
  assumption and is excluded from totals by default. No regulatory penalties are
  modelled.

## E. Bayesian updating assumptions

* Priors are "generic" assumptions (e.g. electric pump fail-to-start mean 1 %,
  equivalent to 100 pseudo-demands; spurious trips lognormal median 3 /yr,
  EF 2, moment-matched to a Gamma prior).
* The two electric pumps are pooled (exchangeable: same design, room and crew).
  Pooling across dissimilar items would need a hierarchical model.
* The synthetic records are produced by the model with a synthetic truth that
  differs from the prior. The update therefore demonstrates the mechanics; it
  does not validate the prior against reality.
* The total release count is updated for display but not fed back, because a
  total cannot identify which cause rate was wrong.

## F. Dependence assumptions

* Three year-level drivers: weather severity (σ = 0.5), logistics constraint
  (σ = 0.25) and asset-integrity condition (σ = 0.3), lognormal with mean 1.
* Model B correlation: weather–logistics 0.6, weather–integrity 0.3,
  logistics–integrity 0.4 (assumption).
* A Gaussian copula has no tail dependence. The validation run includes a much
  stronger correlation stress test.

## G. Mitigation assumptions

Costs, lives (5–20 years) and effectiveness ranges are illustrative. Annualised
cost = capex × capital recovery factor (8 %, asset life) + annual opex. The
budget constraint (USD 5 M) applies to capex. Effectiveness is uncertain and
sampled per simulated world from its own random stream.

## H. Risk appetite

All limits in `config/risk_appetite.yaml` are **hypothetical management
thresholds** for the fictional operator. They are not industry standards,
regulatory criteria or recommendations.
