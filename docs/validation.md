# Validation

Validation here means: *does the code compute what the methodology says it
computes, and are the Monte Carlo estimates stable?* It cannot show that the
parameters are right for any real asset (see [limitations](limitations.md)).

Two layers:

1. **Unit tests** (`pytest`, 106 tests, ~15 seconds) — analytical checks of each
   module against closed forms, brute force or independent simulation.
2. **Model-level checks** (`python scripts/run_validation.py`) — convergence,
   event-tree consistency inside the full engine, dependence behaviour,
   optimiser vs exhaustive search, Bayesian interval coverage, edge cases.
   Output: [`validation_results.md`](validation_results.md).

## Unit tests

| Module | What is checked | Reference used |
|---|---|---|
| distributions | ppf/cdf round trip; sample mean vs analytic mean; error-factor definition; LHS stratification; mean-1 lognormal multiplier | closed form |
| reliability | exponential & Weibull identities; Weibull mean by numerical integration; NHPP increment; series/parallel/k-of-n; standby reliability vs Monte Carlo; 1oo2 PFD vs exact integral and vs Monte Carlo; 2oo3 repairable frequency vs discrete-event simulation; CCF dominance | integration, simulation |
| fault tree | exact (Shannon) = brute-force enumeration on random trees and on the fire-protection tree; naive gate multiplication shown to under-estimate with shared events; minimal cut sets (incl. k-of-n, non-minimal removal); rare-event and MCUB bounds; importance measures | enumeration of 2^n states |
| event tree | outcomes sum to 1; hand-computed path probabilities; sampled outcome frequencies vs analytic; severity-ordered sampling is monotone; degenerate branches | hand calculation |
| Bayesian | conjugate posterior = grid posterior (Beta-Binomial and Gamma-Poisson); moment matching; more data narrows intervals; predictive overdispersion; calibration pipeline leaves the base config untouched | numerical integration |
| asset | capacity bounds and monotonicity over all 64 down-sets; hand-computed dependency cases; topological order; piecewise integration of lost production | hand calculation |
| engine | reproducibility by seed; accounting identities (sources = components); event counts = rates; **CRN monotonicity** (a frequency-only control never increases loss in any year); zero-rate edge case; forced event adds exactly its own loss; chunking; event log reconciles to annual totals | internal consistency |
| metrics | known samples; ES ≥ VaR; exceedance; Euler allocation sums to ES; bootstrap; CRF; appetite classification | closed form |
| optimisation | MILP = brute force on random quadratic knapsacks; budget respected; zero budget; additive knapsack; end-to-end optimiser vs exhaustive | enumeration |

## Model-level results (full run)

From [`validation_results.md`](validation_results.md):

**Convergence** — coefficient of variation across five seeds:

| simulated years | EAL | P95 | P99 | ES99 |
|---|---|---|---|---|
| 10,000 | 1.7 % | 3.2 % | 9.8 % | 5.6 % |
| 50,000 | 1.3 % | 0.8 % | 5.6 % | 4.6 % |
| 100,000 | 0.5 % | 0.4 % | 3.6 % | 2.7 % |

EAL and P95 are stable to well under 1 % at 100k years. P99 and ES99 still
move by a few percent between seeds: they are driven by rare major-accident
years, so their Monte Carlo error falls slowly. Headline tail metrics are
therefore reported with bootstrap intervals, and decisions that hinge on tail
differences use common random numbers.

**Event tree inside the engine** — 185k simulated releases at median
parameters; every outcome share is within |z| < 1 of the analytic event-tree
probability.

**Dependence** — Model A and B have the same event frequencies (ratio 0.998);
Model B raises P90 by 5 %, P95 by 9 %, P99 by 3.5 %, and leaves ES99 unchanged.
A deliberately extreme correlation stress (0.7–0.9) raises P95 by 13 % and ES99
by 3 %.

**Optimisation** — for both EAL and ES99 objectives, the MILP selection equals
the best of all 114 feasible portfolios found by simulating each one.

**Bayesian coverage** — with truth drawn from the prior, nominal 90 % posterior
intervals covered the truth in 90.0 % of 2,000 synthetic data sets.

**Edge cases** — zero value fraction gives zero business interruption; zero
ignition probability gives no fires; a failed firewater header gives PFD = 1;
a one-year run works.

## Distribution assumptions

Distribution families are chosen by the nature of the quantity, not by fit to
data (there is none): lognormal for rates spanning orders of magnitude
(expressed as median and error factor); Beta for probabilities; triangular for
bounded judgements with a most-likely value; lognormal for durations and costs
(positive, right-skewed); mean-one lognormal for year-level multipliers. The
tornado analysis shows how much each assumed range moves the result.
