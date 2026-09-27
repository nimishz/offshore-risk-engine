# Methodology

This document defines the model *before* its implementation details: the system
architecture, the risk taxonomy and how each item is represented, the mathematical
assumptions, the data schema, the dependencies between models, the simulation
workflow and the validation strategy. Parameter values live in `config/` and are
listed in [`data_dictionary.md`](data_dictionary.md); their justification is in
[`assumptions.md`](assumptions.md).

> The asset, its parameters and all synthetic data are fictional. Nothing here
> describes or predicts a real facility.

---

## 1. Question the model is built to answer

For a fictional six-platform, bridge-linked offshore production complex:

1. What is the distribution of annual economic loss from operational hazards
   (not just its mean)?
2. How much of that loss comes from frequent, low-severity production
   interruptions versus rare major-accident events, and how does the answer
   change between the mean and the tail?
3. How reliable are the protection layers (detection, isolation, firewater,
   deluge), and where does their architecture create hidden common-cause
   dependence?
4. Which assumptions drive the answer, and how wide is the uncertainty that
   comes from not knowing parameter values (as opposed to year-to-year
   randomness)?
5. Given a fixed mitigation budget, which combination of controls buys the most
   risk reduction, and does the answer depend on the risk measure chosen?

Everything else in the repository exists to answer one of these questions.

---

## 2. System architecture

```text
                           config/*.yaml
                                │
                     ┌──────────▼──────────┐
                     │  ParameterRegistry  │  epistemic parameters: value, unit,
                     │  (config.py)        │  distribution, basis, rationale
                     └──────────┬──────────┘
         θ ~ epistemic draw     │
     ┌──────────────┬───────────┼──────────────┬─────────────────┐
     ▼              ▼           ▼              ▼                 ▼
 reliability/   fault_tree/  bayesian/     asset/            event_tree/
 PFDavg, k-of-n, exact & cut- conjugate    dependency graph  generic tree +
 standby, CCF,  set evaluation posteriors  → capacity table  HC-release tree
 Weibull NHPP   (Shannon exp.) (optional   (64 down-states)
     │              │         θ override)       │                 │
     └──────┬───────┘                           │                 │
            ▼                                   │                 │
   simulation/derived.py  ── rates λ(θ), protection-layer PFDs(θ), branch probabilities
            │                                   │                 │
            ▼                                   ▼                 ▼
   simulation/engine.py   year-level drivers (Model A indep. / Model B Gaussian copula)
                          → thinned Poisson event counts per source
                          → event-tree outcome per release, escalation, downtime
                          → financial/loss_model.py (direct vs indirect loss)
                          → per-year loss by source × component
            │
            ├── financial/metrics.py      EAL, quantiles, VaR, ES, exceedance, LEC, tail allocation
            ├── simulation/nested.py      two-loop epistemic/aleatory separation
            ├── scenarios/                conditional stress scenarios
            ├── appetite/                 hypothetical limits → RAG status
            ├── mitigation/               controls as parameter transforms, CRN comparison
            ├── optimization/             budget-constrained MILP with pairwise interactions
            ├── sensitivity/              OAT tornado, rank correlation, binned first-order indices
            └── visualization/            matplotlib only; no model logic
```

Design rules:

* **Business logic never lives in the dashboard, notebooks or plotting code.**
  They call library functions.
* **Every stochastic input is sampled by inverse transform (`ppf`) from
  uniforms.** This gives one mechanism for Latin-hypercube sampling, Gaussian
  copulas, one-at-a-time sensitivity (evaluate at P10/P90) and common random
  numbers.
* **Separate random streams per subsystem** (`numpy.random.SeedSequence`
  spawned by name), so that changing a mitigation does not shift the random
  numbers used anywhere else.

---

## 3. The fictional asset

Six bridge-linked platforms. Layout (bridges) and functional dependencies are
declared in `config/asset_config.yaml`.

```text
  WHP ══ PPA ══ PPB ══ UCP ══ LQ
                        ║
                       SWI
```

| Code | Platform | Role in the model |
|---|---|---|
| WHP | Wellhead Platform | Well fluids to PPA/PPB (shares of feed); hydrocarbon inventory |
| PPA | Production Platform A | Separation, 22 kbbl/d nominal; export metering for PPB |
| PPB | Production Platform B | Separation, 18 kbbl/d nominal |
| UCP | Utility/Compression Platform | Main power generation (3×50 % gas turbines), gas compression, electric firewater pumps |
| LQ  | Living Quarters | Accommodation, temporary refuge; manning of the complex |
| SWI | Support/Water Injection | Water injection (reservoir pressure support), diesel standby firewater pump |

### 3.1 Functional dependency

A dependency edge `A → B (impact f)` means: *when B is unavailable, A loses a
fraction f of its capability*. Availability is propagated in topological order:

$$a_p = u_p \prod_{q \in \text{deps}(p)} \bigl(1 - f_{pq}\,(1 - a_q)\bigr)$$

where $u_p \in \{0,1\}$ is whether platform $p$ itself is down. Production
capacity is $C(S) = \sum_{p\in\text{producers}} Q_p\, a_p$ for a down-set $S$.
Because there are only six platforms, $C$ is pre-computed for all $2^6 = 64$
down-sets and looked up by bit-mask during simulation.

An outage in which platforms go down for different durations $d_p$ (all
starting at $t=0$) loses

$$\text{bbl} = \sum_k (t_k - t_{k-1})\,\bigl(Q - C(S_k)\bigr),$$

with $t_k$ the sorted durations and $S_k = \{p : d_p \ge t_k\}$.

### 3.2 Physical escalation

Separate from functional dependency: a major fire or explosion on platform $p$
can escalate across a bridge to each adjacent platform with a probability that
depends on the outcome class and on whether the emergency response was delayed.
Escalation is limited to first-order neighbours (documented limitation).

---

## 4. Risk taxonomy and representation

Not every taxonomy item deserves its own stochastic process. Each is mapped to
the model element that represents it, or explicitly marked as not modelled.

| Category | Item | Representation in the model |
|---|---|---|
| Process safety | Hydrocarbon release / loss of containment | Poisson initiating event per platform; frequency = sum of cause rates (§5.1) |
| | Gas leak vs liquid release | Event-tree phase split (gas/liquid) — changes explosion probability and spill proxy |
| | Overpressure | Release cause: demand rate × PFD of pressure protection (AND of rate and probability) |
| | Ignition | Event-tree branch; size-dependent, reduced if isolated |
| | Explosion | Event-tree branch conditional on ignition; mostly gas, medium/large |
| | Jet fire / pool fire | Outcome classes *minor/major fire* (jet vs pool is not separated in consequence — limitation) |
| Fire protection | Firewater pump failure | Fault tree: 2 electric pumps (shared power, CCF) + diesel standby |
| | Firewater header failure | Fault-tree basic event in series with the pumps |
| | Deluge failure | Fault-tree basic event (valve PFDavg) |
| | Fire detection failure | Fault tree: 2ooN flame detection with CCF; manual activation as recovery |
| | Gas detection failure | Coverage (geometric) × voting hardware PFD |
| | ESD failure | 1oo2 isolation valves with β-factor CCF + logic solver |
| | Foam system failure | **Not modelled separately** — foam matters for pool fires in bunded areas; folded into the fire-protection failure probability |
| Mechanical | Compressor failure | Weibull power-law NHPP (non-constant hazard), partial production loss |
| | Generator failure | 2oo3 repairable redundancy + β-factor CCF → frequency of total power loss |
| | Pump failure | Firewater pumps in the fault tree. Export pumps **not modelled** (2×100 % spared; assumed negligible) |
| | Valve failure | Release cause (valve leak) and ESD valve PFD |
| | Pipeline failure | Export pipeline leak: Poisson event, long outage, spill proxy |
| | Instrumentation failure | Spurious trips: frequent, short platform outages |
| Human / operational | Maintenance error, permit-to-work failure | Release cause: breaking-containment jobs × P(isolation/PTW failure) × P(release \| failure) |
| | Procedural deviation | Folded into the spurious-trip rate and PTW failure probability (not separately identifiable) |
| | Inspection failure | Corrosion-driven release rate and pipeline rate depend on an inspection-effectiveness factor |
| | Delayed emergency response | Per-event Bernoulli branch that multiplies escalation probability |
| External | Extreme weather | Poisson event: precautionary shut-in and de-manning; small damage probability |
| | Marine collision | Poisson event: impact damage and outage; may rupture a riser → spawns a large release on WHP |
| | Supply disruption, helicopter/transport disruption | Year-level *logistics factor* multiplying repair durations (correlated with weather in Model B) |

---

## 5. Mathematical framework

### 5.1 Frequencies

Release frequency on platform $p$ (per year):

$$\lambda_p = k_p\Bigl[\lambda_\text{valve} + \lambda_\text{flange} + \lambda_\text{corr}\,\phi_\text{insp}
+ n_\text{jobs}\,q_\text{PTW}\,p_\text{rel|PTW} + \lambda_\text{demand}^{op}\,\text{PFD}_\text{PSV}\Bigr]$$

$k_p$ is a relative equipment-count factor. The bracket is an OR gate on rates
(rare-event sum) containing two AND gates of a rate with a probability.

Compressor failures follow a power-law non-homogeneous Poisson process (minimal
repair). With Weibull scale $\eta$, shape $\beta$ and age since overhaul $a$, the
expected number of failures of one compressor in the coming year is

$$\Lambda = (\tfrac{a+1}{\eta})^\beta - (\tfrac{a}{\eta})^\beta .$$

$\beta > 1$ means the rate increases with age, so an overhaul (resetting $a$)
changes risk — a constant-rate model cannot express that.

Total power loss for $n$ generators of which $k$ are needed, each failing at
rate $\lambda$ with mean repair time $\tau$ (independent repair, $\lambda\tau \ll 1$),
plus a β-factor common cause:

$$f_\text{sys} \approx \binom{n}{m} m\,\lambda^m \tau^{m-1} + \beta\lambda, \qquad m = n-k+1 .$$

### 5.2 Protection layers (probability of failure on demand)

Low-demand safety functions, periodically proof-tested with interval $T$,
dangerous-undetected rate $\lambda_{DU}$ (simplified IEC 61508-6 forms,
neglecting repair time and diagnostic coverage):

* 1oo1: $\text{PFD} \approx \lambda_{DU} T/2$
* $k$oo$n$ (independent): $\text{PFD} \approx \binom{n}{m}\frac{(\lambda_{DU}T)^m}{m+1}$, $m=n-k+1$
* with β-factor CCF: $\text{PFD} \approx \text{PFD}_{koon}((1-\beta)\lambda_{DU}) + \beta\lambda_{DU}T/2$

The fire-protection function is a fault tree (Figure in notebook 02):

```text
FP_FAIL = OR( FIREWATER_FAIL, DELUGE_VALVE_FAIL, AND(FIRE_DET_FAIL, MANUAL_FAIL) )
FIREWATER_FAIL = OR( HEADER_FAIL, AND(ELECTRIC_PUMPS_FAIL, DIESEL_PUMP_FAIL) )
ELECTRIC_PUMPS_FAIL = AND( PUMP_A_UNAVAILABLE, PUMP_B_UNAVAILABLE )
PUMP_A_UNAVAILABLE = OR( PUMP_A_FTS, PUMP_A_MAINT, POWER_LOSS, CCF_ELEC_PUMPS )
PUMP_B_UNAVAILABLE = OR( PUMP_B_FTS, PUMP_B_MAINT, POWER_LOSS, CCF_ELEC_PUMPS )
```

`POWER_LOSS` and `CCF_ELEC_PUMPS` appear under both pumps. Multiplying gate
probabilities as if inputs were independent gives roughly $(q + p)^2$ where the
correct answer is roughly $q^2 + p$. The fault-tree module evaluates exactly by
**Shannon decomposition** on repeated basic events:

$$P(\text{top}) = p_x\,P(\text{top}\mid x{=}1) + (1-p_x)\,P(\text{top}\mid x{=}0),$$

applied recursively to every repeated event; after conditioning, the remaining
events appear once and gate arithmetic is exact. With $r$ repeated events the
cost is $2^r$ gate evaluations, fine for trees of this size. Minimal cut sets
(MOCUS) are also produced, with the rare-event approximation, the min-cut upper
bound, and Birnbaum / Fussell–Vesely importance.

`POWER_LOSS` is conditional on where the fire is: a fire on UCP is much more
likely to take out main power than a fire elsewhere, so the firewater PFD is
evaluated per platform.

### 5.3 Event tree (hydrocarbon release)

For each release (size $s \in$ {small, medium, large}, phase ∈ {gas, liquid}):

```text
Release ─┬─ Detected (coverage_s × (1−PFD_GD)) ─┬─ Isolated (1−PFD_ESD) ─┬─ no ignition ................ controlled_release
         │                                      │                       └─ ignition (p_ign,s · r_iso) ─┬─ no explosion ─┬─ FP ok → minor_fire
         │                                      │                                                      │                └─ FP fail → major_fire
         │                                      │                                                      └─ explosion ────┬─ FP ok → explosion
         │                                      │                                                                       └─ FP fail → catastrophic
         │                                      └─ Not isolated ─┐
         └─ Not detected ───────────────────────────────────────┴─ no ignition ................ uncontrolled_release
                                                                  └─ ignition (p_ign,s) ─┬─ no explosion ─┬─ FP ok → major_fire
                                                                                         │                └─ FP fail → catastrophic
                                                                                         └─ explosion ────┬─ FP ok → explosion
                                                                                                          └─ FP fail → catastrophic
```

Outcome probabilities are products along paths; they sum to one by
construction (tested). The implementation is a generic tree whose branch
probabilities may be numpy arrays, so the same tree is evaluated for every
simulated event at once. Outcomes are ordered by severity and sampled by
inverse CDF; when a mitigation moves probability towards less severe outcomes,
the same random number maps to the same or a less severe outcome, which keeps
mitigation comparisons low-noise.

### 5.4 Year-level drivers and dependence (Models A and B)

Three latent year-level drivers capture conditions that affect many risks at
once: weather severity $W$, logistics constraint $L$, and asset-integrity /
maintenance-backlog condition $I$. Each has a lognormal marginal with mean 1:

$$M_g = \exp(\sigma_g Z_g - \sigma_g^2/2).$$

* **Model A (independent):** $Z \sim N(0, I)$.
* **Model B (correlated):** $Z \sim N(0, \Sigma)$ — a Gaussian copula with the
  same marginals, so every event frequency and every duration multiplier has the
  same distribution in both models; only the joint behaviour changes. (Expected
  loss still rises slightly under B, because loss scales with frequency ×
  duration and $E[M_I M_L] = e^{\rho\sigma_I\sigma_L} > 1$ when the integrity and
  logistics drivers are positively correlated.)

Event counts are mixed Poisson: $N_{i,y} \sim \text{Poisson}(\lambda_i(\theta)\,M_{g(i),y})$.
$W$ scales weather frequency, $I$ scales release and mechanical frequencies,
and $L$ scales every repair/outage duration. The comparison isolates the effect
of dependence on the aggregate tail. (A Gaussian copula has no tail
dependence; a t-copula would fatten the tail further. Noted as a limitation.)

### 5.5 Aleatory and epistemic uncertainty

| Type | Meaning | Examples | Where sampled |
|---|---|---|---|
| Epistemic | We do not know the true value; more data would narrow it | release cause rates, ignition probabilities, λ_DU, damage-fraction means | once per simulated "world" θ |
| Aleatory | Randomness that would remain even with perfect knowledge | number of events in a year, which branch an event takes, the actual downtime of one repair, year-level drivers, annual oil price | per year / per event |

Two modes:

* **Pooled (single loop):** a fresh θ is drawn for every simulated year. The
  resulting distribution is the *predictive* annual loss distribution, with
  parameter uncertainty integrated out. Used for the headline metrics.
* **Nested (two loop):** outer loop draws $N_\text{out}$ worlds θ; the inner
  loop simulates $N_\text{in}$ years for each. Each world gives its own
  EAL, P95, P99 and loss exceedance curve. The spread across worlds is the
  epistemic uncertainty *about* those metrics.

Why it matters for management: aleatory risk is managed by resilience and risk
transfer (it cannot be reduced by study); epistemic uncertainty can be reduced
by data collection, testing and inspection, which are decisions with a value
of their own.

### 5.6 Financial translation

Per event, with oil price $P$ (USD/bbl, annual average) and value fraction $v$:

| Component | Type | Formula |
|---|---|---|
| Property damage | Direct | damage fraction × platform replacement value |
| Equipment repair | Direct | repair cost (mechanical events, pipeline) |
| Emergency response | Direct | outcome-dependent cost; + evacuation cost when LQ is involved |
| Logistics | Direct | day-rate × repair days × logistics factor |
| Restart | Direct | fixed cost per platform shutdown |
| Business interruption | Indirect | lost bbl × $P$ × $v$ |
| Environmental proxy | Not monetised | bbl of liquid released (index only) |
| Environmental cost | **Scenario assumption only** | proxy × assumed USD/bbl; excluded from totals unless explicitly enabled |

$v$ (the economic loss per deferred barrel as a fraction of price) is below 1
because much deferred production is produced later; it bundles time value,
permanent loss and avoided variable cost. It is one of the most uncertain
parameters and is treated as such.

Losses are recorded on an **occurrence basis**: all consequences of an event
are attributed to the year it occurs, even if the outage runs into the
following year.

### 5.7 Risk metrics

For the simulated annual loss sample $L_1,\dots,L_N$:

| Metric | Definition | Use |
|---|---|---|
| EAL | $\bar L$ | Budgeting; cost-benefit of controls (risk-neutral view) |
| Median | $Q(0.5)$ | A "typical" year; usually far below EAL for skewed losses |
| P75/P90/P95/P99 | $Q(\alpha)$ | Planning levels; P95/P99 are common appetite anchors |
| VaR$_\alpha$ | $Q(\alpha)$ (same as the percentile; the term used in ERM) | Capital/contingency sizing |
| ES$_\alpha$ (TVaR) | $E[L \mid L \ge \text{VaR}_\alpha]$ | Tail severity; coherent, sub-additive |
| $P(L > x)$ | exceedance probability | Appetite statements ("< 5 % chance of losing > X") |
| Max | $\max L_i$ | Only an indicator of sample size; not a risk measure |
| LEC | $x \mapsto P(L > x)$ | Communicates the whole distribution |

Tail allocation by source uses the Euler/co-TVaR decomposition
$\text{ES}_\alpha = \sum_s E[L_s \mid L \ge \text{VaR}_\alpha]$, which sums exactly to the
total and shows which sources drive the tail as opposed to the mean.

Standard errors: $\text{SE}(\bar L) = s/\sqrt N$; percentiles and ES are given
bootstrap confidence intervals.

### 5.8 Mitigation and optimisation

A mitigation is a transformation of θ (e.g. halve a cause rate, add a pump to
the fault tree, reduce repair durations), with its own uncertain effectiveness
(an epistemic parameter). Residual risk is computed by re-running the full
simulation with **common random numbers**; risk reduction is
$\Delta = R_\text{baseline} - R_\text{residual}$ and cost-effectiveness is
annualised cost / $\Delta$ (capital recovery factor at an assumed discount rate
and life).

Portfolio selection with budget $B$, costs $c_i$, binary decisions $x_i$:

$$\max_{x, y}\ \sum_i a_i x_i + \sum_{i<j} b_{ij}\, y_{ij}
\quad\text{s.t.}\quad \sum_i c_i x_i \le B,\ \ y_{ij} \le x_i,\ y_{ij} \le x_j,\ y_{ij} \ge x_i + x_j - 1,$$

where $a_i$ is the simulated stand-alone reduction and
$b_{ij} = \Delta_{ij} - a_i - a_j$ is the simulated pairwise interaction
(negative when two controls protect against the same thing, e.g. an extra
firewater pump and extra deluge coverage). Higher-order interactions are
ignored by the MILP, so the selected portfolio is re-simulated and compared with
exhaustive enumeration of all feasible portfolios (possible here because there
are only nine candidate controls). Solved with `scipy.optimize.milp` (HiGHS).

### 5.9 Sensitivity analysis

1. **One-at-a-time / tornado:** each epistemic parameter moved to its P10 and
   P90 with the rest at their medians; EAL re-simulated with common random
   numbers.
2. **Rank correlation:** Spearman correlation between each parameter and the
   per-world EAL / P99 from the nested run.
3. **First-order variance-based index (given-data estimator):**
   $S_i \approx \operatorname{Var}(E[Y\mid X_i]) / \operatorname{Var}(Y)$, estimated by binning the
   nested-run sample on quantiles of $X_i$. This is a screening estimate, not a
   full Saltelli/Sobol' design; inner-loop Monte Carlo noise inflates
   $\operatorname{Var}(Y)$ slightly and biases indices low.

---

## 6. Data schema

### 6.1 Parameter record (`config/risk_parameters.yaml`)

```yaml
<name>:
  definition: text
  unit: text
  category: taxonomy group
  distribution: {type: lognormal|triangular|uniform|beta|pert|fixed|normal, ...}
  basis: assumption | derived | order-of-magnitude (public literature type named)
  rationale: why this value / range
  uncertainty: epistemic | aleatory
  notes: optional
```

### 6.2 Synthetic data (`data/synthetic/`)

| File | Grain | Columns |
|---|---|---|
| `incident_log.csv` | one row per synthetic event over the observation window | `event_id, date, year, platform, category, subcategory, outcome, downtime_h, direct_cost_usd` |
| `proof_tests.csv` | one row per protective-equipment test | `test_id, date, equipment_tag, equipment_type, result` |
| `equipment_register.csv` | one row per modelled equipment item | `tag, platform, type, description, installed_year` |
| `exposure.csv` | one row per year | `year, platform_years, compressor_years, pipeline_km_years` |

The generator uses a documented "synthetic truth" that deliberately differs
from the prior means, so that Bayesian updating has something to find.

### 6.3 Simulation output (`SimulationResult`)

Per simulated year: loss by source (`n_years × n_sources`), loss by component,
lost production (bbl and equivalent full-complex days), maximum single-event
loss, counts of fires/explosions/evacuations, and the environmental proxy.

---

## 7. Model dependencies

```text
distributions ─► config (ParameterRegistry)
reliability ─► fault_tree (basic-event probabilities)
fault_tree + reliability ─► simulation.derived (PFDs, λ)
bayesian ─► config (optional posterior overrides for selected parameters)
asset (capacity table) ─► simulation.engine
event_tree ─► simulation.engine
financial.loss_model ─► simulation.engine
simulation.engine ─► financial.metrics, nested, scenarios, mitigation, sensitivity
mitigation ─► optimization
appetite ◄─ metrics + derived PFDs
```

No cycles; visualization depends on everything and nothing depends on it.

---

## 8. Simulation workflow

1. Load YAML configuration → `ParameterRegistry`; optionally replace selected
   priors with Bayesian posteriors fitted to the synthetic records.
2. Draw uniforms and transform to epistemic parameters θ (one row per year in
   pooled mode; one row per world in nested mode).
3. Apply mitigation transforms to θ (effectiveness sampled from its own stream).
4. Derive rates and protection-layer PFDs from θ via the reliability and
   fault-tree modules.
5. Draw year-level drivers (Model A or B) and the annual oil price.
6. For each event source draw *candidate* counts at the baseline rate and thin
   them to the mitigated rate (keeps common random numbers aligned).
7. For releases: size, phase, event-tree outcome, delayed-response flag,
   escalation; for all events: platform downtime vectors and direct impacts.
8. Convert to barrels lost via the capacity table, then to money.
9. Aggregate per year by source and component (`numpy.bincount`).
10. Compute metrics, appetite status, contributions, and plots.

Years are processed in fixed-size chunks with chunk-specific seeds, so memory
stays bounded and results are reproducible for a given seed.

---

## 9. Validation strategy

| What | How | Where |
|---|---|---|
| Reliability equations | closed forms vs numerical integration / Monte Carlo; limiting cases (λ=0, n=1) | `tests/test_reliability.py` |
| Fault tree | exact result vs brute-force enumeration of all $2^n$ states; demonstrate naive-gate error with shared events; rare-event ≥ exact ≥ ...; MCUB bounds | `tests/test_fault_tree.py` |
| Event tree | outcome probabilities sum to 1; analytic path probabilities vs sampled frequencies | `tests/test_event_tree.py` |
| Bayesian | conjugate posterior vs grid-integrated posterior | `tests/test_bayesian.py` |
| Distributions | sample moments vs analytic; ppf/cdf round-trip | `tests/test_distributions.py` |
| Asset model | capacity bounds, monotonicity in the down-set, hand-computed cases | `tests/test_asset.py` |
| Engine | reproducibility with seed; mean event counts vs Poisson expectation; zero-rate edge cases; CRN monotonicity (a risk-reducing mitigation never increases loss in any year for frequency-only controls) | `tests/test_engine.py` |
| Metrics | against hand-computed samples; ES ≥ VaR; allocation sums to total | `tests/test_metrics.py` |
| Optimisation | budget respected; MILP = brute force on small instances | `tests/test_optimization.py` |
| Convergence | 10k vs 50k vs 100k years, several seeds; SE and bootstrap CIs | `scripts/run_validation.py` → `docs/validation.md` |
| Model A vs B | identical marginal means (within MC error); different tails | `scripts/run_validation.py` |
