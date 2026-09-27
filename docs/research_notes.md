# Research notes

A log of modelling decisions, the reasoning behind them, and things tried and
rejected. Kept separate from the methodology so the methodology can stay
declarative.

## Decisions

**Why a frequency–severity simulation and not a single deterministic loss?**
Operational loss is dominated by rare, large events. A single "expected" number
hides that the median year is roughly 60 % of the expected year and that 1 % of
years carry about 16 % of all loss. The annual loss distribution is the object
of interest for appetite and capital questions.

**Why build release frequency from causes?** It makes controls act on the
mechanism they address (inspection → corrosion; permit-to-work competence →
maintenance-induced releases). A single total release rate cannot represent
either control.

**Why fault trees only for the protection layers?** That is where architecture
(redundancy, shared support systems, common cause) changes the answer. The
release-frequency "tree" is a sum of rates plus two AND gates, which is
written directly in `simulation/derived.py` rather than forced into a
probability fault tree.

**Why Shannon decomposition instead of BDDs?** The trees here have two
repeated events, so 2^r = 4 gate evaluations is exact, simple, and
vectorises over thousands of parameter draws. A BDD would be the right tool for
trees with many repeated events.

**Why thinning for common random numbers?** Mitigations change rates. If each
run drew Poisson counts independently, the difference between baseline and
mitigated EAL would be dominated by noise for small effects (a 0.1 M effect on a
23 M EAL with a 0.14 M standard error). Drawing candidate events at the
reference rate and accepting each with probability (actual rate / reference
rate) means a frequency-reducing control only ever *removes* events (tested:
no simulated year gets worse). At 50,000 years the paired standard errors of
the EAL reductions are 0.007–0.07 M, against ≈ 0.29 M for the difference of
two independent runs — a factor of 4 to 40.

**Why severity-ordered event-tree leaves?** Same reason: with outcomes sorted
by severity, improving a barrier moves the same random number to an equal or
less severe outcome.

**Why two dependence models?** The question an ERM function asks is "how much
does co-movement between risks change the tail?" That requires holding
marginals fixed and changing only dependence. Findings: correlated year-level
drivers raise P90–P95 by 5–9 % but leave ES99 almost unchanged, because the
extreme tail is made of single catastrophic events, not accumulations. Even
a much stronger correlation stress moves ES99 by only ~3 %.

**Why is EAL slightly higher under Model B?** Not a bug: the integrity driver
scales frequencies and the logistics driver scales durations; when they are
positively correlated, E[frequency × duration] exceeds the product of the
means. The validation table shows event frequencies are unchanged.

**Why nested (two-loop) simulation?** A pooled run answers "what is the
predictive distribution?"; it cannot say how confident we are in the EAL or
P99. The nested run shows the EAL of a "world" ranges roughly 15–33 M (5–95 %)
purely from parameter uncertainty — a factor of about two, and a factor of
3–4 for P99/ES99. That is a statement about knowledge, and it is the reason the
sensitivity analysis is a headline output.

## Findings worth recording

* The most influential assumption is economic (`bi_value_fraction`), not
  engineering. Before refining ignition probabilities, an operator should pin
  down what a deferred barrel is worth.
* Frequent, small events (trips, compressor failures, weather shut-ins)
  make up about half of the EAL but almost none of the tail. Releases and the
  export pipeline make up over 90 % of ES99.
* Gas-detection coverage is the protection-layer parameter that matters most
  economically: undetected releases become unisolated releases, which cause
  complex-wide shutdowns even when they do not ignite.
* Adding a firewater pump or deluge coverage reduces the fire-protection PFD
  (0.024 → 0.020 / 0.016) but barely changes EAL, because fires are rare
  (~0.02 /yr). Their value is safety and tail severity; a pure EAL cost-benefit
  undervalues them. The optimiser's choice changes with the objective.
* A fourth generator removes most of the independent loss-of-power frequency
  but leaves the common-cause term, so its benefit–cost ratio is poor.

## Tried and rejected

* **Independent Poisson draws per mitigation run** — too noisy (see above).
* **A single correlation between annual losses by source** — cannot keep event
  frequencies fixed while changing dependence, and has no mechanism.
* **Machine learning for any component** — no problem here that it solves
  better than the explicit models; not used.

## Further reading (general texts on the methods)

* Rausand, M. & Høyland, A. *System Reliability Theory: Models, Statistical Methods, and Applications*.
* Vesely, W. E. et al. *Fault Tree Handbook* (NUREG-0492), U.S. NRC.
* Mosleh, A. et al. *Guidelines on Modeling Common-Cause Failures in Probabilistic Risk Assessment* (NUREG/CR-5485), U.S. NRC.
* IEC 61508-6, Annex B (simplified PFD equations) and Annex D (β-factor estimation).
* Aven, T. *Quantitative Risk Assessment: The Scientific Platform*.
* McNeil, A., Frey, R. & Embrechts, P. *Quantitative Risk Management*.
* Saltelli, A. et al. *Global Sensitivity Analysis: The Primer*.
* Vose, D. *Risk Analysis: A Quantitative Guide*.

These are cited for methods only; no parameter value in this repository is
taken from them.
