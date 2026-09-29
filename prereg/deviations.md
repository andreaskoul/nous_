# Pre-registration deviations ledger

Append-only record of every change to a pre-registered plan made after it was
written down: gates retired or added, thresholds moved, data or splits changed,
re-freezes. A confirmatory result is read against the plan hash in its lock
file (`prereg/plan_v2.lock.json`) together with the entries below; an entry
dated after the freeze marks the result it affects as deviating from the plan.

Rules: one line per entry, `- <iso-date>: <what changed, why, and whether any
confirmatory result had been seen>`. Never edit or delete past entries; correct
them with a new entry. Append with `manifold_mvp.prereg.log_deviation(text)`.

## Entries

- 2026-09-29: v1 gates retired after the LoCoMo reality check, before any v2 confirmatory run: Gate-1 delta2 = +0.15 top-1 of the context head vs cosine / static / filesystem (honest baselines were missing: BM25+dense hybrid + ms-marco-MiniLM rerank reaches hit@1 0.524 vs 0.164 for the v1 context head), E0 curvature / E1 geodesic-vs-straight (the learned conformal metric collapses to lambda ~ 1 on the real PCA chart), C2 sigma^-1 reconstruction (synthetic paths only; re-posed on procedural data as exploratory P1), and grounding drift (L_gnd had zero gradient). Replaced by prereg/plan_v2.json (status draft, not frozen).
- 2026-09-29: G1a scoped to ANSWERABLE LoCoMo-Conv queries (categories 1-4 rewrites + composed); adversarial rewrites point at a premise-violating trap turn, not at evidence to recall. Plan still draft; no v2 gate result had been seen.
- 2026-09-29: A1/LOCATE: added the premise-consistency per-memory feature name_match (memory speaker named in the query), motivated by the design-validation run (experiments/results/system_one_rerank.json: forced-Choice confidence flags category 5 at AUROC 0.46-0.48). Plan still draft; no v2 gate result had been seen.
- 2026-09-29: C1 estimator revised from per-fold per-category equal-width ECE <= 0.06 to pooled out-of-fold per-category DEBIASED equal-mass ECE <= 0.06 (mean over seeds). Reason: with ~19 open-domain questions per fold the raw estimator's small-sample bias (~0.18 per bin) fails a perfectly calibrated model. Simulation (60 seeds): calibrated model debiased ECE 0.000 at n=96 and n=300 (0% false fails); 20-point overconfidence detected 78% at n=96, 100% at n=300. Plan still draft; no v2 C1 result had been seen. Plan sha256 6dbc4ac2460b -> 87d160d1c4ab.
