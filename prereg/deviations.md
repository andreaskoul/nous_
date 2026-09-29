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
