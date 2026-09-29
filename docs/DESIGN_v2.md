# nous_ v2 — a calibrated memory layer under System-One / System-Two

Supersedes the v1 "memory as geometry" design after the September 2026 reality check
(`docs/REALITY_CHECK_2026-09.md`). Every choice below is tied to an in-harness
measurement or a verified source.

## 1. Positioning

| | v1 (retired) | v2 |
|---|---|---|
| Claim | A curved learned manifold plus state-dependent Hopfield recall beats retrieval | An **open, differentiable, calibrated** candidate + evidence-density layer that System-One decision models (Jev / Laya / cross-encoders) and System-Two LLM readers need **upstream** |
| Where the value is | Geometry of the embedding space | **Recall** of the shortlist, **calibration**, **abstention**, and state where the cue is implicit |
| Relation to Jev | "Locate is like Jev" | Jev reads the query and each memory *jointly* (cross-encoder-like) and needs a shortlist. nous_ supplies the shortlist and a calibrated density with a null option |

Three falsifiable claims, each with a pre-registered kill rule (`prereg/plan_v2.json`):

- **C-state** — Conditioning on *who is asking* and what they have said raises recall
  when the cue is implicit (LoCoMo-Conv implicit / composed). It must beat three arms:
  a same-data static adapter, a shuffled-state control, and the non-learned
  speaker-name prefix.
- **C-recall** — As a first stage, nous_ raises **all-evidence** shortlist recall on
  multi-hop questions. This is the ≈11.5% loss that no reranker can recover
  (2609.34227).
- **C-cal** — Its evidence density is calibrated and can **abstain**, which a forced
  softmax cannot do.

## 2. Architecture

```
            query q  (+ state: asking speaker's persona / session context)
               │
L0  STORE      verbatim turns + metadata (conv, session, order, speaker, date, dia_id)   store.py
               │
L1  CANDIDATES BM25 ⊕ dense (z-score fusion, α fitted leave-one-conversation-out)        candidates.py
               + contiguity expansion (±n turns, same session); per-conversation scope
               │  shortlist S (K ≈ 30–150)
L2  LOCATE     ρ(m | q, s, h) over S ∪ {∅}  — the nous_ core                               locate.py
               residual state head · per-memory features · fitted temperature ·
               learned null slot · conformal evidence set
               │  calibrated density, evidence set, P(∅)
L3  DECIDE     pluggable System-One relevance (Noul-style, neutral ids, quoted memories):   decide.py
               cross-encoder │ Laya (open) │ Jev-compatible HTTP (pinned jev-1.13.0, cached)
               + cross-fitted Platt calibration → abstain / select k
               │
L4  READ       System-Two LLM reader over selected verbatim turns (fixed reader & judge)

P   PROCEDURAL C2: signature KEYS into a raw-trajectory store (Chen-composable)             signature.py, invert.py
X   EXPLORATORY learned metric / geodesics — frozen, only with identity/whitening/random     metric.py, conformal.py,
                controls (E1′)                                                              geodesic.py, curvature.py
U   UPGRADE    encoder swap = orthogonal Procrustes query adapter                           relative.py
    EVAL       conversation-cluster statistics, exact McNemar, Holm, power                  stats.py
    PREREG     frozen plan + hash lock + deviation ledger                                   prereg.py, prereg/
```

## 3. The closed forms

**L1 fusion.** For each conversation's memories, with `z(·)` the within-query z-score:

  `f(m) = α · z(BM25(q, m)) + (1 − α) · z(cos(e_q, e_m))`

α is chosen per held-out conversation on the *other* conversations. The shortlist is
the top-K by `f`, and each anchor is followed by its ±n same-session neighbours.

**L2 Locate.** For shortlist memories `m₁…m_K` with feature vectors `f_m`:

- BM25 z-score, dense cosine and fused score;
- a contiguity flag;
- **speaker_match**: the memory was said by the asking speaker (a state feature);
- **name_match**: the memory's speaker is *named* in the query (a premise-consistency
  feature). LoCoMo's adversarial questions attribute a real turn to the wrong person,
  which confidence alone cannot flag (in-harness AUROC 0.46–0.48);
- L1 rank.

```
φ(q, s, h) = normalize( q + B·A·[s; h] )            B zero-initialised ⇒ φ = q at init
ℓ_m        = e^{log τ} · ⟨φ, m⟩ + wᵀ f_m             (τ ≈ 16 at init; fitted post hoc)
ℓ_∅        = g( max ℓ, ℓ₍₁₎ − ℓ₍₂₎, mean top-3 ℓ, logsumexp ℓ, |S|/K ) + b,  b₀ = −2
ρ          = softmax( ℓ₁ … ℓ_K , ℓ_∅ )             P(∅) = ρ_∅
```

- The residual is identity-anchored because it was the only learned variant that
  beat its encoder in-harness (+0.038 on MiniLM). The from-scratch ContextHead lost
  (−0.096 on bge-base).
- The null logit reads *premise* features of the whole shortlist, because a low max-ρ
  flags unanswerable questions only at chance (AUROC 0.49).
- **Temperature.** Top-1 is τ-invariant, but calibration is not. The scalar `c` that
  multiplies the memory logits is fitted post hoc to minimise the multi-positive NLL
  on held-out conversations (fitted β was 16–40 in-harness, never 8). Alternatively,
  use scalable-softmax scaling `s · log N` (2501.19399).
- **Conformal evidence set** (split conformal, all-evidence recall target 1 − α):
  - For each calibration question, `r = −min_{m ∈ gold} score_m`.
  - Set `thr = −(the ⌈(n+1)(1−α)⌉-th smallest r)`.
  - The set is `{m : score_m ≥ thr}`.
  - Questions inside a conversation are not exchangeable, so calibrate
    leave-one-conversation-out.

**L3 Decide.** Every System-One arm returns a raw relevance `r_m` per memory, and a
Platt map is cross-fitted by conversation:

  `p_m = σ(a · (r_m − μ)/σ_r + b)`,  `P(∅) = ∏_m (1 − p_m)` (independent-Noul null)

This is set against the forced-Choice confidence `max softmax(r)`, which cannot represent
"none of these" (2609.35342).

**Objective** (replaces InfoNCE + GAGA + entropy floor + grounding):

```
L = L_mpNCE + w_kd · KL( softmax(teacher/T) ‖ softmax(ℓ_S/T) ) · T² + w_∅ · ( P(∅) − 1[unanswerable] )²
L_mpNCE = −log Σ_{m ∈ gold} ρ_m         (the ∅ column is the positive for unanswerable rows)
```

- The teacher is the cross-encoder or System-One score over the shortlist. Teacher
  distillation with lexical features is what worked for learned LoCoMo rerankers
  (2605.28062).
- InfoNCE is already a proper scoring objective. What was missing is *recalibration*,
  not RL (Laya card; 2507.16806).
- `L_gnd` is removed (its gradient was 0).
- The entropy floor survives only as an ablation that must improve held-out ECE.
- `L_geo` survives only inside exploratory E1′.

## 4. System-One integration contract (Jev, Laya, cross-encoders)

- **Ask pointwise absolute questions (Noul-style), one per memory, in one request.**
  Do not ask a forced Choice across memories.
- **Use neutral option / memory ids.** Typed heads follow option names (32.5% flips with
  yes/no names, 2609.26758). Laya's `noul` can follow its `false:/true:` labels, so we
  ask a 2-option `choice` with keys A/B instead.
- **Treat memory text as untrusted data and quote it.** Near-margin decisions are
  hijackable (2609.28613).
- **Pin the model version** (`jev-1.13.0`, not `jev-latest`), **cache every response**,
  and report repeat-run flip rates.
- **Always recalibrate on held-out conversations.** Vendor and model-card calibration
  does not transfer across workloads (2609.32160, 2609.26550).
- **Measured in-harness** (`docs/REALITY_CHECK_2026-09.md` §6), over the same 20-turn
  shortlist:

  | Arm | hit@1 | Notes |
  |---|---|---|
  | ms-marco-MiniLM cross-encoder | **0.520** | 22M params |
  | bge-reranker-base | 0.493 | |
  | Laya (zero-shot) | 0.404 | 421M params |
  | hybrid only | 0.326 | |

  - **Default DECIDE arm: the cross-encoder.** Laya and hosted Jev are labelled
    alternative arms.
  - **The abstention signal comes from absolute per-memory relevance**
    (Noul max-p, AUROC 0.63–0.65 for flagging false premises), **not** from forced-Choice
    confidence (0.46–0.48). That is why nous_ carries an explicit null slot plus
    premise features instead of reading abstention off `max ρ`.
- Hosted Jev is an optional, labelled reference arm (`SystemOneHTTPDecider`). It also
  talks to a local `laya-serve`, so the pipeline stays reproducible without a vendor key.

## 5. Evaluation contract

- **Metrics.** Report both *any-evidence* hit@k and *all-evidence* recall@k at
  k ∈ {1, 3, 6, 20, 30, 50}, per category, plus MRR. Include a shortlist-miss vs
  rerank-drop decomposition, tokens per question and latency.
- **Scope.** Per conversation, because every deployed system partitions memory by user
  or conversation.
- **Splits.** 5-fold leave-conversations-out for development. The confirmatory run is
  only on held-out conversations, after `python -m manifold_mvp.prereg freeze` (eval
  refuses otherwise).
- **Statistics.** Conversation-cluster t-intervals **and** cluster bootstrap, exact
  McNemar for paired hit@k, Holm across the confirmatory family, ≥ 5 seeds for learned
  parts, and a power/MDE statement before each gate. PASS requires both lower bounds
  to clear the margin.
- **Baseline ladder.** BM25; dense ×3 encoders, with and without query instructions;
  fusion with LOCO α; fusion + cross-encoder; fusion + Laya; fusion + hosted Jev
  (optional, cached); the same-data static adapter; the shuffled-state control; the
  speaker-name prefix.
- **Gates** (`prereg/plan_v2.json`): X0 (harness reproduces the reality-check bar),
  G1a (C-state), G1b (C-recall), G1c (pipeline non-inferiority), A1 (abstention),
  C1 (calibration), R1 (typed-decision validity). E1′, P1 and U1 are exploratory.

## 6. Kill criteria

- **G1a fails** → drop state-conditioning as a thesis.
- **E1′ fails its controls** → archive `metric.py` / `conformal.py` / `geodesic.py` /
  `curvature.py` as a documented negative result.
- **P1 fails** → drop C2.

If both G1a and E1′ fail, what remains is a well-engineered fusion → calibrated-density →
System-One pipeline with abstention. That is decided now, not after the fact.

## 7. Retired from v1, and why

| Component | Why |
|---|---|
| Gate-1 (Δ ≥ 0.15 top-1 vs cosine / static / exact-key) | Any second-stage scorer clears it for free, and it tested only the argmax of the query map. It failed anyway (−0.110 on bge-base) |
| Conversation-mean state `s`, last-session `h` | Constant within a conversation, so not identifiable. Under per-conversation scope it adds nothing |
| From-scratch `ContextHead` (default path) | Significantly worse than its own encoder on bge-base |
| Learned conformal metric in the retrieval path; E0/E1 as gates | λ → 1 on a real chart; geodesic = straight < cosine |
| `L_gnd`, anchor "grounding" gate | Zero gradient; invariant by construction |
| Fixed β = 8 | Miscalibrated (ECE 0.15–0.28); fitted β is 16–40 |
| "σ⁻¹ inverts the signature" | Truncated signatures are non-injective; the repertoire test is a lookup |
| NTK-mirror controller memory | Unvalidated; knowledge adapters do not compose (2609.17346) |

## 8. Threat model

- **The memory bank is plaintext-equivalent.** Embedding inversion recovers short turns
  (2310.06816), and vectors can be translated from unknown encoders (2510.02348).
- **Storage rules.** Encrypt at rest, per user. Deletion removes both text and vector.
  No raw-vector export.
- **No privacy claims.** Nothing in nous_ (anchors, noise) is claimed to protect privacy.
- **Untrusted input.** Every stored turn is untrusted input to any decision layer.
