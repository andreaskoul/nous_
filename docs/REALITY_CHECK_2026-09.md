# Reality check — September 2026

*Scope:* every load-bearing claim of `nous_` v1, tested two ways: **in-harness on real
data** (LoCoMo, 5-fold leave-conversations-out) and **against the 2025–2026 literature**
(9 research topics + 4 gap-fill topics; 52 load-bearing claims adversarially verified
against primary sources: 10 confirmed, 42 partially confirmed with corrected wording,
0 refuted). Includes the TypeSafe **Jev** "System One" model (launched 2026-09-15).

**Bottom line.** v1 is falsified on real data or unsupported on every load-bearing
claim. What survives is smaller and more useful: an open, calibrated
**candidate + evidence-density layer with an abstain option**, sitting *under* a
System-One decision model and a System-Two LLM reader. See `docs/DESIGN_v2.md`.

---

## 1. In-harness evidence (real LoCoMo)

`experiments/locomo_reality_check.py` → `experiments/results/locomo_reality_check_full.json`.
1,981 questions with resolvable evidence (3 unresolved ids dropped); ranking metrics on
categories 1–4 (n = 1,535); per-conversation retrieval scope; 95% CIs from a
**conversation-cluster bootstrap** (10 clusters — the honest unit).

| System (cats 1–4) | hit@1 | hit@5 | MRR |
|---|---|---|---|
| **Hybrid RRF(BM25, bge-base) + cross-encoder rerank (top-30)** | **0.524** [0.499, 0.548] | **0.721** | **0.613** |
| Hybrid RRF(BM25, bge-base) | 0.304 [0.285, 0.324] | 0.580 | 0.433 |
| Dense bge-small-en-v1.5 | 0.284 [0.265, 0.307] | 0.541 | 0.407 |
| BM25 | 0.264 [0.235, 0.290] | 0.479 | 0.367 |
| Dense bge-base-en-v1.5 | 0.257 [0.231, 0.285] | 0.533 | 0.387 |
| nous_ residual head (bge-base) — *candidate fix* | 0.214 [0.185, 0.246] | 0.493 | 0.346 |
| Dense all-MiniLM-L6-v2 | 0.176 [0.153, 0.199] | 0.401 | 0.284 |
| **nous_ context head, as implemented (bge-base)** | **0.161** [0.140, 0.186] | 0.422 | 0.287 |
| nous_ static head (bge-base) | 0.153 | 0.381 | 0.268 |
| Exact-key "filesystem" | 0.000 | 0.000 | 0.000 |

**Paired contrasts (cluster CI).**
- Context head − dense, same encoder, per-conversation: MiniLM −0.012 [−0.038, 0.020];
  **bge-base −0.096 [−0.123, −0.066]** (significantly *worse*).
- Pooled vs per-conversation scope is indistinguishable, so the conversation "state"
  buys nothing even as a free filter.
- Residual head − dense: MiniLM **+0.038 [0.014, 0.066]**, bge-base **−0.043**
  [−0.068, −0.015]. It looks like a query adapter that compensates for a weak encoder; the
  state's contribution was never isolated.

**Gate-1 (pre-registered: context − best baseline ≥ +0.15 top-1): FAILS.**
Against the baselines it was written for, the margin is −0.025 (MiniLM) and −0.110
(bge-base). Against honest baselines it is −0.374.

**Geometry on a real chart (PCA-16 of MiniLM, learned conformal warp).**
- λ on the data = **1.01 ± 0.07**, so the metric is flat, and E0 is NO-GO (shortcut −0.02).
- E1, top-1 among the 20 nearest: geodesic 0.067 = straight 0.067 < full-dimension cosine 0.133.
- The failure is structural. A linear chart has constant `J`, so `G₀ = VᵀV = I`.
  `L_geo` matches `√λ‖a−b‖` to `‖V(a−b)‖ = ‖a−b‖`, which forces λ → 1. Pullback
  geodesics through a same-dimension or linear decoder are straight lines
  (2505.17517, partially confirmed).

**The Locate density as a decision.**
- With a temperature fitted on *other* conversations (β ≈ 16–40 across all folds, vs
  the configured 8), ECE is **0.04–0.07**, against 0.15–0.28 at β = 8.
- Max-ρ predicts top-1 correctness: AUROC ≈ 0.70.
- It flags adversarial questions at **chance**: AUROC 0.48–0.50. LoCoMo category-5
  questions *have* evidence turns (mteb/LoCoMo qrels, confirmed), so low confidence is
  the wrong signal.

**Defects found in v1 code** (fixed or retired):
- `real.py` built `h` from turns *before the evidence*, which leaks the label. Fixed and regression-tested.
- Memories had no speaker names. Fixed.
- `L_gnd` was applied to the frozen memory bank, so its gradient was exactly 0. Retired.
- The grounding gate rotated points and anchors by the same `Q`, so it was invariant by construction and could never fail. Retired.
- The entropy floor never activated (H = 3.37 nats vs a floor of 1.39). Now used only as an ablation.
- `LinearSignatureInverse` was a global regression, not the insertion method. Renamed.

## 2. Literature reality check (verified)

| v1 claim | Verdict | Key evidence (verification) |
|---|---|---|
| Curved learned manifold → geodesic recall beats straight/cosine (E0/E1) | **Unsupported; freeze as exploratory** | No 2025–26 source shows curvature improving recall. The published "Riemannian memory" gains are flat metrics: CoreMem, 2606.18406 (confirmed), is a global Mahalanobis metric; SuperLocalMemory's Fisher score reduces to cosine, 2603.14588 (partially confirmed). PCA charts destroy retrieval signal (MA-DPR 2509.13562, partially confirmed). The framing itself was already published without experiments ("Memory Has Geometry", 2609.17969, confirmed). |
| State-conditioned query head beats cosine by ≥ +0.15 (Gate-1) | **Falsified in-harness; mis-specified** | From-scratch state rerankers score below dense on LoCoMo (2605.28062, partially confirmed). State helps on the key side (EvoEmbedding 2606.21649, partially confirmed), and query-side state is principled only when the cue is implicit (LoCoMo-Conv 2609.03467, confirmed). |
| Hopfield Locate adds something beyond retrieval | **Only calibration and abstention** | One modern-Hopfield update is one softmax-attention step (2008.02217, partially confirmed). Top-1 is β-invariant. A forced softmax hides uncertain mass (2609.35342, partially confirmed). |
| C2: signature point + learned σ⁻¹ | **Re-scope as a retrieval key** | Truncated signatures are not injective (2606.15332), so the inverse returns a conditional mean. Global signature tokens lose to incremental ones (2602.11805). Raw trajectories beat distilled skills (SkillEvolBench 2605.24117). LoCoMo has no procedures. |
| NTK-mirror controllers as composable memory | **Unsupported; defer** | The repo reports no numbers. Knowledge adapters do not compose under any merge operator (TechQA 57.4 → 23.5; 2609.17346, partially confirmed). |
| Anchors pin meaning / grounding | **Retire** | Inert in training and tautological in eval. Procrustes beats relative representations for encoder swaps (2510.13406, 2605.30596; partially confirmed). |

**About the benchmark itself:**
- LoCoMo QA is saturated and noisy: 6.4% of the answer key is wrong, and the lenient judge accepts 62.8% of vague wrong answers (Penfield audit, partially confirmed).
- LoCoMo is mostly lexical: 97.0% of questions are single-hop and 98.9% are solvable by grep in an oracle analysis (SmartSearch 2603.15599, partially confirmed).
- MTEB rank does not transfer to dialogue memory, and query instructions can hurt (LMEB 2603.12572, partially confirmed).
- Keeping verbatim turns beats LLM extraction (2601.00821).

## 3. Jev / "System One" — what it is and what it changes

**What it is.**
- `jev-1.13.0` from TypeSafe AI is a **closed, API-only, non-autoregressive typed decision model**.
- You send a state plus declared questions and get probabilities over the declared options.
- Three question types: *Choice* (≤ 255 options), *Noul* (absolute yes/no) and *Score*.
- Training method: "RLCD" (RL against proper scoring rules).
- There is no paper, no weights and no fine-tuning.

**Marketing vs measurement.**
- "0% hallucination" holds by construction, because out-of-set options are masked; the vendor calls it "not empirical".
- No public calibration numbers exist.
- The speed and cost multiples come from workflows the vendor chose.

**Independent evidence** (all preprints under two weeks old, from single groups):
- **The typed readout shows no accuracy advantage over a matched label-probability readout.** This rests on absence of evidence: only 5 of 27 papers had the right baseline (2609.32160, partially confirmed).
- **Choice probabilities are distorted relative to Noul** (2609.35342).
- **Decisions follow option names** (32.5% flips with yes/no names; 2609.26758).
- **It is injectable near the decision margin** (2609.28613).

**The memory result that matters (2609.34227, confirmed).** This is pre-registered, on 5 held-out LoCoMo conversations.
- One Jev *Noul* relevance question per turn, over a **dense-only** cosine top-30 shortlist, raises QA accuracy at k=3 from **59.9% to 77.2%**.
- It is non-inferior to a gpt-4o-mini reranker (77.6%) at about ⅓ of the latency.
- It beats Jev-Mem's graph traversal at matched context (77.0% vs 70.6%).
- **But** the gain shrinks to +1.5 at k=20.
- Reranking *lowers* correct abstention (63.6% → 54.1%).
- About 11.5% of questions lose all evidence at the shortlist stage, which no reranker can recover.

Open replicas exist: **Laya** (Apache-2.0, ModernBERT-large 421M, documented RLCD; this
repo installs it) and OpenJev (CC-BY-NC).

**Consequences for nous_:**
1. **Baseline:** a pointwise System-One / cross-encoder reranker over a shortlist is now
   the bar. In-harness, a cross-encoder alone takes hybrid hit@1 from 0.304 to 0.524.
2. **Interface pattern for Locate:** per-memory absolute relevance plus an explicit
   **null** mass, not a forced Choice softmax. Use neutral option ids, quote memory text
   as untrusted, pin the model version and cache responses.
3. **Not** a training component (no gradients), **not** a substitute for pre-registered
   gates (its calibration is workload-dependent), and **not** yet a lifecycle controller
   (Jev-Mem is unablated and lost at matched context).
4. **Positioning:** nous_ is the open, calibrated, high-recall layer that System-One
   models need *upstream*. They need a shortlist (context rot, 32k state limit), and the
   shortlist-miss floor is exactly what a reranker cannot fix.

## 4. The bar nous_ must now beat

In-harness, per-conversation scope, cats 1–4: **hybrid + cross-encoder rerank hit@1 0.524,
MRR 0.613.** Published retrieval bars:
- text-embedding-3-small evidence recall 59.4 / 68.9 / 79.7 / 86.7 at k = 5 / 10 / 25 / 50 (2608.27925);
- training-free session-level BM25 ⊕ max-sim Hit@1 0.752 (2606.04194);
- EvoEmbedding LoCoMo R@10 76.3 (2606.21649);
- LoCoMo-Conv naive MiniLM RAG R@10 of .312 (implicit) and .266 (composed), against .524 / .432 for the best system (2609.03467).

## 5. What survives

- **Verbatim turns as the memory bank.**
- **Locate as a *calibrated* density**, once it has a fitted temperature and a null slot.
- **An identity-anchored residual query adapter**, as a component to be distilled.
- **State-conditioning, only where the cue is implicit** (LoCoMo-Conv, supersession).
- **Signatures as a procedural retrieval key**, tested on procedural data.
- **The pre-registration culture and the cluster-bootstrap harness.**
- **The negative geometry result itself**, which is publishable once run with
  identity / whitening / random-metric controls.

## 6. Design validation (in-harness): System-One arms, Choice vs Noul, instructions

`experiments/system_one_rerank.py` → `experiments/results/system_one_rerank.json`.
Setup: shortlist = hybrid RRF(BM25, bge-small) top-20, per conversation; categories 1–4;
cluster-bootstrap CIs. The Laya arm ran on a stratified 500-question subset
(386 in categories 1–4) because it is expensive on CPU.

| Reranker over the same shortlist | hit@1 | hit@5 | MRR |
|---|---|---|---|
| ms-marco-MiniLM-L-6 cross-encoder (22M) | **0.520** [0.493, 0.548] | 0.698 | **0.603** |
| bge-reranker-base (278M) | 0.493 [0.457, 0.528] | 0.693 | 0.583 |
| Hybrid only (no rerank) | 0.326 [0.300, 0.351] | 0.582 | 0.447 |
| *Subset:* cross-encoder / **Laya** (open System-One, zero-shot, 421M) / hybrid | 0.513 / **0.404** / 0.308 | 0.715 / 0.655 / 0.598 | 0.600 / 0.522 / 0.441 |

**Findings:**

- **The open System-One model helps, but loses to a small cross-encoder.** Laya scores
  +0.096 hit@1 over hybrid and −0.109 against the 22M cross-encoder. This matches the
  audit: typed readouts show no accuracy advantage over label-probability
  (cross-encoder) readouts (2609.32160). Hosted Jev is untested here (no API key), but
  the same study design applies to it through `SystemOneHTTPDecider`.
  **Decision:** the default DECIDE arm is the cross-encoder. Laya and Jev are
  pluggable, labelled arms.
- **Per-memory ("Noul") probabilities are well calibrated after cross-fitted Platt**
  (per-memory ECE 0.004–0.009; base rate 4.5%) for all three arms.
- **Choice vs Noul-with-null (abstention)** — AUROC for flagging category-5 questions:

  | Arm | forced-Choice max prob | Noul max-p |
  |---|---|---|
  | cross-encoder | 0.47 | 0.65 |
  | Laya | 0.46 | 0.63 |
  | bge-reranker | 0.48 | 0.50 (no gain) |

  So the null signal comes from *absolute* per-memory relevance, which a forced
  softmax discards. This is the in-harness analogue of 2609.35342, but it still falls
  short of the A1 bar (0.70). Forced-Choice confidence is still the better predictor of
  top-1 *correctness* (AUROC 0.73 vs 0.71). Selective prediction therefore works:
  top-1 rises from 0.513 at full coverage to 0.585 at 80% coverage. Abstention on
  false premises needs more than confidence.
- **Shortlist ceiling.** At 20 turns the hybrid shortlist holds *any* evidence for 76.7%
  of questions and *all* evidence for 63.9%. Everything a reranker can gain lies inside
  that ceiling. This is C-recall's target (compare 88.5% / 76.9% for a 30-turn cosine
  shortlist in 2609.34227).
- **Query instructions** (bge-small): MRR 0.407 with vs 0.404 without, which is no
  measurable effect. The "instructions hurt" result in LMEB (2603.12572), which used
  task-style instructions on other models, did not replicate at turn level here.
  **Decision:** the default is off, a neutral choice.
