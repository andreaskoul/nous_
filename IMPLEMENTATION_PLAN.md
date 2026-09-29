# Implementation Plan — nous_ v2

Companion to `docs/DESIGN_v2.md` (architecture + math) and
`docs/REALITY_CHECK_2026-09.md` (why v1 was retired). This file tracks the build order,
the evidence each phase must produce, and the risks.

---

## 0. Where we are (2026-09-29)

| Phase | Status | Evidence |
|---|---|---|
| 1 — Analytic MVP on synthetic ground truth (`run_mvp.py`) | ✅ done; kept as the historical record of v1 | synthetic only |
| Reality check on real data + 2025–26 literature + Jev | ✅ done | `experiments/results/locomo_reality_check_full.json`; 52 verified claims |
| v1 thesis (curved metric, geodesic recall, conversation-state head, anchors) | ❌ **retired** | Gate-1 fails (−0.110 on bge-base); λ → 1; `L_gnd` has zero gradient |
| 2′ — v2 modules (store, candidates, locate v2, decide, stats, prereg) | 🔨 in progress | `tests/test_*.py` |
| 2′ — dev evaluation of the v2 gate family (`eval.py`) | 🔨 in progress | `experiments/results/v2_dev_eval.json` |
| 3′ — freeze plan → confirmatory run on **untouched** data | ⏳ | see §3 |

---

## 1. Borrow map (2026 edition)

| Layer | Best open option | Decision |
|---|---|---|
| L0 store | verbatim turns (2601.00821) | **Adopted.** No LLM fact extraction |
| L1 candidates | BM25 ⊕ dense fusion (2606.04194); encoder chosen on LMEB-Dialogue, **not** MTEB (2603.12572) | **Built** (`candidates.py`); no query instructions by default |
| L2 locate | modern-Hopfield read = one attention step (2008.02217) | **Rebuilt as a calibrated density with a null slot** (`locate.py`) — the nous_ core |
| L3 decide | cross-encoders; **Laya** (Apache-2.0, open System-One); hosted **Jev** `jev-1.13.0` (closed) | **Pluggable** (`decide.py`); Jev only as a pinned, cached reference arm |
| Distillation teacher | cross-encoder / Laya over the shortlist (2605.28062) | **Adopted** in the v2 objective |
| Statistics | cluster-t + cluster bootstrap, exact McNemar, Holm (2411.00640) | **Built** (`stats.py`) |
| Pre-registration | frozen plan + hash lock + deviation ledger (2609.34227 style) | **Built** (`prereg.py`, `prereg/`) |
| Learned Riemannian metric (GAGA) | — | **Frozen / exploratory only** (E1′ with controls) |
| Signatures (C2) | — | **Re-scoped** to procedural retrieval keys (P1) |
| NTK-mirror controllers | repo only, no numbers | **Deferred**; behavioural memory only, if ever |

---

## 2. Phase 2′ — build + development evaluation (now)

1. **Modules.** Seven modules with disjoint ownership (`stats`, `prereg`, `store`,
   `candidates`, `decide`, `locate` + `losses`, `c2`/`relative`). Each was implemented,
   adversarially reviewed and fixed, and each has its own offline test file.
2. **Integration.** `manifold_mvp/pipeline.py` does the batching, state, teacher and
   training. `eval.py` runs the gate family (dev mode). `config.py` is the typed view of
   `prereg/plan_v2.json`.
3. **Development evaluation.** 5-fold leave-conversations-out on LoCoMo (G1b, G1c,
   A1, C1) and LoCoMo-Conv (G1a), with 2 encoders × 5 seeds, and X0 reproduction of
   the reality-check bar. Verdicts are **development evidence only**.
4. **System-One arm evidence.** `experiments/system_one_rerank.py` compares a
   cross-encoder, bge-reranker and Laya on the same shortlist, and forced-Choice vs
   Noul-with-null abstention.

Exit criteria: all tests green; X0 reproduces within ±0.02; dev verdicts recorded,
including failures.

## 3. Phase 3′ — confirmatory run

- **Freeze.** `python -m manifold_mvp.prereg freeze prereg/plan_v2.json`, git-tagged.
  Every later change is logged in `prereg/deviations.md`.
- **Held-out data must be untouched.** All 10 LoCoMo conversations and LoCoMo-Conv
  (derived from them) were used in development, so they cannot be confirmatory.
  Candidates:
  - LongMemEval-S *cleaned* (`xiaowu0162/longmemeval-cleaned`): knowledge-update and
    abstention subsets;
  - LoCoMo-Plus (2602.10715);
  - a freshly collected or generated set frozen before any look.
- **End-to-end QA (G1c′).** Needs a fixed reader and a strict judge, which are not
  available offline. Report retrieval-level G1c until then.

## 4. Phase 4′ — System-One integration

- **Laya as teacher and arm.** Refit its temperatures on held-out conversations
  (its card reports ECE 0.466 → 0.081 after refitting), and run the R1 validity checks:
  repeat-run flips, option-name invariance and injection margin.
- **Hosted Jev.** `SystemOneHTTPDecider(model="jev-1.13.0")` with a response cache,
  only with a user-supplied `TYPESAFE_API_KEY`, reported as a labelled reference arm.
- **Optional.** Fine-tune Laya with RLCD (proper-scoring REINFORCE) on
  training-conversation relevance labels, as an open System-One reranker specialised to
  memory.

## 5. Phase 5′ — exploratory tracks (outside the confirmatory family)

- **E1′** graph-geodesic bridging (kNN + session + entity edges; learned low-rank
  local metric) with identity / whitening / random-metric controls. Kill: archive
  `metric.py`, `conformal.py`, `geodesic.py` and `curvature.py` as a negative result.
- **P1** C2 signature keys (lead-lag, windowed, Chen-composable) on SkillEvolBench /
  Mind2Web against mean-pooled and GRU/linear-CDE keys. Kill: drop C2.
- **U1** encoder upgrade by an orthogonal Procrustes query adapter
  (`relative.fit_query_adapter`).
- **Key-side state** (CMR-style temporal context on memories; EvoEmbedding-style
  contextual keys) and supersession/validity fields, evaluated on LongMemEval
  knowledge-update and MemoryAgentBench FactConsolidation.

---

## 6. Risk register (v2)

| Risk | Signal | Handled by |
|---|---|---|
| State is decorative (again) | context ≈ static adapter / shuffled / speaker-prefix | G1a with three controls; kill rule |
| nous_ is just a cheaper copy of its teacher | gains track the teacher and vanish without KD | report recall and latency at matched quality; KD ablation |
| Reranker lowers abstention | correct abstention falls after reranking (seen with Jev: 63.6 → 54.1%) | A1 is gated separately; null slot trained with Brier |
| Typed-decision arms are unstable | option-name flips, nondeterminism, workload-specific calibration | R1 validity gate; neutral ids; cross-fitted Platt |
| Prompt injection through stored turns | near-margin hijacks (2609.28613) | quote memories as untrusted; margin-stratified injection test (planned) |
| 10 clusters is little power | wide cluster CIs | `stats.power_mde` before each gate; confirmatory data beyond LoCoMo |
| Benchmark noise | 6.4% wrong LoCoMo answer keys; lenient judges | retrieval-level gates; report with and without flagged items |
| Privacy | embedding inversion / translation | threat model in DESIGN_v2 §8; no privacy claims |
| Licensing | LoCoMo, LoCoMo-Conv, OpenJev are non-commercial | publishable baselines on Apache/MIT parts (MiniLM, bge, Laya) |
