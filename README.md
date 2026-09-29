# nous_ — a calibrated memory layer under System-One / System-Two

`nous_` retrieves the right **verbatim** memories for a query and says **how sure it
is** — including "none of these". It is an open, differentiable candidate +
evidence-density layer that sits *under* a System-One decision model (TypeSafe **Jev**,
the open **Laya**, or a cross-encoder) and a System-Two LLM reader.

> **v2 (2026-09-29).** v1 claimed that "memory as geometry" (a curved learned manifold
> plus state-dependent Hopfield recall) beats retrieval. A reality check on real
> LoCoMo data and the 2025–26 literature **falsified that thesis**:
> - the v1 context head scores hit@1 0.16, against 0.52 for BM25+dense hybrid + cross-encoder rerank;
> - the learned metric collapses to flat;
> - one loss term had zero gradient.
>
> See [`docs/REALITY_CHECK_2026-09.md`](docs/REALITY_CHECK_2026-09.md) (what failed, with
> verified citations, and what Jev changes) and [`docs/DESIGN_v2.md`](docs/DESIGN_v2.md)
> (the new architecture and its math). The synthetic Phase-1 MVP (`run_mvp.py`) is kept
> as the historical record.

## Architecture

```
L0 STORE       verbatim turns + metadata                                  manifold_mvp/store.py
L1 CANDIDATES  BM25 ⊕ dense fusion (α leave-one-conversation-out)         manifold_mvp/candidates.py
               + contiguity expansion, per-conversation scope
L2 LOCATE      calibrated density over shortlist ∪ {∅}: identity-anchored  manifold_mvp/locate.py
               residual state head, per-memory features, fitted
               temperature, learned null slot, conformal evidence sets
L3 DECIDE      pluggable System-One relevance: cross-encoder | Laya |      manifold_mvp/decide.py
               Jev-compatible HTTP (pinned jev-1.13.0, cached),
               neutral ids, quoted untrusted memories, cross-fitted Platt
L4 READ        System-Two LLM reader (not in this repo)
   glue        batching, state, teacher, v2 objective, metrics            manifold_mvp/pipeline.py
   contract    cluster statistics, Holm, frozen pre-registration          manifold_mvp/stats.py, prereg.py, prereg/
```

Three falsifiable claims, each with a pre-registered kill rule (`prereg/plan_v2.json`):
- **C-state:** the asking speaker's state helps when the cue is implicit (LoCoMo-Conv).
- **C-recall:** a better first stage for multi-evidence questions.
- **C-cal:** a calibrated density that can abstain.

## Quickstart

```bash
pip install -r requirements.txt
# data (non-commercial licences; kept out of git)
curl -L -o data/locomo10.json https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
curl -L -o data/locomo10_dialog.json https://raw.githubusercontent.com/MiuLab/LoCoMo-Conv/main/data/locomo10_dialog.json
curl -L -o data/locomo10_multimem_full.json https://raw.githubusercontent.com/MiuLab/LoCoMo-Conv/main/data/locomo10_multimem_full.json

python tests/test_pipeline.py                  # offline integration checks (+ tests/test_*.py per module)
python eval.py --quick                         # smoke run of the v2 gate family
python eval.py                                 # dev run: 5-fold leave-conversations-out, 2 encoders x 5 seeds
python train.py                                # fit one deployable LOCATE model (+ temperature, null threshold, conformal)
python experiments/locomo_reality_check.py     # the September 2026 reality check
python experiments/system_one_rerank.py        # System-One arms (cross-encoder / bge-reranker / Laya), Choice vs Noul+null
python run_mvp.py --quick                      # historical v1 synthetic MVP
```

A confirmatory run is only valid on a frozen plan:
`python -m manifold_mvp.prereg freeze prereg/plan_v2.json`, then `python eval.py --confirmatory`.
Record any later change with `manifold_mvp.prereg.log_deviation`.

## Using Jev (optional)

`manifold_mvp.decide.SystemOneHTTPDecider` sends one Noul relevance question per memory
to a `POST /v1/systemone` endpoint:
- **Arguments:** `base_url=` (the endpoint), `model="jev-1.13.0"` (always pin the version), and the key in `TYPESAFE_API_KEY`.
- **Caching:** every response is cached.
- **Without a vendor key:** the same class works against a local `laya-serve` (open weights, Apache-2.0).

Treat typed-decision probabilities as **uncalibrated until refit on your own held-out
conversations**. Evidence: 2609.32160, 2609.35342, 2609.26758.

## Status and results

| Artifact | What it shows |
|---|---|
| `experiments/results/locomo_reality_check_full.json` | v1 fails on real data; honest baseline bar hit@1 0.524 |
| `experiments/results/system_one_rerank.json` | System-One reranker arms; Choice vs Noul+null abstention |
| `experiments/results/v2_dev_eval.json` | v2 gate family, development evidence (see `docs/REALITY_CHECK_2026-09.md` §6) |

## Threat model

- **The memory bank is plaintext-equivalent.** Embeddings of short turns can be inverted.
- **Storage.** Encrypt at rest, per user, and delete text and vector together.
- **No privacy claims.** Nothing here claims to protect privacy.
- **Untrusted input.** Stored turns are untrusted input to every decision layer.
