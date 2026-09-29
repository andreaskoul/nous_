#!/usr/bin/env python3
"""
DESIGN-VALIDATION: does a System-One stage belong in Locate, and should the
Locate density be a forced CHOICE or per-memory NOUL decisions with a null?

On LoCoMo (per-conversation scope, cats 1-4 for ranking, all cats for abstention):

  shortlist   hybrid RRF(BM25, dense) top-K, with and without the bge query
              instruction (LMEB 2603.12572 reports instructions HURT on LoCoMo)
  rerankers   on the SAME shortlist:
                ce-minilm   cross-encoder/ms-marco-MiniLM-L-6-v2
                ce-bge      BAAI/bge-reranker-base
                laya        open System-One model (convaiinnovations/laya,
                            Apache-2.0, Jev-compatible), a 2-option choice with
                            neutral keys per memory (its card's advice), P(A)
  calibration per-memory P(relevant) via Platt scaling cross-fitted by
              CONVERSATION fold; Brier / ECE per memory
  decisions   CHOICE   : softmax over the shortlist, confidence = max prob
              NOUL     : calibrated p_i per memory, P(null) = prod(1 - p_i)
              targets  : (a) the shortlist contains no evidence ("nothing I
                         retrieved supports this" -> should abstain),
                         (b) adversarial category 5
Usage:
    python experiments/system_one_rerank.py [--n-sub 500] [--k 20] [--no-laya]
"""
from __future__ import annotations
import argparse, json, os, re, sys, time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
os.environ.setdefault("USE_TF", "0")

from locomo_reality_check import load_locomo, encode, per_question, CATS  # noqa: E402
from manifold_mvp import stats  # noqa: E402

FOLDS = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]


def encode_raw_query(name, texts, cache_dir):
    """Encode queries WITHOUT the bge retrieval instruction."""
    import hashlib
    h = hashlib.md5(("\n".join(texts) + name + "raw").encode()).hexdigest()[:12]
    path = os.path.join(cache_dir, re.sub(r"[^A-Za-z0-9]+", "_", name) + f"_qraw_{h}.pt")
    if os.path.exists(path):
        return torch.load(path)
    from sentence_transformers import SentenceTransformer
    v = SentenceTransformer(name, device="cpu").encode(texts, batch_size=64, convert_to_tensor=True,
                                                       normalize_embeddings=True, show_progress_bar=False).float().cpu()
    torch.save(v, path)
    return v


def platt_fit(x, y, iters=500):
    """1-D logistic regression p = sigmoid(a*x + b) by Newton steps."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    mu, sd = x.mean(), x.std() + 1e-9
    z = (x - mu) / sd
    a, b = 1.0, 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(a * z + b)))
        g = np.array([((p - y) * z).sum(), (p - y).sum()])
        w = p * (1 - p) + 1e-9
        H = np.array([[(w * z * z).sum(), (w * z).sum()], [(w * z).sum(), w.sum()]]) + 1e-6 * np.eye(2)
        step = np.linalg.solve(H, g)
        a, b = a - step[0], b - step[1]
        if np.abs(step).max() < 1e-8:
            break
    return lambda v: 1 / (1 + np.exp(-(a * (np.asarray(v, float) - mu) / sd + b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data/locomo10.json"))
    ap.add_argument("--dense", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--n-sub", type=int, default=500, help="stratified subset for the expensive reranker")
    ap.add_argument("--no-laya", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N questions per conversation")
    ap.add_argument("--out", default=os.path.join(ROOT, "experiments/results/system_one_rerank.json"))
    args = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    cache_dir = os.path.join(ROOT, "data/cache")
    os.makedirs(cache_dir, exist_ok=True)
    t0 = time.time()

    turns, conv_of, sess_of, qa, _ = load_locomo(args.data)
    if args.limit:
        seen = {}
        qa = [x for x in qa if seen.setdefault(x["conv"], []).append(1) or len(seen[x["conv"]]) <= args.limit]
    nq = len(qa)
    qconv = np.array([x["conv"] for x in qa]); qcat = np.array([x["cat"] for x in qa])
    questions = [x["q"] for x in qa]
    fold_of = {c: i for i, f in enumerate(FOLDS) for c in f}
    qfold = np.array([fold_of[c] for c in qconv])

    def cands(i):
        return np.arange(qa[i]["off"], qa[i]["off"] + qa[i]["n"])

    # ---- instruction ablation (dense only) ----------------------------------
    T = encode(args.dense, turns, False, cache_dir)
    Q_instr = encode(args.dense, questions, True, cache_dir)
    Q_raw = encode_raw_query(args.dense, questions, cache_dir)
    main_mask = qcat != 5
    instr = {}
    for name, Q in [("with_instruction", Q_instr), ("no_instruction", Q_raw)]:
        sims = (Q @ T.T).numpy()
        m = [per_question(sims[i, cands(i)], cands(i), qa[i]["ev"]) for i in range(nq)]
        instr[name] = {k: round(float(np.mean([r[k] for r, keep in zip(m, main_mask) if keep])), 4)
                       for k in ("hit@1", "hit@5", "hit@10", "mrr")}
    best_q = Q_raw if instr["no_instruction"]["mrr"] >= instr["with_instruction"]["mrr"] else Q_instr
    print(f"instruction ablation: {instr}  ({time.time()-t0:.0f}s)")

    # ---- shortlist: hybrid RRF (BM25 + dense) --------------------------------
    from rank_bm25 import BM25Okapi
    tok = lambda s: re.findall(r"\w+", s.lower())
    sims = (best_q @ T.T).numpy()
    shortlist, rrf_full = {}, {}
    for c in range(int(conv_of.max()) + 1):
        rows = np.where(conv_of == c)[0]
        bm = BM25Okapi([tok(turns[r]) for r in rows])
        for i in np.where(qconv == c)[0]:
            rb = np.argsort(np.argsort(-bm.get_scores(tok(questions[i]))))
            rd = np.argsort(np.argsort(-sims[i, rows]))
            rrf = 1.0 / (60 + rb) + 1.0 / (60 + rd)
            rrf_full[i] = rrf
            shortlist[i] = rows[np.argsort(-rrf)[: args.k]]
    ev_sets = [set(int(e) for e in x["ev"]) for x in qa]
    in_sl = np.array([len(ev_sets[i] & set(shortlist[i].tolist())) > 0 for i in range(nq)])
    all_in_sl = np.array([ev_sets[i] <= set(shortlist[i].tolist()) for i in range(nq)])
    print(f"shortlist@{args.k}: any-evidence recall {in_sl[main_mask].mean():.3f}, "
          f"all-evidence {all_in_sl[main_mask].mean():.3f}  ({time.time()-t0:.0f}s)")

    # ---- stratified subset for the expensive reranker ------------------------
    rng = np.random.default_rng(0)
    sub = []
    for f in range(len(FOLDS)):
        idx = np.where(qfold == f)[0]
        sub += rng.choice(idx, min(len(idx), args.n_sub // len(FOLDS)), replace=False).tolist()
    sub = np.array(sorted(sub))

    # ---- rerankers -----------------------------------------------------------
    raw = {}   # name -> {qidx: scores over shortlist}
    from sentence_transformers import CrossEncoder
    for name, model in [("ce-minilm", "cross-encoder/ms-marco-MiniLM-L-6-v2"), ("ce-bge", "BAAI/bge-reranker-base")]:
        ce = CrossEncoder(model, device="cpu")
        pairs = [(questions[i], turns[j]) for i in range(nq) for j in shortlist[i]]
        sc = np.asarray(ce.predict(pairs, batch_size=64, show_progress_bar=False), float)
        raw[name] = {i: sc[i * args.k: (i + 1) * args.k] for i in range(nq)}
        print(f"  {name} done ({time.time()-t0:.0f}s)")
    if not args.no_laya:
        import laya
        agent = laya.load("convaiinnovations/laya")
        raw["laya"] = {}
        for n_done, i in enumerate(sub):
            qs = {f"m{j}": {"type": "choice",
                            "instructions": f"Memory: {turns[m]}\nDoes this memory contain information that helps answer the question?",
                            "criteria": {"A": "yes, relevant evidence", "B": "no, not relevant"}}
                  for j, m in enumerate(shortlist[i])}
            r = agent.predict("Question: " + questions[i], qs)["answers"]
            raw["laya"][int(i)] = np.array([r[f"m{j}"]["probabilities"]["A"] for j in range(len(shortlist[i]))])
            if n_done % 100 == 0:
                print(f"    laya {n_done}/{len(sub)} ({time.time()-t0:.0f}s)")
        print(f"  laya done ({time.time()-t0:.0f}s)")

    # ---- evaluate ------------------------------------------------------------
    def full_scores(i, sl_scores):
        """rank shortlist by reranker, then the rest by RRF."""
        c = cands(i)
        sc = rrf_full[i] - 10.0
        pos = {int(m): j for j, m in enumerate(c)}
        for j, m in enumerate(shortlist[i]):
            sc[pos[int(m)]] = 10.0 + sl_scores[j]
        return sc

    out = dict(k=args.k, dense=args.dense, n_questions=nq, n_subset=len(sub),
               instruction_ablation=instr,
               shortlist_recall=dict(any=round(float(in_sl[main_mask].mean()), 4),
                                     all=round(float(all_in_sl[main_mask].mean()), 4)),
               ranking={}, calibration={}, decisions={})
    base = {i: rrf_full[i][np.searchsorted(cands(i), shortlist[i])] for i in range(nq)}
    methods = dict(raw); methods["hybrid-only"] = base
    for scope_name, scope in [("subset", sub), ("all", np.arange(nq))]:
        for name, d in methods.items():
            qs_ = [i for i in scope if i in d and qcat[i] != 5]
            if len(qs_) < len(scope) * 0.5 and scope_name == "all":
                continue
            met = [per_question(full_scores(i, d[i]), cands(i), qa[i]["ev"]) for i in qs_]
            row = {}
            for k in ("hit@1", "hit@5", "mrr"):
                v = np.array([r[k] for r in met])
                mean, lo, hi = stats.cluster_ci(v, qconv[qs_])
                row[k] = [round(mean, 4), round(lo, 4), round(hi, 4)]
            row["n"] = len(qs_)
            out["ranking"][f"{name}@{scope_name}"] = row

    # calibration + decisions on the subset (every method has it)
    for name, d in raw.items():
        qs_ = [int(i) for i in sub if int(i) in d]
        xs = np.concatenate([d[i] for i in qs_])
        ys = np.concatenate([[int(m) in ev_sets[i] for m in shortlist[i]] for i in qs_]).astype(float)
        owner = np.concatenate([[i] * len(shortlist[i]) for i in qs_])
        p_cal = np.empty_like(xs)
        for f in range(len(FOLDS)):
            te = qfold[owner] == f
            if te.any() and (~te).any():
                p_cal[te] = platt_fit(xs[~te], ys[~te])(xs[te])
        out["calibration"][name] = dict(brier=round(float(np.mean((p_cal - ys) ** 2)), 4),
                                        ece=round(stats.ece(p_cal, ys), 4),
                                        base_rate=round(float(ys.mean()), 4))
        # question-level decisions
        conf_choice, p_ans_noul, max_noul = [], [], []
        ptr = 0
        for i in qs_:
            s = d[i]
            z = (s - s.max()) / (s.std() + 1e-9)
            pr = np.exp(3 * z); pr /= pr.sum()
            conf_choice.append(pr.max())
            pi = p_cal[ptr: ptr + len(s)]; ptr += len(s)
            p_ans_noul.append(1 - np.prod(1 - pi))
            max_noul.append(pi.max())
        has_ev = in_sl[qs_]; adv = qcat[qs_] == 5
        top1 = np.array([int(shortlist[i][int(np.argmax(d[i]))]) in ev_sets[i] for i in qs_])
        dec = {}
        for label, score in [("choice_maxprob", conf_choice), ("noul_P(answerable)", p_ans_noul), ("noul_max_p", max_noul)]:
            score = np.array(score)
            cov = score >= np.quantile(score, 0.2)          # abstain on the 20% least confident
            dec[label] = dict(auroc_evidence_in_shortlist=round(stats.auroc(score, has_ev), 4),
                              auroc_flags_adversarial=round(stats.auroc(-score, adv), 4),
                              auroc_top1_correct=round(stats.auroc(score[~adv], top1[~adv]), 4),
                              top1_at_80pct_coverage=round(float(top1[~adv & cov].mean()), 4),
                              top1_full_coverage=round(float(top1[~adv].mean()), 4))
        out["decisions"][name] = dec
    out["runtime_s"] = round(time.time() - t0, 1)
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
