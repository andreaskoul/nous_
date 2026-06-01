"""
Phase 2 data — the REAL corpus (LoCoMo) behind the SAME interface as
synthetic.py, so everything downstream (locate / metric / geodesic / eval) is
unchanged. README's promise: "synthetic.py is the only thing to replace."

LoCoMo (snap-research, CC BY-NC 4.0): 10 long multi-session dialogues, personas
+ temporal event graphs. data/locomo10.json is a list of conversation samples:

    sample["conversation"] = { "session_1": [ {speaker, dia_id, text}, ... ],
                               "session_1_date_time": "...", ... }
    sample["qa"]           = [ {question, answer, evidence:[dia_id,...], category}, ... ]

We map LoCoMo onto the geometric-memory (q, s, h, target) frame:
  * memory bank M = every turn across the split's conversations (cross-conversation
    -> genuine ambiguity & natural hard negatives). target = row of the (first)
    evidence dia_id.
  * q = question embedding.
  * s = conversation fingerprint (mean turn embedding of the QA's conversation)
    -> the disambiguator: the SAME question resolves to different memories in
    different conversations.
  * h = mean embedding of turns preceding the earliest evidence turn.

Roles: this one object serves BOTH as the retrieval corpus (.keys, .query_batch,
anchors) AND as the manifold for the metric (.decode / .sample_latents /
.ambient_dist) via a low-d PCA chart g (its inverse is the decoder for G0=J^T J).

The text encoder is PLUGGABLE (`embed_fn`) so this module is testable offline; the
default is a frozen sentence-transformer, encoded once and cached to disk.
"""
from __future__ import annotations
import json
import os
import re
import torch

_SESSION_RE = re.compile(r"^session_(\d+)$")


# ---- minimal PCA chart (torch only; sklearn not required) --------------------

class TorchPCA:
    """x in R^D  <->  z in R^d.  decode(z) = z @ Vᵀ + mean  is the chart g."""

    def __init__(self, d: int):
        self.d = d
        self.mean = None
        self.V = None          # (D, d), orthonormal columns

    def fit(self, X: torch.Tensor):
        self.mean = X.mean(0)                            # (D,) — NOT keepdim, so
        Xc = X - self.mean                               # decode((d,))->(D,) for jacrev
        # economy SVD; right-singular vectors are the principal axes
        _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
        self.V = Vh[: self.d].T.contiguous()             # (D, d)
        return self

    def encode(self, X: torch.Tensor) -> torch.Tensor:
        return (X - self.mean) @ self.V                  # (.,d)

    def decode(self, Z: torch.Tensor) -> torch.Tensor:
        return Z @ self.V.T + self.mean                  # (.,D)


# ---- default frozen encoder --------------------------------------------------

def make_sentence_encoder(name: str):
    """Returns embed_fn(list[str]) -> (n, D) float tensor, using a frozen
    sentence-transformer. Imported lazily so the package works without it."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(name)
    model.eval()

    @torch.no_grad()
    def embed_fn(texts):
        v = model.encode(list(texts), convert_to_numpy=True,
                         normalize_embeddings=True, show_progress_bar=False)
        return torch.tensor(v, dtype=torch.float32)

    return embed_fn


# ---- LoCoMo parsing ----------------------------------------------------------

def _parse_conversation(sample: dict):
    """-> (turns, qas) for one conversation sample.
    turns: list of dict(dia_id, speaker, text, session, order).
    qas:   list of dict(question, answer, evidence[list dia_id], category)."""
    conv = sample.get("conversation", sample)
    turns, order = [], 0
    sess_keys = sorted((k for k in conv if _SESSION_RE.match(k)),
                       key=lambda k: int(_SESSION_RE.match(k).group(1)))
    for sk in sess_keys:
        sidx = int(_SESSION_RE.match(sk).group(1))
        for t in conv[sk]:
            dia = t.get("dia_id") or t.get("id")
            text = t.get("text") or t.get("clean_text") or ""
            if t.get("blip_caption"):
                text = f"{text} [image: {t['blip_caption']}]"
            if dia is None or not text:
                continue
            turns.append(dict(dia_id=dia, speaker=t.get("speaker", ""),
                              text=text, session=sidx, order=order))
            order += 1
    qas = []
    for qa in sample.get("qa", []):
        ev = qa.get("evidence") or qa.get("evidences") or []
        if isinstance(ev, str):
            ev = [ev]
        q = qa.get("question")
        if q and ev:
            qas.append(dict(question=q, answer=qa.get("answer", ""),
                           evidence=list(ev), category=qa.get("category", -1)))
    return turns, qas


class LoCoMoData:
    """Drop-in for synthetic.ContextCorpus + CurvedManifold on real data."""

    def __init__(self, json_path, split="train", conv_ids=None, embed_fn=None,
                 pca_dim=16, n_anchors=32, emb_cache=None, device="cpu",
                 dtype=torch.float32, pca=None):
        self.device, self.dtype = device, dtype
        with open(json_path) as f:
            data = json.load(f)
        if conv_ids is not None:
            data = [data[i] for i in conv_ids if i < len(data)]

        # ---- gather turns (the memory bank) + QA, leakage-free per split ----
        texts, dia2row, conv_of, qa_items = [], {}, [], []
        for cid, sample in enumerate(data):
            turns, qas = _parse_conversation(sample)
            local_first = len(texts)
            conv_dia = {}
            for t in turns:
                conv_dia[t["dia_id"]] = len(texts)
                dia2row[(cid, t["dia_id"])] = len(texts)
                texts.append(t["text"]); conv_of.append(cid)
            conv_turn_rows = list(range(local_first, len(texts)))
            for qa in qas:
                ev_rows = [conv_dia[d] for d in qa["evidence"] if d in conv_dia]
                if not ev_rows:
                    continue
                first_ev = min(ev_rows)
                hist_rows = [r for r in conv_turn_rows if r < first_ev]
                qa_items.append(dict(question=qa["question"], target=first_ev,
                                    evidence=ev_rows, conv_rows=conv_turn_rows,
                                    hist_rows=hist_rows, category=qa["category"]))
        if not texts:
            raise ValueError(f"No turns parsed from {json_path} (split={split}).")

        # ---- encode (with cache) ----
        embed_fn = embed_fn or (lambda ts: _require_encoder())
        cache = emb_cache and f"{emb_cache}.{split}.pt"
        if cache and os.path.exists(cache):
            keys = torch.load(cache)
        else:
            keys = embed_fn(texts)
            if cache:
                torch.save(keys, cache)
        self.keys = keys.to(device, dtype)                       # (N, D) bank M
        self.D = self.keys.shape[1]

        # ---- question / state / history embeddings per QA ----
        q_emb = (embed_fn([it["question"] for it in qa_items]).to(device, dtype)
                 if qa_items else torch.empty(0, self.D, device=device, dtype=dtype))
        self.q = q_emb / q_emb.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        self.s = torch.stack([self.keys[it["conv_rows"]].mean(0) for it in qa_items]) \
            if qa_items else torch.empty(0, self.D, device=device, dtype=dtype)
        self.s = self.s / self.s.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        self.h = torch.stack([
            self.keys[it["hist_rows"]].mean(0) if it["hist_rows"]
            else torch.zeros(self.D, device=device, dtype=dtype) for it in qa_items]) \
            if qa_items else torch.empty(0, self.D, device=device, dtype=dtype)
        self.target = torch.tensor([it["target"] for it in qa_items], device=device)
        self.evidence = [it["evidence"] for it in qa_items]
        self.category = torch.tensor([it["category"] for it in qa_items], device=device)
        self.conv_of = torch.tensor(conv_of, device=device)

        # ---- anchors (grounding) + PCA chart (the metric's decoder) ----
        K = min(n_anchors, self.keys.shape[0])
        self.cues = self.keys[torch.linspace(0, self.keys.shape[0] - 1, K).long()]
        # reuse a chart fit on TRAIN (so lambda_theta's coordinate system is the
        # same across splits); otherwise fit on this split's bank.
        self.pca = pca if pca is not None else TorchPCA(min(pca_dim, self.D)).fit(self.keys)
        self._lat = self.pca.encode(self.keys)                   # cached latents

    # ----- retrieval-corpus role (mirror ContextCorpus) -----
    def query_batch(self, m=512, seed=7):
        g = torch.Generator().manual_seed(seed)
        idx = torch.randint(0, self.q.shape[0], (m,), generator=g)
        return self.q[idx], self.s[idx], self.target[idx]

    def all_queries(self):
        """(q, s, target, evidence_list) over the whole split — for eval."""
        return self.q, self.s, self.target, self.evidence

    # ----- manifold role (mirror CurvedManifold) -----
    def decode(self, z):                       # chart inverse g: R^d -> R^D
        return self.pca.decode(z)

    def encode_to_latent(self, x):
        return self.pca.encode(x)

    def sample_latents(self, n, seed=1, spread=None):
        g = torch.Generator().manual_seed(seed)
        idx = torch.randint(0, self._lat.shape[0], (n,), generator=g)
        z = self._lat[idx]
        if spread is not None:                 # off-manifold probe: push outward
            z = z + spread * torch.randn(z.shape, generator=g).to(self.device, self.dtype)
        return z

    def ambient_dist(self, a, b):
        return torch.linalg.norm(self.decode(a) - self.decode(b), dim=-1)


def _require_encoder():
    raise RuntimeError(
        "No embed_fn given and no cached embeddings found. Pass embed_fn="
        "make_sentence_encoder(cfg.data.encoder) (needs `pip install "
        "sentence-transformers` and network for the model + LoCoMo download).")
