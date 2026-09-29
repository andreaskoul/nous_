"""
Central configuration for nous_ v2 (see docs/DESIGN_v2.md).

Plain dataclasses, no torch import. The SCIENTIFIC CONTRACT is prereg/plan_v2.json
(hash-locked with `python -m manifold_mvp.prereg freeze`); PreReg below is the typed,
numeric view of that plan's thresholds that eval.py applies, and tests/test_config.py
checks that the two agree. Edit thresholds only in a deviation-logged plan revision,
never after seeing held-out results.

v1 gates (delta2 = +0.15 top-1 vs cosine/static/exact-key; E0/E1; C2 recon; anchor
drift) were retired on 2026-09-29 after the LoCoMo reality check
(docs/REALITY_CHECK_2026-09.md). run_mvp.py keeps its own frozen v1 PREREG as the
historical record of the synthetic Phase-1 MVP.
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class PreReg:
    alpha: float = 0.05                 # one-sided, Holm-adjusted across the confirmatory family
    # G1a  C-state (LoCoMo-Conv implicit + composed, recall@10 of gold turns)
    delta_state: float = 0.03
    # G1b  C-recall (LoCoMo cats 1-4, all-evidence recall@30 on multi-hop)
    delta_recall_mh: float = 0.05
    ni_margin: float = 0.02             # non-inferiority margin (G1b overall, G1c hit@3)
    # A1   abstention (LoCoMo category 5)
    null_auroc: float = 0.70
    null_auroc_lo: float = 0.60
    max_false_abstain: float = 0.10
    # C1   calibration (top-1, per category, held-out folds)
    max_ece: float = 0.06
    calib_slope: tuple = (0.8, 1.25)
    # X0   harness reproduction (in-harness reality-check bar, cats 1-4)
    x0_rerank_hit1: float = 0.524
    x0_minilm_hit1: float = 0.176
    x0_tol: float = 0.02
    k_budgets: tuple = (1, 3, 6, 10, 20, 30, 50)


@dataclass
class LossWeights:
    ret: float = 1.0     # multi-positive InfoNCE over the shortlist U {null}
    kd: float = 0.5      # KL distillation from a reranker / System-One teacher
    null: float = 0.5    # Brier on P(null) for unanswerable questions
    ent: float = 0.0     # entropy floor — ablation only (inactive on LoCoMo, opposes InfoNCE)
    geo: float = 0.0     # GAGA-style metric term — exploratory E1' only, never in retrieval


@dataclass
class ModelCfg:
    rank: int = 32               # low-rank state residual in the identity-anchored head
    tau_init: float = 16.0       # Locate temperature init (fitted beta was 16-40 in-harness)
    null_hidden: int = 32        # premise-feature MLP for the null logit
    conformal_alpha: float = 0.10  # evidence sets target >= 90% all-evidence recall
    temperature: str = "fit"     # "fit" (post-hoc on held-out conversations) | "ssmax"
    # C2 (procedural track; not evaluated on LoCoMo)
    c2_depth: int = 3
    c2_channels: int = 4
    c2_steps: int = 24
    c2_n_proto: int = 12
    c2_jitter: float = 0.03
    c2_seed: int = 1


@dataclass
class TrainCfg:
    lr: float = 2e-3
    steps: int = 300
    batch: int = 64
    seeds: tuple = (0, 1, 2, 3, 4)   # >= 5 seeds for every learned part
    teacher: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    kd_T: float = 1.0


@dataclass
class DataCfg:
    locomo_json: str = "data/locomo10.json"
    locomo_conv_json: str = "data/locomo10_dialog.json"
    locomo_conv_multimem: str = "data/locomo10_multimem_full.json"
    encoders: tuple = ("sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5")
    query_instruction: bool = False  # LMEB: instructions hurt dialogue-memory retrieval
    K: int = 50                      # shortlist size
    contiguity: int = 1              # +/- turns added around each anchor (same session)
    rerank_depth: int = 30
    state_mode: str = "speaker_persona"   # "none" | "speaker_persona" | "last_session"
    conv_styles: tuple = ("implicit", "composed")   # G1a styles
    folds: tuple = ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9))
    cache_dir: str = "data/cache"


@dataclass
class Config:
    prereg: PreReg = field(default_factory=PreReg)
    weights: LossWeights = field(default_factory=LossWeights)
    model: ModelCfg = field(default_factory=ModelCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    data: DataCfg = field(default_factory=DataCfg)
