"""
Central configuration for the end-to-end geometric-memory pipeline.

Everything tunable lives here as plain dataclasses (no torch import, so it loads
anywhere). PREREG mirrors run_mvp.py's frozen gate contract and EXTENDS it with
the two new watched quantities from the plan (entropy floor, anchor-drift cap).
Freeze these before trusting a run, then never touch them post-hoc.
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class PreReg:
    # geometry / recall gates (identical to run_mvp.py)
    tau_kappa: float = 0.02      # E0: min shortcut (1 - L_geo/L_straight) to call curved
    delta1: float = 0.10         # E1: min (geodesic_top1 - straight_top1)
    delta2: float = 0.15         # E2 / GATE-1: min (context_acc - best_baseline_acc)
    c2_max_recon: float = 0.20   # C2: max sigma^-1 reconstruction rel-error
    # new watched quantities (plan stage 5)
    c2_min_separability: float = 3.0   # C2: min inter/intra skill separability
    entropy_floor_frac: float = 0.25   # L_ent: H_floor = frac * log(N_memories)
    max_anchor_drift: float = 1e-2     # grounding: max tolerated anchor-coord drift


@dataclass
class LossWeights:
    ret: float = 1.0     # InfoNCE retrieval (the project floor)
    geo: float = 0.3     # GAGA-style warp + local distance-matching
    ent: float = 0.05    # entropy floor (anti-collapse)
    gnd: float = 0.1     # anchor-coordinate stability (anti-drift)


@dataclass
class ModelCfg:
    beta: float = 8.0          # Hopfield inverse-temperature
    hidden: int = 128          # recall-head width
    conformal_hidden: int = 64  # lambda_theta width
    c2_depth: int = 3          # truncated signature depth
    c2_channels: int = 4       # trajectory channel count (synthetic skills)
    c2_steps: int = 24         # trajectory length T
    c2_n_proto: int = 12       # skill repertoire size (library of recallable skills)
    c2_jitter: float = 0.03    # per-execution jitter around a prototype
    c2_seed: int = 1           # fixed repertoire seed (train & eval share it)


@dataclass
class TrainCfg:
    lr: float = 2e-3
    lr_metric: float = 1e-3    # lambda_theta (separate, gentler)
    steps_warmup: int = 300    # stage A: recall head only, metric frozen
    steps_joint: int = 300     # stage B: + learnable metric (anneal L_geo)
    steps_c2: int = 1200       # parallel: train sigma^-1 over the skill repertoire
    batch: int = 512
    n_hard_neg: int = 0        # 0 => full-bank InfoNCE; >0 => mine that many hard negs
    seed: int = 0


@dataclass
class DataCfg:
    kind: str = "synthetic"                                   # "synthetic" | "locomo"
    encoder: str = "sentence-transformers/all-MiniLM-L6-v2"   # frozen MPS-friendly encoder
    locomo_json: str = "data/locomo10.json"
    emb_cache: str = "data/locomo_emb.pt"
    pca_dim: int = 16           # latent chart dimension d for the metric
    n_facts: int = 256          # synthetic corpus size
    corpus_dim: int = 64        # synthetic embedding dim D
    noise: float = 0.25         # synthetic query noise
    # leakage-free split BY CONVERSATION INDEX (10 conversations in locomo10)
    train_convs: tuple = (0, 1, 2, 3, 4, 5)
    val_convs: tuple = (6, 7)
    test_convs: tuple = (8, 9)

    def convs_for(self, split: str) -> tuple:
        return {"train": self.train_convs, "val": self.val_convs,
                "test": self.test_convs}[split]


@dataclass
class Config:
    prereg: PreReg = field(default_factory=PreReg)
    weights: LossWeights = field(default_factory=LossWeights)
    model: ModelCfg = field(default_factory=ModelCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    data: DataCfg = field(default_factory=DataCfg)
