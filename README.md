# manifold_mvp

An **analytically-defined** prototype of the geometric-memory architecture —
Locate / Traverse / Transform as native operations on a learned manifold, plus
the C2 *trajectory→point* signature bridge — built to run on an **Apple Silicon
M4** (and anywhere else) with nothing heavier than PyTorch.

Every operator has a closed-form mathematical definition (no black-box learned
component is required to make it run); the few learnable parts (the recall head)
are small and optional. The package *is* the Act-II protocol made executable:

```
E0  curvature probe        GATE: is the geometry load-bearing at all?
E1  geodesic vs cosine recall
E2  learned context recall GATE 1: the project floor
E3  density as parallel sampler (optional contribution)
+   C2 signature bridge (recall-as-attractor) ; grounding ; cross-manifold stitch
```

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # torch + numpy is enough
python run_mvp.py                         # auto-selects MPS on an M4
python run_mvp.py --quick                 # ~30s smoke test
python run_mvp.py --precision float64     # forces CPU for the E0 sensitivity run
```

## What runs on the M4 — and what does NOT

- The MVP runs on the **GPU via MPS** using **unified memory**. This is the right
  target for autograd Jacobians (the pullback metric), the energy-minimisation
  geodesic solver, and the Hopfield retrieval.
- The **Neural Engine (ANE)** is *not* used by the research loop. It is reachable
  only through Core ML, is inference-only, and has a fixed op set — it cannot run
  custom Jacobians or geodesic solves. The right use of the ANE is **later**:
  export the *frozen text encoder* to Core ML so embedding inference runs on the
  ANE. That is an inference optimisation, not part of the science.
- **MPS has no float64.** Geometry code sometimes wants it. `device.py` keeps the
  bulk in float32 on-device and auto-falls-back to CPU when you ask for float64.

## Module map

| File | Role | Layer / experiment |
|------|------|--------------------|
| `device.py` | MPS/CPU + the float64 gotcha | infra |
| `synthetic.py` | curved manifold + context corpus w/ known ground truth | data |
| `metric.py` | pullback metric `G=JᵀJ` (+ optional conformal λ) | L0 / L3.2 |
| `geodesic.py` | energy-min geodesic (no Christoffel → MPS-safe) | Traverse |
| `curvature.py` | E0 shortcut-ratio probe + gate | E0 |
| `locate.py` | density `ρ_θ`, Hopfield retrieval, baselines, heads | L2 / E2 |
| `signature.py` | truncated signature (Chen recursion) + Lévy area | C2 |
| `relative.py` | anchor pins (grounding) + Procrustes stitch | Open Q2 / L5 |
| `experiments.py` | E1 geodesic recall, E3 density sampler | E1 / E3 |
| `run_mvp.py` | orchestrator + pre-registered gates + dashboard | — |

## Pre-registration

Thresholds live in `PREREG` at the top of `run_mvp.py`. **Edit them once, before
you trust a run, then never touch them.** Beautiful architectures earn no
benefit of the doubt — a borderline result is a fail.

## Representative output (synthetic, CPU)

```
E0  shortcut 0.28  -> CURVED -> GO
E1  straight 0.87  geodesic 0.87  -> no recall edge (curvature ≠ advantage)
E2  cosine 0.82 | filesystem 0.00 | static 0.79 | CONTEXT 1.00  -> GATE-1 PASS
C2  skill separability 17.7×  | invariant decode rel-err 0.000   -> faithful
GND anchor-coord drift 2e-7 under a change of encoder              -> pinned
```

The E0/E1 split is the point: a manifold can be measurably curved yet the
curvature need not buy recall accuracy. The synthetic world is a scaffold — the
real question is what these numbers do on a **real encoder** (Phase 2).

## End-to-end training & evaluation (Phase 2/3)

`run_mvp.py` validates each operator *in isolation*. The Phase-2/3 pipeline trains
**one model** — the Hopfield recall head `phi(q,s,h)` **+ a learnable conformal
metric `lambda_theta(z)`** — under a single composite loss, with the **C2
trajectory->point bridge** trained in parallel as an invertible module.

```bash
python train.py --quick            # synthetic, no network: full train -> checkpoint
python eval.py  --split test       # gated dashboard (E0/GATE-1/E3/C2/grounding + Recall@k/MRR)
python tests/test_pipeline.py      # unit checks (offline)
```

| File | Role |
|------|------|
| `config.py` | frozen `PREREG` gates + loss weights + hyperparameters (one place) |
| `manifold_mvp/real.py` | **LoCoMo** loader + frozen encoder + PCA chart; drop-in for `synthetic.py` |
| `manifold_mvp/conformal.py` | learnable warp `lambda_theta(z)` → `G = lambda_theta · J^T J` (GAGA-style) |
| `manifold_mvp/losses.py` | `L = w_ret·L_ret + w_geo·L_geo + w_ent·L_ent + w_gnd·L_gnd` |
| `manifold_mvp/invert.py` | genuinely invertible `sigma^-1` (neural + insertion-method baseline) + C2 gate |
| `train.py` / `eval.py` | staged curriculum (head → +metric → C2) / held-out gated validation |

**Composite loss:** InfoNCE retrieval (= the Hopfield logsumexp energy; ANCE-style
hard negatives), a GAGA-style warp + local distance-matching geometry term, an
entropy floor (anti-collapse), and anchor-coordinate grounding (anti-drift). C2 has
its own reconstruction loss + gate (separability ≥3×, recon rel-err ≤0.20).

**Real data (LoCoMo):** `pip install sentence-transformers`, download
`data/locomo10.json` from <https://snap-research.github.io/locomo> (CC BY-NC 4.0),
then `python train.py --data locomo`. Splits are **by conversation** (leakage-free);
the same PCA chart from train is reused at eval so `lambda_theta` transfers.

See `IMPLEMENTATION_PLAN.md` for the phased build-out and the literature map.
