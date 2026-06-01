# Implementation Plan — Geometric-Memory MVP on M4

Companion to the Literature Map, the Formalism, and the Act-II Protocol. This
turns the dossier into a build sequence. The accompanying `manifold_mvp/`
package already implements Phase 1 end-to-end; this document explains the
borrow/build decisions behind it and the path from here.

---

## 1. Borrow map (don't reinvent the wheel) — with M4 verdicts

The search pass checked, per subcomponent, whether a mature implementation
exists *and survives contact with Apple Silicon + modern PyTorch*.

| Subcomponent | Best existing option | M4 verdict | Decision in the MVP |
|---|---|---|---|
| Pullback metric `G=JᵀJ`, geodesics | **stochman** (DTU), **latent-geometry** (PyPI) | stochman is torch-native, fine on MPS; latent-geometry is autodiff-agnostic | **Borrowed the method**, re-implemented in ~50 lines (`metric.py`, `geodesic.py`) to avoid a heavy dep and to control the MPS-safe energy formulation |
| Riemannian optimiser | **geoopt** (Riemannian Adam) | works on MPS | **Deferred** — the MVP's geodesic uses plain Adam on Euclidean control points; pull in geoopt only when the metric itself becomes learnable (Phase 3) |
| Modern Hopfield retrieval | **ml-jku/hopfield-layers** | pinned to torch 1.5/1.6 → friction on torch 2.x | **Re-implemented** — retrieval is `softmax(β·φMᵀ)M`, ~10 lines (`locate.py`); not worth a legacy dep |
| Path signature (C2) | **signatory** (Kidger/Lyons) | **UNUSABLE** — torch 1.8–1.11, py3.7–3.9, no recent macOS wheels, source build must match torch exactly | **Re-implemented natively** — truncated signature via Chen recursion (`signature.py`), fully differentiable, runs on MPS |
| Grounding / anti-drift | **Relative Representations** (Moschella & Rodolà) | pure cos-sim to anchors → trivial | **Re-implemented**, ~10 lines (`relative.py`); doubles as the Layer-5 stitch |
| Warped metric backbone | **GAGA** (2410.12779) | reference for Phase 3 | **Deferred** — adopt as the learned-metric backbone once E2 passes on real data |
| Latent-CoT Navigator | **Coconut / PCCoT** | reference for Phase 4 | **Out of MVP scope** — the density `ρ_θ` is the seed; the unroll comes later |
| Text encoder | **sentence-transformers / MLX-embeddings** | both run on MPS; MLX is ANE-adjacent | **Pluggable** (Phase 2); synthetic stand-in ships so the loop runs offline today |

**Net:** the only thing genuinely worth re-implementing rather than importing is
forced by the platform (signatory) or is so small that a dependency costs more
than it saves (Hopfield, relative reps, the energy geodesic). The heavyweight
borrows (GAGA, geoopt, Coconut) are correctly *deferred* to the phase where they
earn their complexity.

---

## 2. The M4 execution model (read once, save yourself a day)

- **Run on the GPU through MPS, with unified memory.** That is where autograd
  Jacobians (`torch.func.jacrev` + `vmap`), the geodesic energy solve, and
  retrieval belong.
- **The "neural chip" (ANE) is not for this.** It is Core ML, inference-only,
  fixed ops. It cannot run the metric Jacobian or the geodesic optimiser. Its
  legitimate role is **Phase 4**: export the *frozen encoder* to Core ML so the
  embedding forward pass runs on the ANE while everything geometric stays on the
  GPU. Treat it as an inference accelerator, not a compute target for the science.
- **No float64 on MPS.** Keep the metric/geodesic in float32 on-device; use the
  `--precision float64` path (auto-CPU) only for the E0 sensitivity check.
- **Expect a few op fallbacks.** Some `torch.func` transforms silently fall back
  to CPU on MPS; the math is identical, only speed differs. Profile before
  optimising.

---

## 3. Phased build-out

### Phase 0 — Environment (½ day)
`python3 -m venv`, `pip install torch numpy`, `python run_mvp.py --quick`. If the
smoke test prints a dashboard, the platform is good. (Add
`sentence-transformers` only when you reach Phase 2.)

### Phase 1 — Analytic MVP on synthetic ground truth ✅ *(this package)*
All four experiments + C2 + grounding run against a world with a *known* answer,
so each operator is validated in isolation before any real data muddies the
signal. Status: **done and passing** (E0 GO, E2 Gate-1 PASS, C2 faithful).

### Phase 2 — Real encoder, real corpus (1 week) — **the decisive phase**
Swap `synthetic.py` only.
1. Pick an encoder that runs on MPS (a small sentence-transformer is fine).
2. Build a corpus where the correct memory genuinely depends on a state `s`
   (e.g. same query, different conversation history → different right answer).
   This is the honest version of the E2 ambiguity the synthetic corpus fakes.
3. Define the metric: cheapest is the **pullback of a frozen decoder**; if there
   is no decoder, fit a low-d **PCA chart** and use its inverse as `g`, or learn
   a conformal `λ_θ`.
4. Re-run E0→E2 with **the same** `PREREG` thresholds. **Gate 1** is whether the
   context head beats *all three* baselines — including the **filesystem
   landmine** (dumb storage scores ~0 under query noise here, but on a real
   benchmark it is brutal — 74% on LoCoMo). Beating cosine is easy and
   meaningless; beating dumb storage is the test.

### Phase 3 — Make the metric learnable + the C2 bridge invertible (2–3 weeks)
Only if Phase 2 passes.
- Replace the fixed conformal factor with a learned `λ_θ(z)` (or adopt **GAGA**'s
  warped metric); pull in **geoopt** for Riemannian Adam on metric parameters.
- Train a real `σ⁻¹` decoder (the MVP uses a linear inverse on invariants). The
  **C2 kill criterion** is: from `σ(γ)` regenerate a trajectory that, when
  unrolled, reproduces the skill. The MVP's 17.7× skill-separability is the
  necessary precondition; invertibility is the sufficient one.

### Phase 4 — Navigator + ANE inference (open-ended)
- Add the latent-CoT unroll (Coconut/PCCoT style) that walks `ρ_θ` as a
  controlled flow — this is where the density-as-parallel-sampler (E3) becomes a
  real test-time mechanism rather than a top-K readout.
- Export the frozen encoder to Core ML (`coremltools`) so embedding inference
  uses the ANE; keep the geometric loop on MPS.

---

## 4. Risk register (the landmines, with where each is handled)

| Risk | Signal | Handled by |
|---|---|---|
| Manifold is flat → geometry decorative | E0 shortcut ≈ 0 | `curvature.py` gate runs first |
| Curvature present but no recall edge | E0 GO yet E1 flat (seen in the synthetic run!) | E1 reported honestly; do not over-claim from E0 alone |
| Dumb storage beats you | filesystem baseline | mandatory baseline in `locate.py` |
| "State" story is decorative | static head ≈ context head | static-head ablation in E2 |
| Signature loses the skill | low intra/inter separability | C2 separability test in `run_mvp.py` |
| Semantic drift | anchor-coord drift large | relative reps (`relative.py`) |
| Density collapses to a point (feature collapse) | `ρ_θ` entropy → 0 | watch retrieval entropy in E2/E3; add an entropy floor if it appears |
| MPS float64 / op fallback surprises | NaNs or CPU stalls | `device.py` resolves precision; profile before tuning |

---

## 5. What to verify by hand before publishing (carried over from the dossier)

- The exact theorem in *Reasoning by Superposition* (2505.12514) before leaning
  on the superposition story for E3.
- The precise signature/log-signature definition you adopt (Lyons/Kidger) — the
  MVP's truncated form is standard, but pin the convention before Act III.
- That on **real** data the E0 GO actually translates into an E1/E2 win; the
  synthetic split (curved but no recall edge) is a standing warning that it may
  not.
