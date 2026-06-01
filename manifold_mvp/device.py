"""
Device & precision selection for Apple Silicon (M4) and elsewhere.

THE ONE GOTCHA THAT MATTERS ON M-SERIES:
  MPS (Apple's Metal backend) does *not* support float64 at all. Geometry code
  (geodesic ODE / energy minimisation, metric inversion) is often happier in
  float64. So we run the *bulk* of the work in float32 on the GPU (MPS), and
  expose a `--precision float64` escape hatch that automatically falls back to
  CPU for the numerically sensitive curvature probe (E0).

The "Neural Engine" (ANE) is NOT used for the core MVP: it is reachable only
via Core ML and is inference-only with a fixed op set, so it cannot run the
autograd Jacobians / geodesic solves this code is built on. On the M4 the MVP
runs on the GPU + unified memory through MPS. (The frozen text encoder *can*
later be exported to Core ML to run on the ANE — that is an inference
optimisation, not part of the research loop.)
"""
from __future__ import annotations
import torch


def pick_device(prefer: str = "auto") -> torch.device:
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer in ("mps", "auto") and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer in ("cuda", "auto") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve(device: torch.device, dtype: torch.dtype) -> tuple[torch.device, torch.dtype]:
    """MPS cannot do float64 -> silently move such work to CPU."""
    if device.type == "mps" and dtype == torch.float64:
        return torch.device("cpu"), torch.float64
    return device, dtype


def describe(device: torch.device) -> str:
    if device.type == "mps":
        return "Apple Silicon GPU via MPS (unified memory). float32 only on-device."
    if device.type == "cuda":
        return f"CUDA: {torch.cuda.get_device_name(0)}"
    return "CPU"
