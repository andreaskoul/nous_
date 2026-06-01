"""Manifold-MVP: an analytically-defined prototype of the geometric-memory
architecture (Locate/Traverse/Transform + C2 signature bridge), runnable on an
Apple Silicon M4 via MPS. See README.md."""
from . import device, synthetic, metric, geodesic, curvature, locate, signature, relative, experiments  # noqa
from . import conformal, losses, invert, real  # noqa  (Phase 2/3 end-to-end pipeline)
__all__ = ["device", "synthetic", "metric", "geodesic", "curvature",
           "locate", "signature", "relative", "experiments",
           "conformal", "losses", "invert", "real"]
