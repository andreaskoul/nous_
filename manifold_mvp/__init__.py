"""nous_ — a calibrated memory layer under System-One / System-Two (docs/DESIGN_v2.md).

v2 layers: store (L0), candidates (L1), locate (L2), decide (L3), pipeline (glue),
stats + prereg (evaluation contract). Exploratory / historical: metric, conformal,
geodesic, curvature (learned geometry, E1'), signature + invert (C2 procedural keys),
relative (encoder-swap adapter), synthetic + experiments (the Phase-1 MVP, run_mvp.py).

Submodules are imported lazily (`from manifold_mvp import store`), so importing the
package never pulls in heavy optional dependencies."""
import importlib

__all__ = ["store", "candidates", "locate", "decide", "pipeline", "stats", "prereg", "losses",
           "metric", "conformal", "geodesic", "curvature", "signature", "invert", "relative",
           "synthetic", "experiments", "device"]


def __getattr__(name):
    if name in __all__:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
