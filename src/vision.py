"""Vision helpers wrapping the InsightFace analyzer."""
from __future__ import annotations

from functools import lru_cache
from typing import Tuple

from insightface.app import FaceAnalysis


def _build_analyzer(det_size: Tuple[int, int], use_gpu: bool) -> FaceAnalysis:
    """Create a FaceAnalysis instance with the requested execution provider."""
    providers = ["CPUExecutionProvider"]
    ctx_id = -1
    if use_gpu:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        ctx_id = 0
    app = FaceAnalysis(providers=providers)
    try:
        app.prepare(ctx_id=ctx_id, det_size=det_size)
    except RuntimeError:
        if use_gpu:
            # InsightFace could not use CUDA; fall back to CPU so the app still works.
            app = FaceAnalysis(providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=-1, det_size=det_size)
        else:
            raise
    return app


@lru_cache(maxsize=4)
def get_analyzer(det_size: Tuple[int, int] = (640, 640), use_gpu: bool = False) -> FaceAnalysis:
    """Return a cached ``FaceAnalysis`` instance for CPU or GPU execution."""
    return _build_analyzer(det_size, use_gpu)
