"""Backward-compatible trainer exports.

The implementation now lives under core/ modules.
"""

from core.model_registry import MODEL_REGISTRY
from core.trainer import trainer_worker

__all__ = ["MODEL_REGISTRY", "trainer_worker"]