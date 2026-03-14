from __future__ import annotations

import numpy as np

from core.cache import ModelCache
from core.db import fetch_model_path_and_status, touch_last_inference_at


class InferenceService:
    def __init__(self, cache_size: int) -> None:
        self._model_cache = ModelCache(max_size=cache_size)

    def _load_model(self, path: str):
        return self._model_cache.load(path)

    def infer(self, client_id: str, token: str, features: list[float]) -> dict:
        row = fetch_model_path_and_status(token, client_id)
        if row is None:
            return {"error": "unknown token"}

        if row[1] != "completed":
            return {"error": "model not ready"}

        model = self._load_model(row[0])
        x = np.array(features).reshape(1, -1)

        try:
            pred = model.predict(x)
        except Exception as exc:
            return {
                "error": "inference failed",
                "details": f"{type(exc).__name__}: {exc}",
            }

        touch_last_inference_at(token, client_id)
        return {"prediction": int(pred[0])}
