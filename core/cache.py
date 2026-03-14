from __future__ import annotations

import mmap
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import joblib


class ModelCache:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self._cache: "OrderedDict[str, Any]" = OrderedDict()
        self._lock = threading.Lock()

    def load(self, path: str | Path) -> Any:
        path_str = str(path)

        with self._lock:
            cached = self._cache.get(path_str)
            if cached is not None:
                self._cache.move_to_end(path_str)
                return cached

        with open(path_str, "rb") as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                model = joblib.load(mm)

        with self._lock:
            cached = self._cache.get(path_str)
            if cached is not None:
                self._cache.move_to_end(path_str)
                return cached

            self._cache[path_str] = model
            self._cache.move_to_end(path_str)

            if len(self._cache) > self.max_size:
                self._cache.popitem(last=False)

        return model

    def invalidate(self, path: str | Path) -> None:
        path_str = str(path)
        with self._lock:
            self._cache.pop(path_str, None)
