from __future__ import annotations

import threading
from pathlib import Path

from core.db import delete_model_record, fetch_expired_models


class ModelCleanupService:
    def __init__(
        self,
        inactive_ttl_sec: int,
        cleanup_interval_sec: int,
        cleanup_batch_size: int,
        cache_invalidator,
    ) -> None:
        self._inactive_ttl_sec = inactive_ttl_sec
        self._cleanup_interval_sec = cleanup_interval_sec
        self._cleanup_batch_size = max(1, cleanup_batch_size)
        self._cache_invalidator = cache_invalidator

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="model-inactive-cleanup",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self.cleanup_once()
            self._stop_event.wait(self._cleanup_interval_sec)

    def cleanup_once(self) -> int:
        expired = fetch_expired_models(
            self._inactive_ttl_sec,
            limit=self._cleanup_batch_size,
        )
        removed = 0

        for model in expired:
            path, deleted = delete_model_record(model["token"])
            if not deleted:
                continue

            if path:
                self._cache_invalidator(path)
                try:
                    Path(path).unlink(missing_ok=True)
                except OSError:
                    # DB deletion should still succeed even if filesystem cleanup fails.
                    pass

            removed += 1

        return removed
