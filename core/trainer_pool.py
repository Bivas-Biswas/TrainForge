from __future__ import annotations

import multiprocessing
import threading
import time
import uuid

from core.db import fetch_tokens_by_status, mark_token_failed
from core.trainer import trainer_worker


class TrainerPool:
    def __init__(
        self,
        trainer_count: int,
        trainer_timeout_sec: int,
        health_check_interval_sec: int,
    ) -> None:
        self._trainer_count = trainer_count
        self._trainer_timeout_sec = trainer_timeout_sec
        self._health_check_interval_sec = health_check_interval_sec

        self._manager = multiprocessing.Manager()
        self._job_queue = multiprocessing.Queue()

        self._pool: dict[str, multiprocessing.Process] = {}
        self._pool_lock = threading.Lock()
        self._health_state = self._manager.dict()
        self._last_active_token_by_worker: dict[str, str] = {}

        self._monitor_stop_event = threading.Event()
        self._monitor_thread: threading.Thread | None = None

    @property
    def job_queue(self):
        return self._job_queue

    def start(self) -> None:
        with self._pool_lock:
            for _ in range(self._trainer_count):
                worker_id = str(uuid.uuid4())
                self._pool[worker_id] = self._spawn_worker(worker_id)

        self._start_monitor()

    def stop(self) -> None:
        self._monitor_stop_event.set()

        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=2)

        for _ in range(self._trainer_count):
            self._job_queue.put(None)

        with self._pool_lock:
            processes = list(self._pool.values())

        for proc in processes:
            proc.join(timeout=2)

    def enqueue_job(
        self,
        token: str,
        dataset_path: str,
        model_type: str,
        params: dict,
    ) -> None:
        self._job_queue.put((token, dataset_path, model_type, params))

    def health_snapshot(self) -> dict:
        with self._pool_lock:
            snapshots = list(self._pool.items())

        workers = []

        for worker_id, proc in snapshots:
            state = self._health_state.get(worker_id, {})
            workers.append(
                {
                    "worker_id": worker_id,
                    "alive": proc.is_alive(),
                    "status": state.get("status", "unknown"),
                    "token": state.get("token"),
                    "last_heartbeat": state.get("last_heartbeat"),
                }
            )

        return {
            "trainer_count": len(snapshots),
            "workers": workers,
        }

    def _spawn_worker(self, worker_id: str) -> multiprocessing.Process:
        process = multiprocessing.Process(
            target=trainer_worker,
            args=(self._job_queue, self._health_state, worker_id),
        )
        process.daemon = True
        process.start()

        self._health_state[worker_id] = {
            "status": "starting",
            "token": None,
            "last_heartbeat": time.time(),
        }

        return process

    def _restart_worker(self, worker_id: str, reason: str) -> None:
        with self._pool_lock:
            proc = self._pool.get(worker_id)

        state = self._health_state.get(worker_id, {})
        token = self._resolve_worker_token(worker_id, state)

        if proc is not None and proc.is_alive():
            proc.terminate()
            proc.join(timeout=2)

        # Avoid leaving jobs in training forever after worker failures.
        if reason in {"timeout", "dead"}:
            try:
                mark_token_failed(token, f"worker {reason} during training")
            except Exception:
                # Keep monitor loop alive even if DB is temporarily unavailable.
                pass

        if worker_id in self._last_active_token_by_worker:
            del self._last_active_token_by_worker[worker_id]

        with self._pool_lock:
            self._pool[worker_id] = self._spawn_worker(worker_id)

    def _resolve_worker_token(self, worker_id: str, state: dict) -> str | None:
        token = state.get("token") or self._last_active_token_by_worker.get(worker_id)
        if token:
            return token

        try:
            training_tokens = set(fetch_tokens_by_status("training"))
        except Exception:
            return None

        active_training_tokens = {
            worker_state.get("token")
            for worker_state in self._health_state.values()
            if worker_state.get("status") == "training" and worker_state.get("token")
        }

        unresolved = sorted(training_tokens - active_training_tokens)
        if unresolved:
            return unresolved[0]

        return None

    def _monitor_trainers(self) -> None:
        while not self._monitor_stop_event.is_set():
            now = time.time()

            with self._pool_lock:
                snapshots = list(self._pool.items())

            for worker_id, proc in snapshots:
                state = self._health_state.get(worker_id, {})
                last = state.get("last_heartbeat", 0)
                status = state.get("status", "unknown")
                token = state.get("token")

                if status == "training" and token:
                    self._last_active_token_by_worker[worker_id] = token
                elif status in {"idle", "stopped"}:
                    self._last_active_token_by_worker.pop(worker_id, None)

                if not proc.is_alive():
                    self._restart_worker(worker_id, "dead")
                    continue

                if status == "training" and (now - last) > self._trainer_timeout_sec:
                    self._restart_worker(worker_id, "timeout")

            self._monitor_stop_event.wait(self._health_check_interval_sec)

    def _start_monitor(self) -> None:
        self._monitor_stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_trainers,
            daemon=True,
            name="trainer-health-monitor",
        )
        self._monitor_thread.start()
