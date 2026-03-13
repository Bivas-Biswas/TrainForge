from __future__ import annotations

import os
import resource
import time
from multiprocessing.synchronize import Lock
from queue import Empty
from typing import Any

import joblib

from core.config import MAX_TRAIN_CPU_SECONDS, MAX_TRAIN_MEMORY_BYTES, MODEL_DIR
from core.db import update_model_status
from core.model_registry import build_model, load_dataset


def set_limits() -> None:
    resource.setrlimit(resource.RLIMIT_AS, (MAX_TRAIN_MEMORY_BYTES, MAX_TRAIN_MEMORY_BYTES))
    resource.setrlimit(resource.RLIMIT_CPU, (MAX_TRAIN_CPU_SECONDS, MAX_TRAIN_CPU_SECONDS))


def train_model(token: str, dataset_path: str, model_type: str, params: dict[str, Any]) -> None:
    set_limits()

    x, y = load_dataset(dataset_path)
    model = build_model(model_type, params)
    model.fit(x, y)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"{token}.bin"
    tmp = path.with_suffix(".bin.tmp")

    joblib.dump(model, tmp)
    os.replace(tmp, path)

    update_model_status(token, "completed")


def _heartbeat(health_state, worker_id: str, status: str, token: str | None = None) -> None:
    health_state[worker_id] = {
        "status": status,
        "token": token,
        "last_heartbeat": time.time(),
    }


def trainer_worker(job_queue, health_state, worker_id: str) -> None:
    _heartbeat(health_state, worker_id, "idle")

    while True:
        _heartbeat(health_state, worker_id, "idle")

        job = job_queue.get()
        if job is None:
            _heartbeat(health_state, worker_id, "stopped")
            break

        token, dataset, model_type, params = job
        _heartbeat(health_state, worker_id, "training", token)
        update_model_status(token, "training")

        try:
            train_model(token, dataset, model_type, params)
        except Exception:
            update_model_status(token, "failed")
        finally:
            _heartbeat(health_state, worker_id, "idle")
