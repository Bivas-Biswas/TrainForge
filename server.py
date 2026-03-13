import uuid
import multiprocessing
import threading
import time

from fastapi import FastAPI
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import numpy as np
from core.cache import ModelCache
from core.config import (
    HEALTH_CHECK_INTERVAL_SEC,
    MODEL_CACHE_SIZE,
    MODEL_DIR,
    TRAINER_COUNT,
    TRAINER_TIMEOUT_SEC,
)
from core.db import (
    create_model_record,
    fetch_model_path_and_status,
    fetch_model_status_details,
    init_db,
    mark_token_failed,
)
from core.model_registry import MODEL_REGISTRY
from core.schemas import InferRequest, TrainRequest
from core.trainer import trainer_worker

executor = ThreadPoolExecutor(max_workers=10)

manager = multiprocessing.Manager()

job_queue = multiprocessing.Queue()

model_cache = ModelCache(max_size=MODEL_CACHE_SIZE)
trainer_pool = {}
trainer_pool_lock = threading.Lock()
trainer_health = manager.dict()
monitor_stop_event = threading.Event()
monitor_thread = None

@asynccontextmanager
async def lifespan(app: FastAPI):

    start_trainer_pool()
    start_trainer_monitor()

    yield

    monitor_stop_event.set()

    if monitor_thread is not None:
        monitor_thread.join(timeout=2)

    # shutdown logic (optional)
    for _ in range(TRAINER_COUNT):
        job_queue.put(None)

    with trainer_pool_lock:
        processes = list(trainer_pool.values())

    for p in processes:
        p.join(timeout=2)


app = FastAPI(lifespan=lifespan)

init_db()


def spawn_trainer(worker_id: str):

    p = multiprocessing.Process(
        target=trainer_worker,
        args=(job_queue, trainer_health, worker_id)
    )

    p.daemon = True
    p.start()

    trainer_health[worker_id] = {
        "status": "starting",
        "token": None,
        "last_heartbeat": time.time(),
    }

    return p

def start_trainer_pool():

    with trainer_pool_lock:
        for _ in range(TRAINER_COUNT):
            worker_id = str(uuid.uuid4())
            trainer_pool[worker_id] = spawn_trainer(worker_id)


def _restart_worker(worker_id: str, reason: str):
    with trainer_pool_lock:
        proc = trainer_pool.get(worker_id)

    state = trainer_health.get(worker_id, {})
    token = state.get("token")

    if proc is not None and proc.is_alive():
        proc.terminate()
        proc.join(timeout=2)

    # If a worker dies mid-training, ensure that token does not stay in "training" forever.
    if reason in {"timeout", "dead"}:
        mark_token_failed(token, f"worker {reason} during training")

    with trainer_pool_lock:
        trainer_pool[worker_id] = spawn_trainer(worker_id)


def monitor_trainers():
    while not monitor_stop_event.is_set():
        now = time.time()

        with trainer_pool_lock:
            snapshots = list(trainer_pool.items())

        for worker_id, proc in snapshots:
            state = trainer_health.get(worker_id, {})
            last = state.get("last_heartbeat", 0)
            status = state.get("status", "unknown")

            if not proc.is_alive():
                _restart_worker(worker_id, "dead")
                continue

            if status == "training" and (now - last) > TRAINER_TIMEOUT_SEC:
                _restart_worker(worker_id, "timeout")

        monitor_stop_event.wait(HEALTH_CHECK_INTERVAL_SEC)


def start_trainer_monitor():
    global monitor_thread

    monitor_stop_event.clear()
    monitor_thread = threading.Thread(
        target=monitor_trainers,
        daemon=True,
        name="trainer-health-monitor",
    )
    monitor_thread.start()


@app.get("/trainers/health")
def trainers_health():
    with trainer_pool_lock:
        snapshots = list(trainer_pool.items())

    data = []

    for worker_id, proc in snapshots:
        state = trainer_health.get(worker_id, {})
        data.append(
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
        "workers": data,
    }

@app.get("/models/types")
def model_types():
    return {"supported_model_types": list(MODEL_REGISTRY)}


@app.post("/train")
def train(req: TrainRequest):

    token = str(uuid.uuid4())

    dataset = req.dataset_path
    model_type = req.model_type
    params = req.params

    if model_type not in MODEL_REGISTRY:
        return {
            "error": f"Unknown model_type '{model_type}'",
            "supported_model_types": list(MODEL_REGISTRY),
        }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = str(MODEL_DIR / f"{token}.bin")

    create_model_record(token=token, status="queued", path=path, model_type=model_type)

    job_queue.put((token, dataset, model_type, params))

    return {"token": token, "status": "queued", "model_type": model_type}

@app.get("/status/{token}")
def status(token: str):

    details = fetch_model_status_details(token)

    if not details:
        return {"error": "unknown token"}

    model_status, error_message = details

    response = {"state": model_status}
    if error_message:
        response["error_message"] = error_message

    return response

def load_model(path):
    return model_cache.load(path)

def infer_worker(token, features):

    row = fetch_model_path_and_status(token)
    if row is None:
        return {"error": "unknown token"}

    if row[1] != "completed":
        return {"error": "model not ready"}

    model = load_model(row[0])

    X = np.array(features).reshape(1, -1)

    try:
        pred = model.predict(X)
    except Exception as exc:
        return {
            "error": "inference failed",
            "details": f"{type(exc).__name__}: {exc}",
        }

    return {"prediction": int(pred[0])}


@app.post("/infer")
def infer(req: InferRequest):

    token = req.token
    features = req.features

    future = executor.submit(
        infer_worker,
        token,
        features
    )

    return future.result()

if __name__ == "__main__":

    import uvicorn

    uvicorn.run("server:app", port=8000)