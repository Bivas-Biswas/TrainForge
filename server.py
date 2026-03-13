import uuid
import sqlite3
import mmap
import multiprocessing
import threading
import time
from collections import OrderedDict

from fastapi import FastAPI
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import numpy as np
import joblib

from trainer_worker import trainer_worker, MODEL_REGISTRY

MODEL_DIR = "model_store"

executor = ThreadPoolExecutor(max_workers=10)

manager = multiprocessing.Manager()

job_queue = multiprocessing.Queue()

TRAINER_COUNT = 2
MODEL_CACHE_SIZE = 32
HEALTH_CHECK_INTERVAL_SEC = 5
TRAINER_TIMEOUT_SEC = 300

model_cache = OrderedDict()
model_cache_lock = threading.Lock()
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

def init_db():
    conn = sqlite3.connect("metadata.db")

    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS models(
        token TEXT PRIMARY KEY,
        status TEXT,
        path TEXT,
        model_type TEXT DEFAULT 'svm'
    )
    """)

    # Migrate existing tables that may lack the model_type column.
    try:
        c.execute("ALTER TABLE models ADD COLUMN model_type TEXT DEFAULT 'svm'")
    except sqlite3.OperationalError:
        pass

    conn.commit()
    conn.close()

init_db()


def mark_token_failed(token: str):
    if not token:
        return

    conn = sqlite3.connect("metadata.db")
    c = conn.cursor()

    c.execute(
        "UPDATE models SET status='failed' WHERE token=?",
        (token,)
    )

    conn.commit()
    conn.close()


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

    if reason == "timeout":
        mark_token_failed(token)

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
def train(req: dict):

    token = str(uuid.uuid4())

    dataset = req["dataset_path"]
    model_type = req.get("model_type", "svm")
    params = req.get("params", {})

    if model_type not in MODEL_REGISTRY:
        return {
            "error": f"Unknown model_type '{model_type}'",
            "supported_model_types": list(MODEL_REGISTRY),
        }

    path = f"{MODEL_DIR}/{token}.bin"

    conn = sqlite3.connect("metadata.db")
    c = conn.cursor()

    c.execute(
        "INSERT INTO models VALUES (?, ?, ?, ?)",
        (token, "queued", path, model_type)
    )

    conn.commit()
    conn.close()

    job_queue.put((token, dataset, model_type, params))

    return {"token": token, "status": "queued", "model_type": model_type}

@app.get("/status/{token}")
def status(token: str):

    conn = sqlite3.connect("metadata.db")
    c = conn.cursor()

    c.execute(
        "SELECT status FROM models WHERE token=?",
        (token,)
    )

    row = c.fetchone()

    conn.close()

    if not row:
        return {"error": "unknown token"}

    return {"state": row[0]}

def load_model(path):

    with model_cache_lock:
        cached = model_cache.get(path)

        if cached is not None:
            # Bump most recently used model to the end.
            model_cache.move_to_end(path)
            return cached

    with open(path, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            model = joblib.load(mm)

    with model_cache_lock:
        cached = model_cache.get(path)

        if cached is not None:
            model_cache.move_to_end(path)
            return cached

        model_cache[path] = model
        model_cache.move_to_end(path)

        if len(model_cache) > MODEL_CACHE_SIZE:
            # Drop least recently used model.
            model_cache.popitem(last=False)

        return model

def infer_worker(token, features):

    conn = sqlite3.connect("metadata.db")
    c = conn.cursor()

    c.execute(
        "SELECT path,status FROM models WHERE token=?",
        (token,)
    )

    row = c.fetchone()

    conn.close()

    if row[1] != "completed":
        return {"error": "model not ready"}

    model = load_model(row[0])

    X = np.array(features).reshape(1, -1)

    pred = model.predict(X)

    return {"prediction": int(pred[0])}


@app.post("/infer")
def infer(req: dict):

    token = req["token"]
    features = req["features"]

    future = executor.submit(
        infer_worker,
        token,
        features
    )

    return future.result()

if __name__ == "__main__":

    import uvicorn

    uvicorn.run("server:app", port=8000)