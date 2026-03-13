import uuid
import sqlite3
import mmap
import multiprocessing
import threading
from collections import OrderedDict

from fastapi import FastAPI
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import numpy as np
import joblib

from trainer_worker import trainer_worker

MODEL_DIR = "model_store"

executor = ThreadPoolExecutor(max_workers=10)

manager = multiprocessing.Manager()

job_queue = multiprocessing.Queue()

TRAINER_COUNT = 20
MODEL_CACHE_SIZE = 32

model_cache = OrderedDict()
model_cache_lock = threading.Lock()

@asynccontextmanager
async def lifespan(app: FastAPI):

    start_trainer_pool()

    yield

    # shutdown logic (optional)
    for _ in range(TRAINER_COUNT):
        job_queue.put(None)

    for p in trainer_pool:
        p.join()


app = FastAPI(lifespan=lifespan)

def init_db():
    conn = sqlite3.connect("metadata.db")

    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS models(
        token TEXT PRIMARY KEY,
        status TEXT,
        path TEXT
    )
    """)

    conn.commit()
    conn.close()

init_db()

trainer_pool = []

def start_trainer_pool():

    for _ in range(TRAINER_COUNT):

        p = multiprocessing.Process(
            target=trainer_worker,
            args=(job_queue,)
        )

        p.daemon = True
        p.start()

        trainer_pool.append(p)

@app.post("/train")
def train(req: dict):

    token = str(uuid.uuid4())

    dataset = req["dataset_path"]
    params = req.get("params", {})

    path = f"{MODEL_DIR}/{token}.bin"

    conn = sqlite3.connect("metadata.db")
    c = conn.cursor()

    c.execute(
        "INSERT INTO models VALUES (?, ?, ?)",
        (token, "queued", path)
    )

    conn.commit()
    conn.close()

    job_queue.put((token, dataset, params))

    return {"token": token, "status": "queued"}

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