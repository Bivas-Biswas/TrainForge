from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI

from core.config import (
    HEALTH_CHECK_INTERVAL_SEC,
    MODEL_CACHE_SIZE,
    MODEL_DIR,
    TRAINER_COUNT,
    TRAINER_TIMEOUT_SEC,
)
from core.db import create_model_record, fetch_model_status_details, init_db
from core.inference_service import InferenceService
from core.model_registry import MODEL_REGISTRY
from core.schemas import InferRequest, TrainRequest
from core.trainer_pool import TrainerPool


def create_app() -> FastAPI:
    init_db()

    trainer_pool = TrainerPool(
        trainer_count=TRAINER_COUNT,
        trainer_timeout_sec=TRAINER_TIMEOUT_SEC,
        health_check_interval_sec=HEALTH_CHECK_INTERVAL_SEC,
    )
    inference_service = InferenceService(cache_size=MODEL_CACHE_SIZE)
    executor = ThreadPoolExecutor(max_workers=10)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        trainer_pool.start()
        yield
        trainer_pool.stop()
        executor.shutdown(wait=False)

    app = FastAPI(lifespan=lifespan)

    @app.get("/trainers/health")
    def trainers_health():
        return trainer_pool.health_snapshot()

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

        create_model_record(
            token=token,
            status="queued",
            path=path,
            model_type=model_type,
        )

        trainer_pool.enqueue_job(token, dataset, model_type, params)

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

    @app.post("/infer")
    def infer(req: InferRequest):
        future = executor.submit(inference_service.infer, req.token, req.features)
        return future.result()

    return app
