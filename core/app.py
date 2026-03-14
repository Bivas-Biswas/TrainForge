from __future__ import annotations

from pathlib import Path
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI

from core.config import (
    HEALTH_CHECK_INTERVAL_SEC,
    MODEL_CLEANUP_BATCH_SIZE,
    MODEL_CLEANUP_INTERVAL_SEC,
    MODEL_CACHE_SIZE,
    MODEL_DIR,
    MODEL_INACTIVE_TTL_SEC,
    TRAINER_COUNT,
    TRAINER_TIMEOUT_SEC,
)
from core.db import (
    create_model_record,
    delete_client_and_models,
    delete_client_model_record,
    fetch_client_details_with_models,
    fetch_client_storage_info,
    fetch_model_status_details,
    init_db,
    is_client_over_storage_limit,
    register_client_activity,
)
from core.inference_service import InferenceService
from core.model_cleanup import ModelCleanupService
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
    model_cleanup_service = ModelCleanupService(
        inactive_ttl_sec=MODEL_INACTIVE_TTL_SEC,
        cleanup_interval_sec=MODEL_CLEANUP_INTERVAL_SEC,
        cleanup_batch_size=MODEL_CLEANUP_BATCH_SIZE,
        cache_invalidator=inference_service.invalidate_model,
    )
    executor = ThreadPoolExecutor(max_workers=10)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        trainer_pool.start()
        model_cleanup_service.start()
        yield
        model_cleanup_service.stop()
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

        client_id = req.client_id
        dataset = req.dataset_path
        model_type = req.model_type
        params = req.params

        register_client_activity(client_id)
        if is_client_over_storage_limit(client_id):
            storage_info = fetch_client_storage_info(client_id)
            total_models_storage_bytes, maximum_models_storage_support_bytes = (
                storage_info if storage_info is not None else (0, 0)
            )
            return {
                "error": "client storage quota exceeded",
                "client_id": client_id,
                "total_models_storage_bytes": total_models_storage_bytes,
                "maximum_models_storage_support_bytes": maximum_models_storage_support_bytes,
            }

        if model_type not in MODEL_REGISTRY:
            return {
                "error": f"Unknown model_type '{model_type}'",
                "supported_model_types": list(MODEL_REGISTRY),
            }

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        path = str(MODEL_DIR / f"{token}.bin")

        create_model_record(
            token=token,
            client_id=client_id,
            status="queued",
            path=path,
            model_type=model_type,
        )

        trainer_pool.enqueue_job(token, dataset, model_type, params)

        return {
            "client_id": client_id,
            "token": token,
            "status": "queued",
            "model_type": model_type,
        }

    @app.get("/status/{token}")
    def status(token: str, client_id: str):
        register_client_activity(client_id)
        details = fetch_model_status_details(token, client_id)

        if not details:
            return {"error": "unknown token"}

        model_status, error_message = details

        response = {"state": model_status}
        if error_message:
            response["error_message"] = error_message

        return response

    @app.post("/infer")
    def infer(req: InferRequest):
        future = executor.submit(
            inference_service.infer,
            req.client_id,
            req.token,
            req.features,
        )
        return future.result()

    @app.get("/clients/{client_id}")
    def get_client_details(client_id: str):
        register_client_activity(client_id)
        details = fetch_client_details_with_models(client_id)
        if details is None:
            return {"error": "unknown client"}
        return details

    @app.delete("/clients/{client_id}")
    def delete_client(client_id: str):
        model_paths, deleted = delete_client_and_models(client_id)
        if not deleted:
            return {"error": "unknown client"}

        removed_files = 0
        for path in set(model_paths):
            inference_service.invalidate_model(path)
            try:
                Path(path).unlink(missing_ok=True)
                removed_files += 1
            except OSError:
                pass

        return {
            "client_id": client_id,
            "deleted": True,
            "deleted_models": len(model_paths),
            "deleted_model_files": removed_files,
        }

    @app.delete("/clients/{client_id}/models/{token}")
    def delete_client_model(client_id: str, token: str):
        path, deleted = delete_client_model_record(client_id, token)
        if not deleted:
            return {"error": "unknown token for client"}

        removed_file = False
        if path:
            inference_service.invalidate_model(path)
            try:
                Path(path).unlink(missing_ok=True)
                removed_file = True
            except OSError:
                removed_file = False

        return {
            "client_id": client_id,
            "token": token,
            "deleted": True,
            "deleted_model_file": removed_file,
        }

    return app
