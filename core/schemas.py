from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    client_id: str
    dataset_path: str
    model_type: str = "svm"
    params: dict[str, Any] = Field(default_factory=dict)


class InferRequest(BaseModel):
    client_id: str
    token: str
    features: list[float]
