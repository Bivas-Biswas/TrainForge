from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    dataset_path: str
    model_type: str = "svm"
    params: dict[str, Any] = Field(default_factory=dict)


class InferRequest(BaseModel):
    token: str
    features: list[float]
