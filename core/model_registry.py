from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


ModelBuilder = Callable[[dict[str, Any]], Any]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    builder: ModelBuilder


def _build_svm(params: dict[str, Any]) -> Any:
    return svm.SVC(**params)


def _build_random_forest(params: dict[str, Any]) -> Any:
    return RandomForestClassifier(**params)


def _build_logistic_regression(params: dict[str, Any]) -> Any:
    return LogisticRegression(**params)


def _build_knn(params: dict[str, Any]) -> Any:
    return KNeighborsClassifier(**params)


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "svm": ModelSpec("svm", _build_svm),
    "random_forest": ModelSpec("random_forest", _build_random_forest),
    "logistic_regression": ModelSpec("logistic_regression", _build_logistic_regression),
    "knn": ModelSpec("knn", _build_knn),
}


def build_model(model_type: str, params: dict[str, Any]) -> Any:
    spec = MODEL_REGISTRY.get(model_type)
    if spec is None:
        supported = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model_type '{model_type}'. Supported: {supported}")

    return spec.builder(params)


def load_dataset(dataset_path: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(dataset_path, delimiter=",")
    x = data[:, :-1]
    y = data[:, -1]
    return x, y
