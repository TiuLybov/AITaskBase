from typing import Callable, Dict, Any, List, Optional, Union
from dataclasses import dataclass

import numpy as np
import mlflow

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    log_loss,
    roc_auc_score
)


@dataclass
class MetricSpec:
    name: str
    params: Dict[str, Any]

    @classmethod
    def from_config(cls, cfg: Union[str, Dict[str, Any]]) -> "MetricSpec":
        if isinstance(cfg, str):
            return cls(name=cfg, params={})
        if isinstance(cfg, dict):
            return cls(
                name=cfg["name"],
                params=cfg.get("params", {}),
            )
        raise TypeError(f"Invalid metric config: {cfg}")


@dataclass
class Metric:
    fn: Callable
    requires_y_pred: bool = False
    requires_y_proba: bool = False


METRICS_REGISTRY: Dict[str, Metric] = {}


def register_metric(
    name: str,
    *,
    requires_y_pred: bool = False,
    requires_y_proba: bool = False,
):
    def wrapper(fn: Callable):
        METRICS_REGISTRY[name] = Metric(
            fn=fn,
            requires_y_pred=requires_y_pred,
            requires_y_proba=requires_y_proba,
        )
        return fn
    return wrapper


def get_metric(name: str) -> Metric:
    if name not in METRICS_REGISTRY:
        raise KeyError(
            f"Metric '{name}' not registered. "
            f"Available: {list(METRICS_REGISTRY.keys())}"
        )
    return METRICS_REGISTRY[name]


@register_metric("accuracy", requires_y_pred=True)
def accuracy(y_true, y_pred, **kwargs) -> float:
    return accuracy_score(y_true, y_pred)


@register_metric("f1", requires_y_pred=True)
def f1(y_true, y_pred, average="macro", **kwargs) -> float:
    return f1_score(y_true, y_pred, average=average)


@register_metric("precision", requires_y_pred=True)
def precision(y_true, y_pred, average="macro", zero_division=0, **kwargs) -> float:
    return precision_score(y_true, y_pred, average=average, zero_division=zero_division)


@register_metric("recall", requires_y_pred=True)
def recall(y_true, y_pred, average="macro", zero_division=0, **kwargs) -> float:
    return recall_score(y_true, y_pred, average=average, zero_division=zero_division)


@register_metric("logloss", requires_y_proba=True)
def multiclass_logloss(y_true, y_proba, **kwargs) -> float:
    return log_loss(y_true, y_proba)


@register_metric("dataset_size")
def dataset_size(y_true, **kwargs) -> int:
    return int(len(y_true))


@register_metric("num_classes")
def num_classes(y_true, **kwargs) -> int:
    return int(len(np.unique(y_true)))


@register_metric("class_imbalance_ratio")
def class_imbalance_ratio(y_true, **kwargs) -> float:
    _, counts = np.unique(y_true, return_counts=True)
    return float(counts.max() / counts.min())


@register_metric("major_class_fraction")
def major_class_fraction(y_true, **kwargs) -> float:
    _, counts = np.unique(y_true, return_counts=True)
    return float(counts.max() / counts.sum())


@register_metric("auc", requires_y_proba=True)
def multiclass_auc(
    y_true,
    y_proba,
    average: str = "macro",
    multi_class: str = "ovr",
    **kwargs,
) -> float:
    return roc_auc_score(
        y_true,
        y_proba,
        average=average,
        multi_class=multi_class,
    )


def compute_metrics(
    metrics_cfg: List[Union[str, Dict[str, Any]]],
    *,
    y_true = None,
    y_pred: Optional[np.ndarray] = None,
    y_proba: Optional[np.ndarray] = None,
    prefix: str = "",
    log_to_mlflow: bool = False,
) -> Dict[str, float]:
    results: Dict[str, float] = {}

    for raw_cfg in metrics_cfg:
        spec = MetricSpec.from_config(raw_cfg)
        metric = get_metric(spec.name)

        if metric.requires_y_pred and y_pred is None:
            raise ValueError(f"Metric '{spec.name}' requires y_pred")
        if metric.requires_y_proba and y_proba is None:
            raise ValueError(f"Metric '{spec.name}' requires y_proba")

        value = metric.fn(
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
            **spec.params,
        )

        metric_name = _build_metric_name(prefix, spec)
        value = float(value)

        results[metric_name] = value

        if log_to_mlflow:
            mlflow.log_metric(metric_name, value)

    return results

def _sanitize_value(value) -> str:
    return str(value).replace(".", "_").replace(" ", "")

def _build_metric_name(prefix: str, spec: MetricSpec) -> str:
    if not spec.params:
        return f"{prefix}{spec.name}"

    params_part = "_".join(
        f"{k}_{_sanitize_value(v)}"
        for k, v in sorted(spec.params.items())
    )

    return f"{prefix}{spec.name}_{params_part}"
