"""
Мониторинг: метрики Prometheus и дрейф входных данных (PSI).

PSI (Population Stability Index) сравнивает распределение признака в последних заявках
с распределением на обучении по одним и тем же бинам:
    PSI = Σ_b (a_b − e_b) · ln(a_b / e_b),   e — доля в бине на train, a — в окне заявок.
Банковское правило: < 0.1 — стабильно, 0.1–0.25 — заметный сдвиг, > 0.25 — модель пора проверять.
"""
from __future__ import annotations

import threading
from collections import deque

import numpy as np
import pandas as pd
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

REGISTRY = CollectorRegistry()
REQUESTS = Counter("scoring_requests_total", "Запросы к API", ["endpoint", "status"], registry=REGISTRY)
DECISIONS = Counter("scoring_decisions_total", "Решения модели", ["decision"], registry=REGISTRY)
LATENCY = Histogram("scoring_latency_seconds", "Время обработки запроса", ["endpoint"], registry=REGISTRY,
                    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0))
PD_HIST = Histogram("scoring_pd", "Распределение предсказанного PD", registry=REGISTRY,
                    buckets=(0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9))
PSI_GAUGE = Gauge("scoring_feature_psi", "PSI признака в окне последних заявок", ["feature"], registry=REGISTRY)
MODEL_INFO = Gauge("scoring_model_info", "Версия загруженной модели", ["version"], registry=REGISTRY)


def psi(expected: np.ndarray, actual: np.ndarray, eps: float = 1e-4) -> float:
    e = np.clip(np.asarray(expected, float), eps, None)
    a = np.clip(np.asarray(actual, float), eps, None)
    e, a = e / e.sum(), a / a.sum()
    return float(((a - e) * np.log(a / e)).sum())


def bin_shares(values: np.ndarray, edges: list[float]) -> np.ndarray:
    idx = np.searchsorted(np.asarray(edges, float), values, side="right")
    return np.bincount(idx, minlength=len(edges) + 1) / max(len(values), 1)


class DriftMonitor:
    """Хранит последние window заявок (признаки и PD) и считает PSI к эталону из metadata.json."""

    def __init__(self, reference: dict, window: int = 1000, min_size: int = 100):
        self.reference = reference
        self.min_size = min_size
        self.buffer: deque = deque(maxlen=window)
        self.lock = threading.Lock()

    def add(self, features: pd.DataFrame, pd_values: list[float]) -> None:
        cols = [c for c in self.reference if c != "pd"]
        with self.lock:
            for row, p in zip(features[cols].to_dict("records"), pd_values):
                self.buffer.append({**row, "pd": p})

    def report(self) -> dict:
        with self.lock:
            frame = pd.DataFrame(list(self.buffer))
        if len(frame) < self.min_size:
            return {"status": "not_enough_data", "window": len(frame), "min_size": self.min_size, "features": {}}
        features = {}
        for col, ref in self.reference.items():
            value = psi(ref["shares"], bin_shares(frame[col].to_numpy(float), ref["edges"]))
            PSI_GAUGE.labels(col).set(value)
            features[col] = {"psi": round(value, 4),
                             "level": "ok" if value < 0.1 else "warning" if value < 0.25 else "alert"}
        worst = max(f["psi"] for f in features.values())
        return {"status": "ok" if worst < 0.1 else "warning" if worst < 0.25 else "alert",
                "window": len(frame), "features": features}
