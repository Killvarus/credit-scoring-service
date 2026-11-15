"""
HTTP API кредитного скоринга.

    uvicorn app.main:app --host 0.0.0.0 --port 8000
Документация: http://localhost:8000/docs
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from . import monitoring as mon
from .features import matrix
from .model import ModelService
from .schemas import Application, BatchRequest, BatchResponse, Prediction
from .settings import Settings

log = logging.getLogger("scoring")
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    service = ModelService.load(settings.model_dir)
    state["model"] = service
    state["drift"] = mon.DriftMonitor(service.metadata["reference"], settings.drift_window, settings.drift_min_size)
    mon.MODEL_INFO.labels(service.version).set(1)
    log.info("model %s loaded from %s", service.version, settings.model_dir)
    yield
    state.clear()


app = FastAPI(title="Credit scoring API", version="1.0.0", lifespan=lifespan,
              description="PD-модель (LightGBM) на данных UCI Default of Credit Card Clients.")


def _score(applications: list[Application], endpoint: str) -> list[dict]:
    model: ModelService | None = state.get("model")
    if model is None:
        mon.REQUESTS.labels(endpoint, "unavailable").inc()
        raise HTTPException(503, "модель не загружена")
    start = time.perf_counter()
    frame = pd.DataFrame([a.model_dump() for a in applications])
    result = model.predict(frame)
    state["drift"].add(matrix(frame), [r["pd"] for r in result])
    for r in result:
        mon.DECISIONS.labels(r["decision"]).inc()
        mon.PD_HIST.observe(r["pd"])
    mon.LATENCY.labels(endpoint).observe(time.perf_counter() - start)
    mon.REQUESTS.labels(endpoint, "ok").inc()
    return result


@app.get("/health")
def health() -> dict:
    """Liveness: процесс жив."""
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict:
    """Readiness: модель загружена, можно слать трафик."""
    if "model" not in state:
        raise HTTPException(503, "модель не загружена")
    return {"status": "ready", "model_version": state["model"].version}


@app.get("/model")
def model_info() -> dict:
    meta = state["model"].metadata
    return {k: meta[k] for k in ("version", "trained_at", "data_sha256", "metrics", "threshold", "params", "n_train")}


@app.post("/predict", response_model=Prediction)
def predict(application: Application) -> dict:
    return _score([application], "predict")[0]


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest) -> dict:
    return {"predictions": _score(request.applications, "predict_batch")}


@app.get("/drift")
def drift() -> dict:
    """PSI входных признаков и PD в окне последних заявок относительно обучающей выборки."""
    return state["drift"].report()


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(mon.REGISTRY), media_type=CONTENT_TYPE_LATEST)
