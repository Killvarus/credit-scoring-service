"""
Шаги пайплайна переобучения. Каждый шаг — обычная функция: её вызывают и CLI (training/train.py),
и DAG Airflow (airflow/dags/), и тесты.

extract → validate → split → train → evaluate → compare_with_production → save (promote)
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from app.features import FEATURES, PAY_COLS, PAYAMT_COLS, RAW_COLUMNS, matrix

TARGET = "default"
OPENML_ID = 42477
# Параметры — лучшие из случайного поиска в репозитории credit-scoring (5-fold CV)
PARAMS = dict(objective="binary", learning_rate=0.02, num_leaves=7, min_child_samples=20, colsample_bytree=0.7,
              reg_lambda=20.0, subsample=0.8, subsample_freq=1, verbose=-1, seed=42, num_threads=4)
DRIFT_FEATURES = ["LIMIT_BAL", "AGE", "PAY_1", "util_mean", "pay_ratio_mean", "delay_max"]
GAIN_GOOD, LOSS_BAD = 1.0, 5.0  # условная экономика: +1 за хорошего одобренного, −5 за дефолтника


class DataValidationError(ValueError):
    pass


def extract(cache: Path | None = None) -> pd.DataFrame:
    """UCI Default of Credit Card Clients с OpenML (колонки x1..x23 переименовываются)."""
    if cache is not None and cache.exists():
        return pd.read_csv(cache)
    from sklearn.datasets import fetch_openml

    frame = fetch_openml(data_id=OPENML_ID, as_frame=True, parser="auto").frame
    frame.columns = [*RAW_COLUMNS, TARGET]
    frame[TARGET] = frame[TARGET].astype(int)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(cache, index=False)
    return frame


def validate(frame: pd.DataFrame) -> dict:
    """Проверки схемы и диапазонов перед обучением — дешёвая защита от «тихо сломанных» данных."""
    errors = []
    missing = set(RAW_COLUMNS + [TARGET]) - set(frame.columns)
    if missing:
        raise DataValidationError(f"нет колонок: {sorted(missing)}")
    if frame[RAW_COLUMNS + [TARGET]].isna().any().any():
        errors.append("есть пропуски")
    if not frame[TARGET].isin([0, 1]).all():
        errors.append("целевая переменная не бинарная")
    if (frame["LIMIT_BAL"] <= 0).any():
        errors.append("LIMIT_BAL ≤ 0")
    if not frame["AGE"].between(18, 100).all():
        errors.append("AGE вне [18, 100]")
    if not frame[PAY_COLS].stack().between(-2, 9).all():
        errors.append("статус платежа вне [-2, 9]")
    if (frame[PAYAMT_COLS] < 0).any().any():
        errors.append("отрицательный платёж")
    rate = frame[TARGET].mean()
    if not 0.05 <= rate <= 0.5:
        errors.append(f"доля дефолтов {rate:.3f} вне [0.05, 0.5] — данные подозрительные")
    if len(frame) < 1000:
        errors.append(f"слишком мало строк: {len(frame)}")
    if errors:
        raise DataValidationError("; ".join(errors))
    return {"rows": len(frame), "default_rate": float(rate)}


def split(frame: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """train 60% / valid 20% (ранняя остановка и порог) / test 20% (итоговые метрики)."""
    train, test = train_test_split(frame, test_size=0.2, stratify=frame[TARGET], random_state=seed)
    train, valid = train_test_split(train, test_size=0.25, stratify=train[TARGET], random_state=seed)
    return train, valid, test


def train(train_df: pd.DataFrame, valid_df: pd.DataFrame, params: dict = PARAMS, rounds: int = 2000) -> lgb.Booster:
    dtrain = lgb.Dataset(matrix(train_df), train_df[TARGET], free_raw_data=False)
    dvalid = lgb.Dataset(matrix(valid_df), valid_df[TARGET], reference=dtrain)
    return lgb.train({**params, "metric": "auc"}, dtrain, num_boost_round=rounds, valid_sets=[dvalid],
                     callbacks=[lgb.early_stopping(100, verbose=False)])


def best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Порог PD, максимизирующий условную прибыль на valid."""
    order = np.argsort(p)
    bad = np.cumsum(y[order])
    good = np.arange(1, len(y) + 1) - bad
    profit = GAIN_GOOD * good - LOSS_BAD * bad
    k = int(np.argmax(profit))
    return float((p[order][k] + p[order][min(k + 1, len(y) - 1)]) / 2)


def evaluate(booster: lgb.Booster, valid_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    threshold = best_threshold(valid_df[TARGET].to_numpy(), booster.predict(matrix(valid_df)))
    y, p = test_df[TARGET].to_numpy(), booster.predict(matrix(test_df))
    fpr, tpr, _ = roc_curve(y, p)
    auc = roc_auc_score(y, p)
    approved = p < threshold
    return {
        "threshold": threshold,
        "metrics": {
            "roc_auc": round(float(auc), 4), "gini": round(float(2 * auc - 1), 4), "ks": round(float(np.max(tpr - fpr)), 4),
            "brier": round(float(brier_score_loss(y, p)), 4), "approval_rate": round(float(approved.mean()), 4),
            "bad_rate_approved": round(float(y[approved].mean()), 4) if approved.any() else None,
            "best_iteration": int(booster.best_iteration),
        },
    }


def reference_stats(train_df: pd.DataFrame, booster: lgb.Booster) -> dict:
    """Децили признаков и PD на train — эталон для мониторинга дрейфа (PSI) в сервисе."""
    X = matrix(train_df)
    columns = {c: X[c].to_numpy() for c in DRIFT_FEATURES}
    columns["pd"] = booster.predict(X)
    ref = {}
    for name, values in columns.items():
        edges = np.unique(np.quantile(values, np.linspace(0.1, 0.9, 9)))
        shares = np.bincount(np.searchsorted(edges, values, side="right"), minlength=len(edges) + 1) / len(values)
        ref[name] = {"edges": edges.tolist(), "shares": shares.tolist()}
    return ref


def compare_with_production(new_metrics: dict, production_dir: Path, tolerance: float = 0.005) -> tuple[bool, str]:
    """Champion / challenger: новая модель выкатывается, если ROC-AUC не хуже текущей больше чем на tolerance."""
    meta_path = production_dir / "metadata.json"
    if not meta_path.exists():
        return True, "в проде нет модели"
    current = json.loads(meta_path.read_text(encoding="utf-8"))["metrics"]["roc_auc"]
    new = new_metrics["roc_auc"]
    if new + tolerance < current:
        return False, f"ROC-AUC {new:.4f} хуже текущей {current:.4f}"
    return True, f"ROC-AUC {new:.4f} против текущей {current:.4f}"


def data_hash(frame: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).values.tobytes()).hexdigest()


def save(booster: lgb.Booster, evaluation: dict, reference: dict, frame: pd.DataFrame, n_train: int, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    trained_at = datetime.now(timezone.utc)
    metadata = {
        "version": trained_at.strftime("%Y%m%d-%H%M%S"),
        "trained_at": trained_at.isoformat(timespec="seconds"),
        "data_sha256": data_hash(frame),
        "n_train": n_train,
        "features": FEATURES,
        "params": {k: v for k, v in PARAMS.items() if k not in ("verbose", "num_threads")},
        **evaluation,
        "reference": reference,
    }
    # model_to_string + запись из Python: C-API LightGBM на Windows не пишет в пути с не-ASCII символами
    (out_dir / "model.txt").write_text(booster.model_to_string(num_iteration=booster.best_iteration), encoding="utf-8")
    (out_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def log_mlflow(metadata: dict, model_dir: Path) -> None:
    """Логирование в MLflow, если задан MLFLOW_TRACKING_URI и пакет установлен."""
    if not os.getenv("MLFLOW_TRACKING_URI"):
        return
    import mlflow

    mlflow.set_experiment("credit-scoring")
    with mlflow.start_run(run_name=metadata["version"]):
        mlflow.log_params(metadata["params"])
        mlflow.log_metrics({k: v for k, v in metadata["metrics"].items() if isinstance(v, (int, float))})
        mlflow.log_metric("threshold", metadata["threshold"])
        mlflow.set_tag("data_sha256", metadata["data_sha256"])
        mlflow.log_artifacts(str(model_dir), artifact_path="model")
