"""
Загрузка модели и предсказание.

Артефакт — нативный текстовый формат LightGBM (model.txt) плюс metadata.json. Нативный формат
не зависит от версий sklearn/pickle и читается любой версией LightGBM.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import FEATURES, matrix

PDO, BASE_SCORE, BASE_ODDS = 20, 600, 50


def pd_to_score(p: float) -> int:
    """Балл скоркарты: 600 при шансах «хороший : плохой» = 50 : 1, каждые +20 баллов удваивают шансы."""
    p = min(max(p, 1e-6), 1 - 1e-6)
    return int(round(BASE_SCORE + PDO / math.log(2) * (math.log((1 - p) / p) - math.log(BASE_ODDS))))


@dataclass
class ModelService:
    booster: lgb.Booster
    metadata: dict

    @classmethod
    def load(cls, directory: str | Path) -> ModelService:
        directory = Path(directory)
        booster = lgb.Booster(model_str=(directory / "model.txt").read_text(encoding="utf-8"))
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if booster.feature_name() != FEATURES:
            raise ValueError("признаки модели не совпадают с app.features.FEATURES — артефакт от другой версии кода")
        return cls(booster, metadata)

    @property
    def version(self) -> str:
        return self.metadata["version"]

    @property
    def threshold(self) -> float:
        return float(self.metadata["threshold"])

    def predict(self, applications: pd.DataFrame, n_reasons: int = 3) -> list[dict]:
        X = matrix(applications)
        pd_ = self.booster.predict(X)
        contrib = self.booster.predict(X, pred_contrib=True)[:, :-1]  # последний столбец — базовое значение
        out = []
        for p, row in zip(pd_, contrib):
            top = np.argsort(-row)[:n_reasons]
            out.append({
                "pd": float(p),
                "score": pd_to_score(float(p)),
                "decision": "approve" if p < self.threshold else "reject",
                "threshold": self.threshold,
                "model_version": self.version,
                "reasons": [{"feature": FEATURES[i], "contribution": round(float(row[i]), 4)} for i in top if row[i] > 0],
            })
        return out
