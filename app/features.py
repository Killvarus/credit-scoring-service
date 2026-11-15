"""
Признаки заявки — та же логика, что в исследовательском репозитории credit-scoring.

Один модуль используют и обучение (training/), и сервис (app/), и пакетный скоринг (batch/):
так признаки на обучении и в проде гарантированно совпадают (нет training-serving skew).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PAY_COLS = [f"PAY_{i}" for i in range(1, 7)]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYAMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]
CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE"]
RAW_COLUMNS = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", *PAY_COLS, *BILL_COLS, *PAYAMT_COLS]
ENGINEERED = [
    *[f"util_{i}" for i in range(1, 7)], "util_mean", "util_max", "util_trend",
    "pay_ratio_1", "pay_ratio_mean", "pay_ratio_min",
    "delay_max", "delay_count", "delay_recent2", "months_since_delay",
    "n_revolving", "n_paid_full", "n_no_usage",
    "spend_mean_rel", "pay_mean_rel", "log_limit",
]
FEATURES = [*RAW_COLUMNS, *ENGINEERED]


def clean(frame: pd.DataFrame) -> pd.DataFrame:
    """Недокументированные категории UCI сводим к «другое»."""
    frame = frame.copy()
    frame["EDUCATION"] = frame["EDUCATION"].where(frame["EDUCATION"].between(1, 4), 4)
    frame["MARRIAGE"] = frame["MARRIAGE"].where(frame["MARRIAGE"].between(1, 3), 3)
    return frame


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = clean(frame)
    limit = out["LIMIT_BAL"].astype(float)
    bills = out[BILL_COLS].to_numpy(float)
    pays = out[PAYAMT_COLS].to_numpy(float)
    status = out[PAY_COLS].to_numpy(float)

    util = bills / limit.to_numpy()[:, None]
    for i in range(6):
        out[f"util_{i + 1}"] = util[:, i]
    out["util_mean"] = util.mean(1)
    out["util_max"] = util.max(1)
    out["util_trend"] = util[:, 0] - util[:, 5]

    prev_bill = np.clip(bills[:, 1:], 1.0, None)
    pay_ratio = np.clip(pays[:, :5] / prev_bill, 0, 2)
    pay_ratio[bills[:, 1:] <= 0] = 1.0
    out["pay_ratio_1"] = pay_ratio[:, 0]
    out["pay_ratio_mean"] = pay_ratio.mean(1)
    out["pay_ratio_min"] = pay_ratio.min(1)

    delay = np.clip(status, 0, None)
    has_delay = status >= 1
    out["delay_max"] = delay.max(1)
    out["delay_count"] = has_delay.sum(1)
    out["delay_recent2"] = delay[:, :2].sum(1)
    out["months_since_delay"] = np.where(has_delay.any(1), has_delay.argmax(1), 7)
    out["n_revolving"] = (status == 0).sum(1)
    out["n_paid_full"] = (status == -1).sum(1)
    out["n_no_usage"] = (status == -2).sum(1)

    spend = bills[:, :5] - bills[:, 1:] + pays[:, :5]
    out["spend_mean_rel"] = spend.mean(1) / limit
    out["pay_mean_rel"] = pays.mean(1) / limit
    out["log_limit"] = np.log(limit)
    return out


def matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Сырые колонки заявки → матрица признаков модели в фиксированном порядке."""
    return add_features(frame)[FEATURES].astype(float)
