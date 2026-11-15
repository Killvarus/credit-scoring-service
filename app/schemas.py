"""Схемы запросов и ответов. Pydantic отклоняет некорректные заявки до модели (HTTP 422)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

PayStatus = Field(ge=-2, le=9, description="-2 нет трат, -1 оплачено полностью, 0 минимальный платёж, k — просрочка k мес.")

EXAMPLE = {
    "LIMIT_BAL": 120000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 2, "AGE": 26,
    "PAY_1": -1, "PAY_2": 2, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 2,
    "BILL_AMT1": 2682, "BILL_AMT2": 1725, "BILL_AMT3": 2682, "BILL_AMT4": 3272, "BILL_AMT5": 3455, "BILL_AMT6": 3261,
    "PAY_AMT1": 0, "PAY_AMT2": 1000, "PAY_AMT3": 1000, "PAY_AMT4": 1000, "PAY_AMT5": 0, "PAY_AMT6": 2000,
}


class Application(BaseModel):
    """Заявка в формате UCI Default of Credit Card Clients. Индекс 1 — последний месяц."""

    model_config = ConfigDict(json_schema_extra={"example": EXAMPLE}, extra="forbid")

    LIMIT_BAL: float = Field(gt=0, le=10_000_000, description="кредитный лимит")
    SEX: int = Field(ge=1, le=2)
    EDUCATION: int = Field(ge=0, le=6)
    MARRIAGE: int = Field(ge=0, le=3)
    AGE: int = Field(ge=18, le=100)
    PAY_1: int = PayStatus
    PAY_2: int = PayStatus
    PAY_3: int = PayStatus
    PAY_4: int = PayStatus
    PAY_5: int = PayStatus
    PAY_6: int = PayStatus
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float = Field(ge=0)
    PAY_AMT2: float = Field(ge=0)
    PAY_AMT3: float = Field(ge=0)
    PAY_AMT4: float = Field(ge=0)
    PAY_AMT5: float = Field(ge=0)
    PAY_AMT6: float = Field(ge=0)


class Reason(BaseModel):
    feature: str
    contribution: float = Field(description="вклад в логит PD (TreeSHAP); > 0 повышает риск")


class Prediction(BaseModel):
    pd: float = Field(description="вероятность дефолта в следующем месяце")
    score: int = Field(description="балл: 600 при шансах 50:1, +20 баллов — шансы вдвое лучше")
    decision: Literal["approve", "reject"]
    threshold: float
    model_version: str
    reasons: list[Reason] = Field(description="три признака, сильнее всего повышающие риск")


class BatchRequest(BaseModel):
    applications: list[Application] = Field(min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    predictions: list[Prediction]
