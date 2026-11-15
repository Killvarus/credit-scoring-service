"""
Еженедельное переобучение скоринговой модели (Airflow, TaskFlow API).

extract → validate → train_and_evaluate → decide ─┬─ promote → (сервис подхватит при рестарте)
                                                  └─ skip
Шаги — функции из training/pipeline.py; промежуточные данные лежат в DATA_DIR, между задачами
передаются только пути и метрики (XCom не для датафреймов).
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

try:  # Airflow 3
    from airflow.sdk import dag, task
except ImportError:  # Airflow 2.x
    from airflow.decorators import dag, task

PROJECT = Path(os.getenv("PROJECT_DIR", "/opt/project"))
DATA_DIR = Path(os.getenv("DATA_DIR", "/opt/data"))
PRODUCTION_DIR = Path(os.getenv("MODEL_DIR", PROJECT / "artifacts"))
sys.path.insert(0, str(PROJECT))


@dag(schedule="@weekly", start_date=datetime(2026, 1, 1), catchup=False, tags=["ml", "credit-scoring"],
     default_args={"retries": 1})
def credit_scoring_retrain():
    @task
    def extract() -> str:
        from training import pipeline as P

        path = DATA_DIR / "raw" / f"credit_{datetime.now():%Y%m%d}.csv"
        P.extract(cache=path)
        return str(path)

    @task
    def validate(path: str) -> str:
        import pandas as pd

        from training import pipeline as P

        P.validate(pd.read_csv(path))  # DataValidationError роняет задачу — дальше не идём
        return path

    @task
    def train_and_evaluate(path: str) -> dict:
        import pandas as pd

        from training import pipeline as P

        frame = pd.read_csv(path)
        train_df, valid_df, test_df = P.split(frame)
        booster = P.train(train_df, valid_df)
        evaluation = P.evaluate(booster, valid_df, test_df)
        candidate = DATA_DIR / "candidates" / datetime.now().strftime("%Y%m%d-%H%M%S")
        P.save(booster, evaluation, P.reference_stats(train_df, booster), frame, len(train_df), candidate)
        return {"candidate": str(candidate), "metrics": evaluation["metrics"]}

    @task.branch
    def decide(result: dict) -> str:
        from training import pipeline as P

        promote, reason = P.compare_with_production(result["metrics"], PRODUCTION_DIR)
        print(reason)
        return "promote" if promote else "skip"

    @task
    def promote(result: dict) -> None:
        import json
        import shutil

        from training import pipeline as P

        candidate = Path(result["candidate"])
        P.log_mlflow(json.loads((candidate / "metadata.json").read_text(encoding="utf-8")), candidate)
        for name in ("model.txt", "metadata.json"):
            shutil.copy2(candidate / name, PRODUCTION_DIR / name)

    @task
    def skip() -> None:
        print("кандидат хуже текущей модели — оставляем прод как есть")

    result = train_and_evaluate(validate(extract()))
    decide(result) >> [promote(result), skip()]


credit_scoring_retrain()
