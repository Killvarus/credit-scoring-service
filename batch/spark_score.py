"""
Пакетный скоринг на Spark: миллионы заявок за ночь, те же признаки и модель, что в API.

    spark-submit batch/spark_score.py --input applications.csv --output scored/ [--model artifacts]

mapInPandas отдаёт каждому исполнителю куски таблицы как pandas DataFrame: внутри работает тот же
app.features + LightGBM, что и в онлайне. Модель читается на исполнителе один раз на партицию.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def score_partitions(model_dir: str):
    def run(batches):
        from app.model import ModelService

        service = ModelService.load(model_dir)
        for batch in batches:
            preds = service.predict(batch, n_reasons=0)
            yield batch.assign(pd=[p["pd"] for p in preds], score=[p["score"] for p in preds],
                               decision=[p["decision"] for p in preds])
    return run


def main():
    from pyspark.sql import SparkSession

    from app.features import RAW_COLUMNS

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=str(ROOT / "artifacts"))
    args = parser.parse_args()

    spark = SparkSession.builder.appName("credit-scoring-batch").getOrCreate()
    frame = spark.read.csv(args.input, header=True, inferSchema=True).select(*RAW_COLUMNS)
    schema = frame.schema.add("pd", "double").add("score", "long").add("decision", "string")
    scored = frame.mapInPandas(score_partitions(args.model), schema=schema)
    scored.write.mode("overwrite").parquet(args.output)
    summary = scored.groupBy("decision").count().collect()
    print({row["decision"]: row["count"] for row in summary})
    spark.stop()


if __name__ == "__main__":
    main()
