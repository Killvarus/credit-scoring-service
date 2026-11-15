"""
Полный цикл переобучения из командной строки.

    python -m training.train                       # данные с OpenML → artifacts/ (если не хуже текущей)
    python -m training.train --out /tmp/candidate --force
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from . import pipeline as P

ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger("training")


def run(out_dir: Path, production_dir: Path, cache: Path | None, force: bool = False) -> dict:
    frame = P.extract(cache)
    log.info("validate: %s", P.validate(frame))
    train_df, valid_df, test_df = P.split(frame)
    booster = P.train(train_df, valid_df)
    evaluation = P.evaluate(booster, valid_df, test_df)
    log.info("metrics: %s", evaluation["metrics"])
    promote, reason = P.compare_with_production(evaluation["metrics"], production_dir)
    if not (promote or force):
        log.warning("модель не выкатывается: %s", reason)
        return {"promoted": False, "reason": reason, **evaluation}
    metadata = P.save(booster, evaluation, P.reference_stats(train_df, booster), frame, len(train_df), out_dir)
    P.log_mlflow(metadata, out_dir)
    log.info("сохранено в %s (%s)", out_dir, reason)
    return {"promoted": True, "reason": reason, "version": metadata["version"], **evaluation}


def main():
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--production", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--cache", type=Path, default=ROOT / "data" / "credit_default.csv")
    parser.add_argument("--force", action="store_true", help="сохранить даже если хуже текущей")
    args = parser.parse_args()
    print(json.dumps(run(args.out, args.production, args.cache, args.force), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
